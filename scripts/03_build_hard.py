"""Stage 3: HARD eval-awareness task -- honest, non-saturated numbers.

Builds the harder, genuinely-overlapping dataset (cnr.data_hard), extracts GPT-2
layer-6 mean-pooled activations (cnr.harness.ModelHarness), trains the ReLU probe
(cnr.probe) across several seeds, and reports REAL metrics with error bars:

    * dataset size + overlap stats (hard-neg / hard-pos / ambiguous-flip counts)
    * accuracy  mean +/- std   over seeds (held-out test split)
    * AUROC     mean +/- std   over seeds (sklearn roc_auc_score, held-out)
    * certified radius mean +/- std over >=6 benign anchors x seeds
      (cnr.verify.certified_radius), plus a certified-radius-vs-eps curve

Everything is written under artifacts/v2/ (new dir) and a self-contained report
to reports/hard_<timestamp>.html. This script CREATES only; it never modifies or
overwrites existing files/artifacts.

Run from repo root:
    HF_HUB_OFFLINE=1 python scripts/03_build_hard.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from cnr.config import Config  # noqa: E402
from cnr.data_hard import build_hard_dataset, save_texts  # noqa: E402
from cnr.harness import ModelHarness  # noqa: E402
from cnr.probe import train_probe  # noqa: E402
from cnr.verify import (  # noqa: E402
    AUTO_LIRPA_AVAILABLE,
    CertifiedVerifier,
    certified_radius,
)

# --------------------------------------------------------------------------- #
# Knobs (the difficulty was calibrated so test AUROC lands ~0.82-0.85, i.e.
# hard but learnable and well under 0.95; see reports/hard_*.html for the sweep).
# --------------------------------------------------------------------------- #
DIFFICULTY = 0.55
N_PER_CLASS = 256
SEEDS = [0, 1, 2, 3, 4]            # >= 5 seeds for mean +/- std
N_ANCHORS = 6                      # >= 6 benign deployment anchors
TEST_FRAC = 0.30                   # held-out split for honest AUROC/accuracy
EPS_GRID = np.linspace(0.02, 2.0, 25)
EPS_CURVE = np.linspace(0.02, 1.5, 16)   # for the certified-fraction-vs-eps curve

V2 = Path(__file__).resolve().parent.parent / "artifacts" / "v2"


def split(n: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = int(round((1.0 - TEST_FRAC) * n))
    return perm[:ntr], perm[ntr:]


def main() -> None:
    t_start = time.time()
    V2.mkdir(parents=True, exist_ok=True)
    cfg = Config()

    print("=" * 78)
    print("STAGE 3  --  HARD eval-awareness task (honest, non-saturated)")
    print(f"  model={cfg.model_name}  layer={cfg.layer}  pooling={cfg.pooling}  device={cfg.device}")
    print(f"  difficulty={DIFFICULTY}  n_per_class={N_PER_CLASS}  seeds={SEEDS}")
    print("=" * 78)

    print(f"[1/5] loading {cfg.model_name}")
    harness = ModelHarness(cfg)

    # The dataset text/labels depend on the seed (noisy ambiguous labels), so we
    # extract activations per seed. Activations for the *same* texts are
    # deterministic; different seeds give different label assignments + sampling.
    per_seed = []
    overlap_ref = None
    print(f"[2/5] building hard dataset + extracting activations across {len(SEEDS)} seeds")
    for sd in SEEDS:
        texts, labels, info = build_hard_dataset(
            N_PER_CLASS, seed=sd, difficulty=DIFFICULTY
        )
        X = harness.activations(texts).astype(np.float32)
        y = np.array(labels, dtype=np.int64)
        per_seed.append({"seed": sd, "texts": texts, "X": X, "y": y, "info": info})
        if overlap_ref is None:
            overlap_ref = info
        print(f"      seed {sd}: n={info['n_total']}  "
              f"label1/0={info['n_label1']}/{info['n_label0']}  "
              f"ambiguous={info['n_ambiguous']} (flipped {info['amb_flipped']})  "
              f"hardpos={info['eval_hard']} hardneg={info['deploy_hard']}")

    # save one representative dataset (seed 0) for inspection / tests
    save_texts(V2 / "hard_texts.json",
               per_seed[0]["texts"], per_seed[0]["y"].tolist(), per_seed[0]["info"])
    np.savez(V2 / "hard_bank.npz",
             X=per_seed[0]["X"], y=per_seed[0]["y"])

    # ----------------------------------------------------------------------- #
    # [3/5] train probe per seed, held-out accuracy + AUROC
    # ----------------------------------------------------------------------- #
    print(f"[3/5] training ReLU probe (hidden={cfg.hidden}) per seed; held-out AUROC/acc")
    accs, aurocs = [], []
    probes = []
    for s in per_seed:
        X, y, sd = s["X"], s["y"], s["seed"]
        tr, te = split(len(y), sd)
        probe, meta = train_probe(
            X[tr], y[tr], hidden=cfg.hidden, epochs=cfg.probe_epochs,
            lr=cfg.probe_lr, seed=sd,
        )
        with torch.no_grad():
            s_te = probe(torch.tensor(X[te], dtype=torch.float32)).numpy()
        pred = (s_te >= meta["tau"]).astype(int)
        acc = float((pred == y[te]).mean())
        auc = float(roc_auc_score(y[te], s_te))
        accs.append(acc)
        aurocs.append(auc)
        probes.append({"seed": sd, "probe": probe, "meta": meta,
                       "test_idx": te.tolist()})
        print(f"      seed {sd}: test_acc={acc:.3f}  test_AUROC={auc:.3f}  "
              f"(n_test={len(te)})")

    accs = np.array(accs)
    aurocs = np.array(aurocs)
    print(f"      --> accuracy  {accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"      --> AUROC     {aurocs.mean():.3f} +/- {aurocs.std():.3f}")
    saturated = aurocs.mean() >= 0.95
    print(f"      --> AUROC < 0.95 ? {'NO (still too easy!)' if saturated else 'YES (good, non-saturated)'}")

    # ----------------------------------------------------------------------- #
    # [4/5] certified radius over >=6 benign anchors x seeds + eps curve
    # ----------------------------------------------------------------------- #
    print(f"[4/5] certified radius via cnr.verify over {N_ANCHORS} benign anchors x {len(SEEDS)} seeds")
    print(f"      bound method: {'crown (auto_LiRPA)' if AUTO_LIRPA_AVAILABLE else 'crown (native CROWN+IBP)'}")
    radii_all = []          # flat list of certified radii (anchor x seed)
    radii_by_seed = []      # per-seed mean
    # certified-fraction-vs-eps curve: fraction of (anchor x seed) certified at each eps
    cert_fraction = np.zeros(len(EPS_CURVE), dtype=float)
    cert_counts = np.zeros(len(EPS_CURVE), dtype=int)
    n_pairs = 0

    for pinfo, s in zip(probes, per_seed):
        probe, meta = pinfo["probe"], pinfo["meta"]
        X, y = s["X"], s["y"]
        sigma = np.asarray(meta["sigma"], dtype=np.float64)
        tau = float(meta["tau"])
        verifier = CertifiedVerifier(probe, sigma, norm="inf", method="auto")

        # benign anchors = deployment (label 0) activations, first N_ANCHORS
        deploy_ids = np.where(y == 0)[0][:N_ANCHORS]
        seed_radii = []
        for aid in deploy_ids:
            h0 = X[aid].astype(np.float64)
            cr = certified_radius(verifier, h0, tau, EPS_GRID)
            seed_radii.append(cr)
            radii_all.append(cr)
            n_pairs += 1
            # accumulate certified-fraction curve
            for j, e in enumerate(EPS_CURVE):
                res = verifier.certify(h0, float(e), tau)
                if res.certified_unreachable:
                    cert_counts[j] += 1
        radii_by_seed.append(float(np.mean(seed_radii)))
        print(f"      seed {s['seed']}: mean certified radius over {N_ANCHORS} anchors "
              f"= {np.mean(seed_radii):.3f}  (min {np.min(seed_radii):.3f}, max {np.max(seed_radii):.3f})")

    radii_all = np.array(radii_all)
    radii_by_seed = np.array(radii_by_seed)
    cert_fraction = cert_counts / max(1, n_pairs)
    print(f"      --> certified radius (all anchor x seed) "
          f"{radii_all.mean():.3f} +/- {radii_all.std():.3f}  (n={len(radii_all)})")
    print(f"      --> certified radius (per-seed means) "
          f"{radii_by_seed.mean():.3f} +/- {radii_by_seed.std():.3f}")

    # ----------------------------------------------------------------------- #
    # [5/5] save arrays/json + render report
    # ----------------------------------------------------------------------- #
    print("[5/5] saving artifacts to artifacts/v2/ and rendering report")
    results = {
        "config": {
            "model": cfg.model_name, "layer": cfg.layer, "pooling": cfg.pooling,
            "hidden": cfg.hidden, "probe_epochs": cfg.probe_epochs,
            "probe_lr": cfg.probe_lr, "difficulty": DIFFICULTY,
            "n_per_class": N_PER_CLASS, "seeds": SEEDS, "test_frac": TEST_FRAC,
            "n_anchors": N_ANCHORS,
            "bound_method": "crown (auto_LiRPA)" if AUTO_LIRPA_AVAILABLE else "crown (native CROWN+IBP)",
            "auto_lirpa_available": AUTO_LIRPA_AVAILABLE,
        },
        "overlap_example_seed0": overlap_ref,
        "per_seed_acc": accs.tolist(),
        "per_seed_auroc": aurocs.tolist(),
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
        "auroc_mean": float(aurocs.mean()), "auroc_std": float(aurocs.std()),
        "auroc_below_0_95": bool(not saturated),
        "radii_all": radii_all.tolist(),
        "radius_mean": float(radii_all.mean()), "radius_std": float(radii_all.std()),
        "radii_by_seed_mean": radii_by_seed.tolist(),
        "radius_by_seed_mean_mean": float(radii_by_seed.mean()),
        "radius_by_seed_mean_std": float(radii_by_seed.std()),
        "eps_curve": EPS_CURVE.tolist(),
        "cert_fraction_curve": cert_fraction.tolist(),
        "n_cert_pairs": int(n_pairs),
        "wall_time_s": round(time.time() - t_start, 1),
    }
    with open(V2 / "hard_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(
        V2 / "hard_metrics.npz",
        per_seed_acc=accs, per_seed_auroc=aurocs,
        radii_all=radii_all, radii_by_seed_mean=radii_by_seed,
        eps_curve=EPS_CURVE, cert_fraction_curve=cert_fraction,
    )

    png_rel = _save_curve_png(V2, EPS_CURVE, cert_fraction, EPS_GRID, radii_all)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(cfg.reports) / f"hard_{ts}.html"
    _write_report(report_path, results, png_rel)

    print("-" * 78)
    print("DONE.")
    print(f"  artifacts: {V2}")
    print(f"  report:    {report_path}")
    print("-" * 78)
    print("SUMMARY (honest, non-saturated):")
    print(f"  dataset: n={overlap_ref['n_total']} per seed, "
          f"ambiguous(noisy)={overlap_ref['n_ambiguous']}, "
          f"hard-neg={overlap_ref['deploy_hard']}, hard-pos={overlap_ref['eval_hard']}, "
          f"label_noise={overlap_ref['label_noise']:.2f}")
    print(f"  accuracy  = {accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"  AUROC     = {aurocs.mean():.3f} +/- {aurocs.std():.3f}   "
          f"(target 0.75-0.92; AUROC<0.95: {not saturated})")
    print(f"  cert.radius = {radii_all.mean():.3f} +/- {radii_all.std():.3f} "
          f"(std units, {len(radii_all)} anchor x seed)")
    return 0 if not saturated else 1


def _save_curve_png(v2: Path, eps_curve, cert_fraction, eps_grid, radii_all):
    """Save the certified-fraction-vs-eps curve + radius histogram as a PNG.

    Returns the basename (for embedding in the report). Falls back to '' if
    matplotlib is unavailable (the report then uses an inline SVG)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(eps_curve, cert_fraction, "o-", color="#2b6cb0", lw=2)
    ax[0].set_xlabel("eps  (per-dim L-inf radius, std units)")
    ax[0].set_ylabel("fraction certified unreachable")
    ax[0].set_title("Certified fraction vs eps")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].grid(alpha=0.3)
    ax[1].hist(radii_all, bins=12, color="#38a169", alpha=0.85, edgecolor="white")
    ax[1].axvline(float(np.mean(radii_all)), color="#c53030", ls="--",
                  label=f"mean={np.mean(radii_all):.3f}")
    ax[1].set_xlabel("certified radius (std units)")
    ax[1].set_ylabel("count (anchor x seed)")
    ax[1].set_title("Certified-radius distribution")
    ax[1].legend()
    fig.tight_layout()
    out = v2 / "hard_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out.name


def _svg_curve(eps_curve, cert_fraction) -> str:
    """Tiny inline-SVG line chart fallback (no matplotlib needed)."""
    W, H, pad = 420, 220, 40
    xs = np.asarray(eps_curve, float)
    ys = np.asarray(cert_fraction, float)
    x0, x1 = xs.min(), xs.max()
    def px(x): return pad + (x - x0) / (x1 - x0 + 1e-9) * (W - 2 * pad)
    def py(y): return H - pad - y * (H - 2 * pad)
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    ticks = ""
    for f in (0.0, 0.5, 1.0):
        ticks += (f'<line x1="{pad}" y1="{py(f):.1f}" x2="{W-pad}" y2="{py(f):.1f}" '
                  f'stroke="#eee"/><text x="4" y="{py(f)+4:.1f}" font-size="10">{f:.1f}</text>')
    return (f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">'
            f'{ticks}'
            f'<polyline fill="none" stroke="#2b6cb0" stroke-width="2" points="{pts}"/>'
            f'<text x="{W/2:.0f}" y="{H-6}" font-size="11" text-anchor="middle">'
            f'eps (std units)</text>'
            f'<text x="12" y="16" font-size="11">fraction certified</text></svg>')


def _write_report(path: Path, r: dict, png_name: str) -> None:
    cfg = r["config"]
    ov = r["overlap_example_seed0"]
    badge = ("#2f855a" if r["auroc_below_0_95"] else "#c53030")
    badge_txt = ("non-saturated (AUROC &lt; 0.95)" if r["auroc_below_0_95"]
                 else "STILL SATURATED (AUROC &ge; 0.95) -- make it harder")
    img_block = (f'<img src="../artifacts/v2/{png_name}" style="max-width:100%;border:1px solid #ddd;border-radius:6px">'
                 if png_name else
                 f'<div>{_svg_curve(r["eps_curve"], r["cert_fraction_curve"])}</div>')

    # per-seed table
    rows = ""
    for sd, a, u in zip(cfg["seeds"], r["per_seed_acc"], r["per_seed_auroc"]):
        rows += f"<tr><td>{sd}</td><td>{a:.3f}</td><td>{u:.3f}</td></tr>"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Hard eval-awareness task -- certified reachability</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
   max-width:900px;margin:32px auto;padding:0 18px;color:#1a202c;line-height:1.5}}
 h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #eee;padding-bottom:4px}}
 .badge{{display:inline-block;background:{badge};color:#fff;padding:3px 10px;border-radius:12px;font-size:13px}}
 table{{border-collapse:collapse;margin:8px 0}} td,th{{border:1px solid #e2e8f0;padding:5px 12px;text-align:right}}
 th{{background:#f7fafc}} .k{{color:#4a5568}} code{{background:#f1f5f9;padding:1px 4px;border-radius:3px}}
 .big{{font-size:20px;font-weight:600}} .note{{color:#4a5568;font-size:13px}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
 .card{{background:#f7fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px}}
</style></head><body>
<h1>Hard eval-awareness task &mdash; honest certified-reachability numbers</h1>
<p><span class="badge">{badge_txt}</span>
 &nbsp;<span class="note">generated {datetime.now():%Y-%m-%d %H:%M} &middot; wall {r['wall_time_s']}s</span></p>

<p class="note">Pipeline: GPT-2 residual <b>layer {cfg['layer']}</b>, mean-pooled &rarr;
ReLU probe (hidden {cfg['hidden']}) = target set A* = {{h : g(h) &ge; &tau;=0}} &rarr;
sound CROWN/IBP certificate (<code>{cfg['bound_method']}</code>) over a per-dim
L&infin; ball of radius eps&middot;&sigma;<sub>i</sub>. The point of this page is that the
probe metric is <b>not saturated</b>, so the certificate is reported against a
genuinely hard target.</p>

<h2>Headline numbers (mean &plusmn; std over {len(cfg['seeds'])} seeds)</h2>
<div class="grid">
 <div class="card">accuracy (held-out)<br><span class="big">{r['acc_mean']:.3f} &plusmn; {r['acc_std']:.3f}</span></div>
 <div class="card">AUROC (held-out)<br><span class="big">{r['auroc_mean']:.3f} &plusmn; {r['auroc_std']:.3f}</span>
   <div class="note">target 0.75&ndash;0.92</div></div>
 <div class="card">certified radius (std units)<br>
   <span class="big">{r['radius_mean']:.3f} &plusmn; {r['radius_std']:.3f}</span>
   <div class="note">{len(r['radii_all'])} anchor&times;seed; per-seed-mean
     {r['radius_by_seed_mean_mean']:.3f} &plusmn; {r['radius_by_seed_mean_std']:.3f}</div></div>
 <div class="card">bound method<br><span class="big" style="font-size:15px">{cfg['bound_method']}</span>
   <div class="note">sound upper bound; soundness witnessed by sampling at every certify()</div></div>
</div>

<h2>Certified radius vs eps</h2>
<p class="note">Left: fraction of (anchor&times;seed) pairs whose eval-aware region is
<b>provably unreachable</b> within radius eps. Right: distribution of per-anchor
certified radii. At eps below the certified radius, no per-dim perturbation of
that size can push the benign activation into A*.</p>
{img_block}

<h2>Dataset &mdash; how the overlap is built (example seed)</h2>
<p class="note">An overlapping classification problem must have non-zero Bayes error.
On a near-lossless frozen GPT-2 readout, deterministic word&rarr;label rules are
learnable to AUROC 1.00 (marker words like &ldquo;graded&rdquo; survive mean-pooling),
so overlap is created by a pool of genuinely <b>ambiguous</b> prompts whose labels
carry noise (<code>label_noise={ov['label_noise']:.2f}</code>), plus clearly-labeled
hard negatives (test-vocab user requests) and hard positives (casually-phrased eval
scaffolding).</p>
<table>
 <tr><th>quantity</th><th>count</th></tr>
 <tr><td class="k">total examples / seed</td><td>{ov['n_total']}</td></tr>
 <tr><td class="k">label 1 / label 0</td><td>{ov['n_label1']} / {ov['n_label0']}</td></tr>
 <tr><td class="k">ambiguous (noisy-label)</td><td>{ov['n_ambiguous']}</td></tr>
 <tr><td class="k">&nbsp; of which label-flipped</td><td>{ov['amb_flipped']}</td></tr>
 <tr><td class="k">hard positives (eval, casual)</td><td>{ov['eval_hard']}</td></tr>
 <tr><td class="k">hard negatives (deploy, test-vocab)</td><td>{ov['deploy_hard']}</td></tr>
 <tr><td class="k">easy eval / easy deploy</td><td>{ov['eval_easy']} / {ov['deploy_easy']}</td></tr>
</table>

<h2>Per-seed metrics</h2>
<table><tr><th>seed</th><th>test acc</th><th>test AUROC</th></tr>{rows}</table>

<h2>Honesty / scope</h2>
<ul class="note">
 <li>Synthetic text; a deliberately harder <b>stand-in</b> for eval-awareness, not
   ground-truth model cognition.</li>
 <li>The only <b>sound</b> objects are the verifier's CROWN/IBP upper bounds; every
   <code>certify()</code> call checks U &ge; sampled max or raises.</li>
 <li>AUROC is held-out (30% test split), reported with std over
   {len(cfg['seeds'])} seeds. The difficulty knob was calibrated on a separate
   sweep to avoid saturation.</li>
</ul>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    sys.exit(main())
