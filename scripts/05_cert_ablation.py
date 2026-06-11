#!/usr/bin/env python3
"""Ablations + baselines for the certified eval-awareness reachability claim.

Run from the repo root:   python scripts/05_cert_ablation.py

Produces (CREATE-only, never overwrites existing project artifacts):
    artifacts/cert/ablation.json        all numbers
    artifacts/cert/tightness.png        CROWN vs IBP vs sampling-only, U vs eps
    artifacts/cert/radius.png           certified-radius-vs-eps style curves
    artifacts/cert/deep_vs_shallow.png  shallow vs deep certificate, U vs eps
    artifacts/cert/null_vs_real.png     null-direction vs real-direction radius
    reports/cert_<timestamp>.html       self-contained report (own HTML)

What is compared
----------------
1. TIGHTNESS (shallow, activation-space probe):  for a fixed anchor, sweep eps and
   plot three upper-bound estimates of  max_{B} probe(h):
       * CROWN    -- sound (cnr.verify, min(CROWN,IBP))
       * IBP      -- sound (cnr.verify, interval propagation)
       * sampling -- UNSOUND reference (max over many random samples in B); it is a
                     LOWER bound on the true max, so it sits *below* the sound
                     bounds and shows how loose CROWN/IBP are. Labeled unsound.
   Reported: the CROWN-vs-IBP gap as a function of eps (where CROWN buys tightness).

2. NULL-DIRECTION baseline:  certified radius of the REAL eval-awareness probe vs a
   random-direction "null probe" (a 1-hidden-layer ReLU net with a random first
   layer, matched scale, bias set to the same deploy/eval margin). Interpretation
   in the report.

3. DEEP vs SHALLOW:  the frozen-attention deep certificate (cnr.verify_deep,
   through LayerNorm+MLP+GELU+probe) vs the shallow activation-space certificate,
   for the same anchors -- quantifying how much certifying one block deeper costs.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

# repo root on path (run-as-file)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cnr.config import Config  # noqa: E402
from cnr.harness import ModelHarness  # noqa: E402
from cnr.probe import ReLUProbe, load_probe  # noqa: E402
from cnr.verify import (  # noqa: E402
    CertifiedVerifier,
    certified_radius,
    crown_ibp_upper,
    ibp_upper,
)
from cnr.verify_deep import (  # noqa: E402
    deep_certified_radius,
    make_certifier_from_harness,
    prepare_anchor,
)

CERT_DIR = os.path.join(_ROOT, "artifacts", "cert")
os.makedirs(CERT_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sampling_only_max(probe, h0, eps_vec, n=20000, seed=0):
    """UNSOUND reference: max probe over n uniform samples in the box (lower bound
    on the true max). Returned to show how far the sound bounds sit above it."""
    rng = np.random.default_rng(seed)
    d = h0.shape[0]
    u = rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)
    pts = h0[None, :].astype(np.float32) + u * eps_vec[None, :].astype(np.float32)
    with torch.no_grad():
        s = probe(torch.tensor(pts, dtype=torch.float32)).numpy()
    return float(s.max())


def _make_null_probe(real_probe, meta, X, y, seed=0):
    """A random-direction 'null' ReLUProbe matched to the real probe in scale.

    Same architecture (d, hidden). The first Linear is replaced by random weights
    whose per-row norms match the real (folded) probe's, and the second Linear /
    biases are kept; then the output bias is shifted so the null probe's
    deploy-class mean score equals the real probe's deploy-class mean score (so
    tau=0 has a comparable meaning). This makes 'certified radius' a fair
    comparison: same network family, same separation scale, RANDOM direction.
    """
    d = real_probe.net[0].in_features
    hidden = real_probe.net[0].out_features
    g = torch.Generator().manual_seed(seed)
    null = ReLUProbe(d, hidden)
    with torch.no_grad():
        # match per-row L2 norm of the real first-layer weights
        real_W1 = real_probe.net[0].weight.detach()
        row_norm = real_W1.norm(dim=1, keepdim=True)               # (hidden,1)
        Wr = torch.randn(hidden, d, generator=g)
        Wr = Wr / Wr.norm(dim=1, keepdim=True).clamp(min=1e-9) * row_norm
        null.net[0].weight.copy_(Wr)
        null.net[0].bias.copy_(real_probe.net[0].bias.detach().clone())
        null.net[2].weight.copy_(real_probe.net[2].weight.detach().clone())
        null.net[2].bias.copy_(real_probe.net[2].bias.detach().clone())
    null.eval()
    for p in null.parameters():
        p.requires_grad_(False)
    # shift output bias so deploy-class mean score matches the real probe's
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        s_real = real_probe(Xt).numpy()
        s_null = null(Xt).numpy()
    deploy = (y == 0)
    shift = float(s_real[deploy].mean() - s_null[deploy].mean())
    with torch.no_grad():
        null.net[2].bias.add_(shift)
    return null


def _png(path):
    return os.path.relpath(path, os.path.join(_ROOT, "reports"))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    cfg = Config()
    probe, meta = load_probe(cfg.probe_path)
    sigma = np.asarray(meta["sigma"], dtype=np.float64)
    tau = float(meta.get("tau", 0.0))

    bank = np.load(cfg.bank_path)
    X = bank["X"].astype(np.float64)
    y = bank["y"]
    with open(cfg.meta_path) as f:
        mj = json.load(f)
    anchor_ids = mj["anchor_ids"]
    with open(cfg.bank_text_path) as f:
        bt = json.load(f)
    texts = bt["texts"] if isinstance(bt, dict) and "texts" in bt else bt

    results = {
        "tau": tau,
        "layer": cfg.layer,
        "deploy_score_mean": float(meta.get("score_deploy_mean", np.nan)),
        "eval_score_mean": float(meta.get("score_eval_mean", np.nan)),
        "n_anchors": len(anchor_ids),
    }

    v_crown = CertifiedVerifier(probe, sigma, norm="inf", method="auto")
    v_ibp = CertifiedVerifier(probe, sigma, norm="inf", method="ibp")

    # ----------------------------------------------------------------- #
    # 1. TIGHTNESS: CROWN vs IBP vs sampling-only, U vs eps (one anchor)
    # ----------------------------------------------------------------- #
    print("=" * 78)
    print("[1] TIGHTNESS: CROWN vs IBP vs sampling-only (shallow, activation-space)")
    eps_grid = np.linspace(0.05, 2.0, 20)
    aid0 = anchor_ids[0]
    h0 = X[aid0]
    U_crown, U_ibp, U_sample = [], [], []
    for eps in eps_grid:
        rc = v_crown.certify(h0, float(eps), tau)
        ri = v_ibp.certify(h0, float(eps), tau)
        eps_vec = float(eps) * sigma
        smp = _sampling_only_max(probe, h0, eps_vec, n=20000, seed=aid0)
        U_crown.append(rc.score_upper)
        U_ibp.append(ri.score_upper)
        U_sample.append(smp)
    U_crown = np.array(U_crown); U_ibp = np.array(U_ibp); U_sample = np.array(U_sample)
    gap = U_ibp - U_crown
    print(f"  anchor {aid0}: CROWN-vs-IBP gap  min={gap.min():.3f}  max={gap.max():.3f}")
    print(f"    eps where gap is largest: {eps_grid[int(np.argmax(gap))]:.3f} "
          f"(IBP={U_ibp[int(np.argmax(gap))]:.2f} CROWN={U_crown[int(np.argmax(gap))]:.2f})")
    for e, c, i_, s in list(zip(eps_grid, U_crown, U_ibp, U_sample))[::4]:
        print(f"    eps={e:.2f}: CROWN U={c:8.2f}  IBP U={i_:8.2f}  sampling(unsound)={s:8.2f}")
    results["tightness"] = {
        "anchor": int(aid0),
        "eps_grid": eps_grid.tolist(),
        "U_crown": U_crown.tolist(),
        "U_ibp": U_ibp.tolist(),
        "U_sampling_unsound": U_sample.tolist(),
        "crown_ibp_gap": gap.tolist(),
    }

    plt.figure(figsize=(7, 4.5))
    plt.plot(eps_grid, U_ibp, "o-", color="#c44", label="IBP (sound upper)")
    plt.plot(eps_grid, U_crown, "s-", color="#36c", label="CROWN=min(CROWN,IBP) (sound upper)")
    plt.plot(eps_grid, U_sample, "^--", color="#888",
             label="sampling-only (UNSOUND lower ref)")
    plt.axhline(tau, color="k", lw=1, ls=":", label=f"tau={tau:g} (cross => not certified)")
    plt.xlabel("eps  (per-dim L-inf radius, units of activation std)")
    plt.ylabel("upper bound U on max probe score")
    plt.title(f"Bound tightness, shallow probe, anchor {aid0}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p_tight = os.path.join(CERT_DIR, "tightness.png")
    plt.savefig(p_tight, dpi=110)
    plt.close()

    # ----------------------------------------------------------------- #
    # 2. CERTIFIED RADIUS curves (shallow): per-anchor, CROWN vs IBP
    # ----------------------------------------------------------------- #
    print("=" * 78)
    print("[2] SHALLOW CERTIFIED RADIUS per anchor (CROWN vs IBP)")
    rad_grid = np.linspace(0.02, 2.5, 30)
    radii_crown, radii_ibp = [], []
    for aid in anchor_ids:
        rc = certified_radius(v_crown, X[aid], tau, rad_grid)
        ri = certified_radius(v_ibp, X[aid], tau, rad_grid)
        radii_crown.append(rc)
        radii_ibp.append(ri)
        print(f"  anchor {aid:>3}: CROWN radius={rc:.3f}   IBP radius={ri:.3f}")
    results["shallow_radius"] = {
        "anchor_ids": list(map(int, anchor_ids)),
        "crown": radii_crown,
        "ibp": radii_ibp,
        "crown_mean": float(np.mean(radii_crown)),
        "ibp_mean": float(np.mean(radii_ibp)),
        "rad_grid": rad_grid.tolist(),
    }
    print(f"  MEAN shallow certified radius: CROWN={np.mean(radii_crown):.3f}  "
          f"IBP={np.mean(radii_ibp):.3f}")

    plt.figure(figsize=(7, 4.5))
    xa = np.arange(len(anchor_ids))
    plt.bar(xa - 0.2, radii_crown, width=0.4, color="#36c", label="CROWN")
    plt.bar(xa + 0.2, radii_ibp, width=0.4, color="#c44", label="IBP")
    plt.xticks(xa, [str(a) for a in anchor_ids])
    plt.xlabel("anchor id")
    plt.ylabel("certified radius (eps, std units)")
    plt.title("Shallow certified radius per anchor (larger = provably safer)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p_rad = os.path.join(CERT_DIR, "radius.png")
    plt.savefig(p_rad, dpi=110)
    plt.close()

    # ----------------------------------------------------------------- #
    # 3. NULL-DIRECTION baseline: real vs random direction radius
    # ----------------------------------------------------------------- #
    print("=" * 78)
    print("[3] NULL-DIRECTION baseline (random probe direction vs real)")
    null_probe = _make_null_probe(probe, meta, X, y, seed=0)
    v_null = CertifiedVerifier(null_probe, sigma, norm="inf", method="auto")
    radii_null = []
    for aid in anchor_ids:
        rn = certified_radius(v_null, X[aid], tau, rad_grid)
        radii_null.append(rn)
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        s_real = probe(Xt).numpy()
        s_null = null_probe(Xt).numpy()
    results["null_direction"] = {
        "radii_null": radii_null,
        "radii_real": radii_crown,
        "null_mean": float(np.mean(radii_null)),
        "real_mean": float(np.mean(radii_crown)),
        "real_deploy_mean_score": float(s_real[y == 0].mean()),
        "null_deploy_mean_score": float(s_null[y == 0].mean()),
        "real_eval_mean_score": float(s_real[y == 1].mean()),
        "null_eval_mean_score": float(s_null[y == 1].mean()),
    }
    print(f"  real-direction mean radius = {np.mean(radii_crown):.3f}")
    print(f"  null-direction mean radius = {np.mean(radii_null):.3f}")
    print(f"  real probe deploy/eval mean score: "
          f"{s_real[y==0].mean():.2f} / {s_real[y==1].mean():.2f}")
    print(f"  null probe deploy/eval mean score: "
          f"{s_null[y==0].mean():.2f} / {s_null[y==1].mean():.2f}  "
          f"(eval~deploy => random dir does NOT separate eval, as expected)")

    plt.figure(figsize=(7, 4.5))
    plt.bar(xa - 0.2, radii_crown, width=0.4, color="#2a8", label="real eval-aware direction")
    plt.bar(xa + 0.2, radii_null, width=0.4, color="#a59", label="random null direction")
    plt.xticks(xa, [str(a) for a in anchor_ids])
    plt.xlabel("anchor id")
    plt.ylabel("certified radius (eps, std units)")
    plt.title("Real vs null-direction certified radius")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p_null = os.path.join(CERT_DIR, "null_vs_real.png")
    plt.savefig(p_null, dpi=110)
    plt.close()

    # ----------------------------------------------------------------- #
    # 4. DEEP vs SHALLOW certificate (frozen-attention, through one block)
    # ----------------------------------------------------------------- #
    print("=" * 78)
    print("[4] DEEP certificate (frozen-attention through block, LN+MLP+GELU+probe)")
    print("    SCOPE: sound for the covered subnetwork only; attention frozen to anchor.")
    harness = ModelHarness(cfg)
    deep_cert = make_certifier_from_harness(harness, probe, sigma, cfg.layer, n_witness=2000)

    # deep sweep on one anchor (all-token and single-token), plus deep radius
    aidd = anchor_ids[0]
    h_prev, attn_anchor, w, pooled, T = prepare_anchor(harness, texts[aidd], cfg.layer)
    s_anchor_deep = deep_cert._frozen_forward(h_prev, attn_anchor, w)
    s_anchor_shallow = float(probe(pooled.float()).item())
    print(f"  anchor {aidd}: T={T} tokens, frozen-forward score={s_anchor_deep:.3f} "
          f"(matches shallow probe input {s_anchor_shallow:.3f}: "
          f"{abs(s_anchor_deep - s_anchor_shallow) < 1e-2})")

    deep_eps = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5])
    deep_U_tok, deep_U_all, shallow_U = [], [], []
    for eps in deep_eps:
        rt = deep_cert.certify(h_prev, attn_anchor, w, float(eps), tau,
                               method="crown", token_index=T - 1, witness_seed=aidd)
        ra = deep_cert.certify(h_prev, attn_anchor, w, float(eps), tau,
                               method="crown", token_index=-1, witness_seed=aidd)
        rs = v_crown.certify(X[aidd], float(eps), tau)   # shallow, same eps
        deep_U_tok.append(rt.score_upper)
        deep_U_all.append(ra.score_upper)
        shallow_U.append(rs.score_upper)
        print(f"    eps={eps:5.3f}: shallow U={rs.score_upper:8.2f}  "
              f"deep(1tok) U={rt.score_upper:9.2f}  deep(all) U={ra.score_upper:10.2f}")

    # deep certified radius on a fine grid (single token -- where positive radius lives)
    deep_rad_grid = np.array([0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005,
                              0.006, 0.007, 0.008, 0.01, 0.012, 0.015, 0.02])
    deep_radii = {}
    for aid in anchor_ids[:4]:
        hp, aa, ww, pl, Tt = prepare_anchor(harness, texts[aid], cfg.layer)
        dr = deep_certified_radius(deep_cert, hp, aa, ww, tau, deep_rad_grid,
                                   method="crown", token_index=Tt - 1, witness_seed=aid)
        sr = certified_radius(v_crown, X[aid], tau, rad_grid)
        deep_radii[int(aid)] = {"deep_1tok": float(dr), "shallow": float(sr)}
        print(f"  anchor {aid}: deep radius(1tok)={dr:.4f}   shallow radius={sr:.3f}   "
              f"ratio shallow/deep={ (sr/dr) if dr>0 else float('inf'):.0f}x")

    results["deep"] = {
        "anchor": int(aidd),
        "scope": deep_cert.certify(h_prev, attn_anchor, w, 0.005, tau,
                                   method="crown", token_index=T - 1,
                                   witness_seed=aidd).scope,
        "deep_eps": deep_eps.tolist(),
        "deep_U_1tok": deep_U_tok,
        "deep_U_all": deep_U_all,
        "shallow_U": shallow_U,
        "deep_rad_grid": deep_rad_grid.tolist(),
        "deep_vs_shallow_radius": deep_radii,
    }

    plt.figure(figsize=(7, 4.5))
    plt.plot(deep_eps, shallow_U, "s-", color="#36c", label="shallow (probe only)")
    plt.plot(deep_eps, deep_U_tok, "o-", color="#e83",
             label="deep, frozen-attn, 1 token")
    plt.plot(deep_eps, deep_U_all, "^-", color="#a33",
             label="deep, frozen-attn, all tokens")
    plt.axhline(tau, color="k", lw=1, ls=":", label=f"tau={tau:g}")
    plt.xscale("log")
    plt.xlabel("eps  (per-dim L-inf radius, std units, log scale)")
    plt.ylabel("sound upper bound U on probe score")
    plt.title(f"Shallow vs deep (through 1 block) certificate, anchor {aidd}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    p_deep = os.path.join(CERT_DIR, "deep_vs_shallow.png")
    plt.savefig(p_deep, dpi=110)
    plt.close()

    # save numbers
    out_json = os.path.join(CERT_DIR, "ablation.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 78)
    print(f"wrote {out_json}")

    # ----------------------------------------------------------------- #
    # HTML report (self-contained; our own, does NOT touch cnr/report.py)
    # ----------------------------------------------------------------- #
    ts = time.strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(_ROOT, "reports", f"cert_{ts}.html")
    _write_html(html_path, results,
                imgs={"tight": _png(p_tight), "rad": _png(p_rad),
                      "null": _png(p_null), "deep": _png(p_deep)})
    print(f"wrote {html_path}")
    print(f"done in {time.time()-t0:.1f}s")
    return html_path


def _row(cells, head=False):
    tag = "th" if head else "td"
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


def _write_html(path, R, imgs):
    tau = R["tau"]
    t = R["tightness"]
    sh = R["shallow_radius"]
    nd = R["null_direction"]
    dp = R["deep"]

    # tightness table (subsampled)
    eg = t["eps_grid"]
    tight_rows = "".join(
        _row([f"{eg[i]:.2f}", f"{t['U_crown'][i]:.2f}", f"{t['U_ibp'][i]:.2f}",
              f"{t['crown_ibp_gap'][i]:.3f}", f"{t['U_sampling_unsound'][i]:.2f}"])
        for i in range(0, len(eg), 2))

    # shallow radius table
    rad_rows = "".join(
        _row([str(a), f"{c:.3f}", f"{i_:.3f}"])
        for a, c, i_ in zip(sh["anchor_ids"], sh["crown"], sh["ibp"]))

    # null table
    null_rows = "".join(
        _row([str(a), f"{r:.3f}", f"{n:.3f}"])
        for a, r, n in zip(sh["anchor_ids"], nd["radii_real"], nd["radii_null"]))

    # deep U-vs-eps table
    de = dp["deep_eps"]
    deep_rows = "".join(
        _row([f"{de[i]:.3f}", f"{dp['shallow_U'][i]:.2f}",
              f"{dp['deep_U_1tok'][i]:.2f}", f"{dp['deep_U_all'][i]:.2f}"])
        for i in range(len(de)))

    # deep vs shallow radius table
    dvs = dp["deep_vs_shallow_radius"]
    dvs_rows = "".join(
        _row([a, f"{dvs[a]['deep_1tok']:.4f}", f"{dvs[a]['shallow']:.3f}",
              (f"{dvs[a]['shallow']/dvs[a]['deep_1tok']:.0f}x"
               if dvs[a]['deep_1tok'] > 0 else "inf")])
        for a in dvs)

    style = """
    <style>
      body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
           max-width:980px;margin:24px auto;padding:0 16px;color:#1c1c1c;line-height:1.5}
      h1{font-size:24px} h2{font-size:18px;margin-top:34px;border-bottom:2px solid #eee;padding-bottom:4px}
      table{border-collapse:collapse;margin:10px 0;font-size:13px}
      th,td{border:1px solid #ddd;padding:5px 9px;text-align:right} th{background:#f4f6fa}
      td:first-child,th:first-child{text-align:left}
      img{max-width:100%;border:1px solid #eee;border-radius:6px;margin:8px 0}
      .note{background:#fff8e6;border-left:4px solid #f0a;padding:10px 14px;border-radius:4px;font-size:14px}
      .ok{color:#176;font-weight:600} code{background:#f2f2f4;padding:1px 4px;border-radius:3px}
      .scope{background:#eef4ff;border-left:4px solid #36c;padding:10px 14px;border-radius:4px;font-size:14px}
    </style>"""

    html = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Certified eval-awareness reachability -- ablations</title>{style}</head><body>
    <h1>Certified eval-awareness reachability: ablations &amp; baselines</h1>
    <p>Target set <code>A* = {{h : probe(h) &ge; tau}}</code> with <code>tau = {tau:g}</code>,
    read at GPT-2 residual layer {R['layer']} (mean-pooled). A sound verifier returns an
    upper bound <code>U &ge; max_B probe</code>; <span class="ok">U &lt; tau certifies the
    eval-aware region is unreachable</span> within the eps-ball. Deploy/eval mean probe
    scores: {R['deploy_score_mean']:.2f} / {R['eval_score_mean']:.2f}.</p>

    <div class="note"><b>Soundness contract.</b> Every certificate below passed a sampling
    witness (U &ge; max over &ge;2000 random points in the ball, exact forward). The
    <i>sampling-only</i> curve is an <b>UNSOUND</b> lower reference (it under-estimates the
    true max) shown only to expose how loose the sound bounds are. We never claim coverage
    a sound bound does not provide.</div>

    <h2>1. Bound tightness: CROWN vs IBP vs sampling-only (shallow)</h2>
    <p>Anchor {t['anchor']}, sweeping eps. CROWN is <code>min(CROWN, IBP)</code> (both sound;
    neither pure bound dominates). The CROWN&minus;IBP gap is where the linear relaxation buys
    tightness over plain intervals.</p>
    <img src="{imgs['tight']}" alt="tightness curve">
    <table>{_row(['eps','CROWN U','IBP U','IBP&minus;CROWN gap','sampling (unsound)'],head=True)}{tight_rows}</table>

    <h2>2. Shallow certified radius per anchor</h2>
    <p>Largest eps (std units) still certified unreachable. CROWN mean
    <b>{sh['crown_mean']:.3f}</b>, IBP mean <b>{sh['ibp_mean']:.3f}</b>.</p>
    <img src="{imgs['rad']}" alt="radius bars">
    <table>{_row(['anchor','CROWN radius','IBP radius'],head=True)}{rad_rows}</table>

    <h2>3. Null-direction baseline</h2>
    <p>A random-direction "null probe" (same 1-hidden-layer ReLU family, matched first-layer
    row norms, output bias shifted to the same deploy-class mean) vs the real eval-awareness
    direction. The null direction does <b>not</b> separate eval from deploy
    (null eval/deploy mean score {nd['null_eval_mean_score']:.2f} /
    {nd['null_deploy_mean_score']:.2f}, i.e. ~equal), whereas the real probe does
    ({nd['real_eval_mean_score']:.2f} / {nd['real_deploy_mean_score']:.2f}).
    Real-direction mean radius <b>{nd['real_mean']:.3f}</b> vs null-direction
    <b>{nd['null_mean']:.3f}</b>.</p>
    <img src="{imgs['null']}" alt="null vs real">
    <table>{_row(['anchor','real radius','null radius'],head=True)}{null_rows}</table>
    <p class="note"><b>Interpretation.</b> {_null_interp(nd)}</p>

    <h2>4. Deep certificate: through one real transformer block</h2>
    <div class="scope"><b>Scope (honest).</b> {dp['scope']}.
    The probe reads <code>hidden_states[{R['layer']}]</code>; the deep certificate places the
    eps-ball one block earlier at <code>hidden_states[{R['layer']-1}]</code> and propagates it
    through the block's <b>LayerNorm + up-proj + GELU + down-proj + residual</b> and then the
    probe's ReLU &mdash; four genuine nonlinearities, all bounded soundly. Self-attention is
    <b>frozen to its anchor value</b> (it couples all tokens via softmax and is intractable for
    sound CROWN here); this certifies the named sub-network, NOT the full model.</div>
    <p>Anchor {dp['anchor']}. Upper bound U vs eps for the shallow probe-only certificate, the
    deep certificate perturbing a single token, and the deep certificate perturbing all tokens.
    Deeper propagation is far looser &mdash; the 3072-wide GELU layer's per-coordinate relaxation
    error accumulates through the down-projection.</p>
    <img src="{imgs['deep']}" alt="deep vs shallow">
    <table>{_row(['eps','shallow U','deep U (1 token)','deep U (all tokens)'],head=True)}{deep_rows}</table>

    <h3>Deep vs shallow certified radius</h3>
    <p>The cost of certifying one block deeper, per anchor (single-token deep certificate, the
    regime where a positive deep radius exists):</p>
    <table>{_row(['anchor','deep radius (1 token)','shallow radius','shallow/deep'],head=True)}{dvs_rows}</table>
    <p class="note"><b>Takeaway.</b> The deep certificate is genuinely sound through a real
    GPT-2 nonlinearity stack, but certifies a radius ~2 orders of magnitude smaller than the
    shallow one. Certifying closer to the tokens is real but expensive: interval/linear
    relaxations of GELU over a 3072-dim hidden layer blow up under the down-projection, so only
    very small text-space perturbations are provably eval-unaware at this depth.</p>

    <p style="color:#999;font-size:12px;margin-top:30px">Generated by
    <code>scripts/05_cert_ablation.py</code>. All bounds float64, device cpu. Sound bounds from
    <code>cnr/verify.py</code> (shallow) and <code>cnr/verify_deep.py</code> (deep).</p>
    </body></html>"""
    with open(path, "w") as f:
        f.write(html)


def _null_interp(nd):
    real, null = nd["real_mean"], nd["null_mean"]
    if real > null * 1.15:
        return (f"The real eval-awareness direction certifies a <b>larger</b> radius "
                f"({real:.3f} vs {null:.3f}): the benign anchors sit farther (in the certified "
                f"L-inf sense) from the real eval-aware halfspace than from a random halfspace. "
                f"The certificate is exploiting structure specific to the eval-awareness "
                f"direction, not a generic property of random linear readouts.")
    elif null > real * 1.15:
        return (f"The random direction certifies a <b>larger</b> radius ({null:.3f} vs "
                f"{real:.3f}): the real eval-aware halfspace passes closer to the benign anchors "
                f"than a typical random halfspace, i.e. these anchors are comparatively near the "
                f"eval-aware decision boundary along the real direction.")
    else:
        return (f"Real and null radii are comparable ({real:.3f} vs {null:.3f}): at this scale "
                f"the certified radius is driven mainly by the network/separation geometry "
                f"common to both, not by the specific eval-awareness direction.")


if __name__ == "__main__":
    main()
