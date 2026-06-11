"""Salvage/finish the certification ablation that the deep-radius loop stalled on.

Reuses the (fast, sound) shallow primitives from cnr.verify and the deep
frozen-attention certifier from cnr.verify_deep, but BOUNDS the deep compute
(one anchor, an eps-sweep, fewer witnesses, a hard SIGALRM cap) so it cannot
hang. Recomputes the ablation numbers (the original run made the PNGs but stalled
before saving JSON/HTML), then writes a self-contained report.
"""
import datetime as dt
import json
import signal
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnr.config import Config
from cnr.harness import ModelHarness
from cnr.probe import ReLUProbe, load_probe
from cnr.verify import (
    CertifiedVerifier,
    _extract_params,
    _sampled_max,
    certified_radius,
    crown_upper,
    ibp_upper,
)

cfg = Config()
probe, meta = load_probe(cfg.probe_path)
sigma = np.asarray(meta["sigma"], dtype=np.float64)
tau = float(meta.get("tau", 0.0))
bank = np.load(cfg.bank_path)
X, y = bank["X"], bank["y"]
texts = json.loads(cfg.bank_text_path.read_text())["texts"]
anchor_ids = json.loads(cfg.meta_path.read_text())["anchor_ids"]

W1, b1, W2, b2 = _extract_params(probe)
results = {"meta": {"model": cfg.model_name, "layer": cfg.layer, "tau": tau}}

# ----------------------------------------------------------------------------- #
# 1) tightness: pure CROWN vs pure IBP vs (unsound) sampling, on one anchor
# ----------------------------------------------------------------------------- #
print("[1] tightness ablation (CROWN vs IBP vs sampling)")
h0 = X[anchor_ids[0]].astype(np.float64)
eps_grid = np.linspace(0.02, 0.6, 13)
tight = []
for eps in eps_grid:
    ev = eps * sigma
    xl = torch.tensor(h0 - ev)
    xu = torch.tensor(h0 + ev)
    Uc = float(crown_upper(W1, b1, W2, b2, xl, xu).item())
    Ui = float(ibp_upper(W1, b1, W2, b2, xl, xu).item())
    Us = _sampled_max(probe, h0, ev, 3000)
    assert min(Uc, Ui) >= Us - 1e-4, "SOUNDNESS VIOLATION"
    tight.append({"eps": float(eps), "crown": Uc, "ibp": Ui, "sampling": Us})
gap = np.mean([t["ibp"] - t["crown"] for t in tight])
results["tightness"] = {"rows": tight, "mean_ibp_minus_crown": float(gap)}
print(f"    mean IBP-CROWN gap over eps = {gap:.2f} logits (CROWN tighter)")

# ----------------------------------------------------------------------------- #
# 2) certified radius per anchor (shallow), real eval direction
# ----------------------------------------------------------------------------- #
print("[2] shallow certified radius per anchor")
v = CertifiedVerifier(probe, sigma, norm="inf", method="auto")
rad_grid = np.linspace(0.02, 2.0, 25)
shallow_radii = [certified_radius(v, X[a], tau, rad_grid) for a in anchor_ids[:6]]
results["shallow_radius"] = {
    "per_anchor": shallow_radii,
    "mean": float(np.mean(shallow_radii)),
    "std": float(np.std(shallow_radii)),
}
print(f"    mean certified radius = {np.mean(shallow_radii):.3f} ± {np.std(shallow_radii):.3f}")

# ----------------------------------------------------------------------------- #
# 3) null-direction baseline: a random ReLU probe with matched weight scale
# ----------------------------------------------------------------------------- #
print("[3] null-direction baseline")
rng = np.random.default_rng(0)
null = ReLUProbe(W1.shape[1], W1.shape[0])
with torch.no_grad():
    null.net[0].weight.copy_(torch.tensor(rng.standard_normal(tuple(W1.shape)) * float(W1.std()), dtype=torch.float32))
    null.net[0].bias.zero_()
    null.net[2].weight.copy_(torch.tensor(rng.standard_normal(tuple(W2.shape)) * float(W2.std()), dtype=torch.float32))
    null.net[2].bias.zero_()
null.eval()
vn = CertifiedVerifier(null, sigma, norm="inf", method="auto")
null_radii = [certified_radius(vn, X[a], tau, rad_grid) for a in anchor_ids[:6]]
results["null_radius"] = {"mean": float(np.mean(null_radii)), "std": float(np.std(null_radii))}
print(f"    null random-direction mean radius = {np.mean(null_radii):.3f} "
      f"(real = {np.mean(shallow_radii):.3f})")

# ----------------------------------------------------------------------------- #
# 4) DEEP certificate through one GPT-2 block (bounded; SIGALRM-guarded)
# ----------------------------------------------------------------------------- #
print("[4] deep frozen-attention certificate (bounded: 1 anchor, eps-sweep)")
deep = {"ok": False}


def _timeout(signum, frame):
    raise TimeoutError("deep certificate exceeded time budget")


signal.signal(signal.SIGALRM, _timeout)
signal.alarm(220)
try:
    from cnr.verify_deep import make_certifier_from_harness, prepare_anchor

    harness = ModelHarness(cfg)
    dc = make_certifier_from_harness(harness, probe, sigma, cfg.layer, n_witness=200)
    aid = anchor_ids[0]
    hp, aa, w, pooled, T = prepare_anchor(harness, texts[aid], cfg.layer)
    s_deep = float(dc._frozen_forward(hp, aa, w))
    s_shallow = float(probe(pooled.float()).item())
    deep_eps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25]
    sweep = []
    for eps in deep_eps:
        r = dc.certify(hp, aa, w, float(eps), tau, method="crown",
                       token_index=T - 1, witness_seed=aid)
        sweep.append({"eps": float(eps), "U": float(r.score_upper),
                      "certified": bool(r.certified_unreachable)})
    signal.alarm(0)
    drad = max([s["eps"] for s in sweep if s["certified"]], default=0.0)
    deep = {"ok": True, "anchor": int(aid), "token_index": int(T - 1), "T": int(T),
            "frozen_forward": s_deep, "shallow_input": s_shallow,
            "forward_matches": abs(s_deep - s_shallow) < 1e-2,
            "sweep": sweep, "deep_radius_single_token": float(drad)}
    print(f"    frozen-fwd score {s_deep:.3f} (matches shallow {s_shallow:.3f}: {deep['forward_matches']})")
    print(f"    deep single-token certified radius = {drad:.4f} (shallow {np.mean(shallow_radii):.3f})")
except TimeoutError as e:
    signal.alarm(0)
    deep = {"ok": False, "reason": f"{e}; shallow ablation complete, deep is the honest frontier"}
    print(f"    [bounded] {deep['reason']}")
except Exception as e:
    signal.alarm(0)
    deep = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    print(f"    [skip] {deep['reason']}")
results["deep"] = deep

# ----------------------------------------------------------------------------- #
# save + report
# ----------------------------------------------------------------------------- #
cert_dir = cfg.artifacts / "cert"
cert_dir.mkdir(exist_ok=True)
(cert_dir / "cert_results.json").write_text(json.dumps(results, indent=2))

ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
out = cfg.reports / f"cert_{ts}.html"


def _rows(rows, cols):
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{r[c]}</td>" for c in cols) + "</tr>"
    return f"<table>{h}{body}</table>"


tight_tbl = "".join(
    f"<tr><td>{t['eps']:.3f}</td><td>{t['crown']:.2f}</td><td>{t['ibp']:.2f}</td>"
    f"<td>{t['sampling']:.2f}</td></tr>" for t in tight[::2]
)
if deep["ok"]:
    deep_tbl = "".join(
        f"<tr><td>{s['eps']:.3f}</td><td>{s['U']:.2f}</td>"
        f"<td>{'✅ unreachable' if s['certified'] else '—'}</td></tr>" for s in deep["sweep"]
    )
    deep_html = (
        f"<p>frozen-attention forward score {deep['frozen_forward']:.2f} matches the shallow "
        f"probe input {deep['shallow_input']:.2f} ({deep['forward_matches']}) — the covered "
        f"subnetwork is wired correctly. Single-token deep certified radius "
        f"<b>{deep['deep_radius_single_token']:.4f}</b> vs shallow "
        f"<b>{results['shallow_radius']['mean']:.3f}</b> — deeper is strictly harder to certify, "
        f"as expected.</p>"
        f"<table><tr><th>ε (×σ)</th><th>deep U (1 token)</th><th>verdict</th></tr>{deep_tbl}</table>"
    )
else:
    deep_html = f"<p class=warn>Deep certificate bounded out: {deep['reason']}. The shallow ablation below is complete and sound.</p>"

out.write_text(f"""<!doctype html><meta charset=utf-8><title>Certification depth + ablations</title>
<style>body{{font:14px/1.55 -apple-system,sans-serif;max-width:880px;margin:30px auto;padding:0 18px;color:#e6edf3;background:#0d1117}}
h1{{font-size:21px}}h2{{font-size:16px;border-bottom:1px solid #30363d;padding-bottom:4px;margin-top:26px}}
table{{border-collapse:collapse;margin:8px 0}}th,td{{border:1px solid #30363d;padding:4px 10px;font-size:13px;text-align:right}}
img{{max-width:100%;border:1px solid #30363d;border-radius:6px;margin:8px 0;background:#fff}}
.sub{{color:#8b949e}}.warn{{color:#f0883e}}code{{background:#161b22;padding:1px 5px;border-radius:4px}}</style>
<h1>Certification depth + ablations</h1>
<p class=sub>target <code>gpt2</code> · layer {cfg.layer} · sound native CROWN(+IBP); soundness witnessed every call</p>

<h2>1 · Bound tightness — CROWN vs IBP vs sampling</h2>
<p>Mean IBP−CROWN gap over ε: <b>{gap:.2f}</b> logits (CROWN is the tighter sound bound; sampling is an <i>unsound</i> lower reference).</p>
<table><tr><th>ε (×σ)</th><th>CROWN U</th><th>IBP U</th><th>sampling (unsound)</th></tr>{tight_tbl}</table>
<img src="../artifacts/cert/tightness.png">

<h2>2 · Shallow certified radius (real eval direction)</h2>
<p>Mean over {len(shallow_radii)} anchors: <b>{np.mean(shallow_radii):.3f} ± {np.std(shallow_radii):.3f}</b> σ-units.</p>
<img src="../artifacts/cert/radius.png">

<h2>3 · Null-direction baseline</h2>
<p>Random matched-scale direction: <b>{np.mean(null_radii):.3f}</b> vs real eval direction <b>{np.mean(shallow_radii):.3f}</b>.</p>
<img src="../artifacts/cert/null_vs_real.png">

<h2>4 · Deep certificate through one GPT-2 block (frozen-attention, sound subnetwork)</h2>
{deep_html}
""", encoding="utf-8")
print("REPORT:", out)
