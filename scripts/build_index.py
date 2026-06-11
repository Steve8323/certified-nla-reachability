"""Build one combined index/dashboard over all component reports + the paper.

Links the latest report of each kind and collects the headline numbers, with the
same honest framing as the individual reports. Writes reports/INDEX_<ts>.html.
"""
import datetime as dt
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnr.config import Config

cfg = Config()
R = cfg.reports


def latest(pat):
    fs = sorted(glob.glob(str(R / pat)), key=os.path.getmtime, reverse=True)
    return os.path.basename(fs[0]) if fs else None


comps = [
    ("Bottom half — direct probe, full chain", latest("report_*.html"),
     "certify → regularize → invert → close-loop. Certified radius <b>0.29→0.49</b> via the differentiable bound; GCG inversion <b>−8.95→+4.72</b> (crosses τ, beats random); loop-closure agrees (5/5 eval neighbors)."),
    ("Top half — NLA→classifier→ReLU surrogate", latest("tophalf_*.html"),
     "Surrogate is a faithful shadow of the NLA-text classifier (held-out corr <b>0.99</b>, sign <b>1.00</b>); CROWN radius ≈0.13; the off-manifold gaming finding originates here."),
    ("Harder task — honest, non-saturated", latest("hard_*.html"),
     "Calibrated label-noise overlap (not phrasing): <b>AUROC 0.841 ± 0.021</b>, acc 0.740 ± 0.019, certified radius 0.078 ± 0.073 (AUROC &lt; 0.95 confirmed). 5 seeds."),
    ("Off-manifold gaming sweep", latest("gaming_*.html"),
     "Loop-closure catch rate <b>1.000 off-manifold</b> (P(eval | g≥τ) = 0.000) vs 0.738 on-manifold; genuine-eval crossing 24.5σ on-manifold, <b>never within 25σ off-manifold</b>."),
    ("Certification depth + ablations", latest("cert_*.html"),
     "CROWN tighter than IBP by 0.72 logits; real eval direction 0.295 vs null 0.213; <b>deep certificate one block back: radius 0.005 vs 0.295 (~60× harder)</b> — the scale wall, quantified. 9 tests pass."),
    ("Real open-model NLA", latest("realnla_*.html"),
     "Real NLAs located (kitft/nla-models); injection format verified <b>byte-for-byte</b>; smallest is Qwen2.5-7B (15GB) — no GPT-2 NLA, so honest <b>partial</b> + migration path, no fabricated readout."),
]

ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
out = R / f"INDEX_{ts}.html"
pdf_rel = "../paper/main.pdf"

rows = ""
for name, link, desc in comps:
    href = link or "#"
    miss = "" if link else " <span style='color:#f85149'>(missing)</span>"
    rows += (f"<tr><td><a href='{href}'>{name}</a>{miss}</td><td>{desc}</td></tr>")

html = f"""<!doctype html><meta charset=utf-8><title>Certified NLA Reachability — project index</title>
<style>
 body{{font:14px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;margin:32px auto;padding:0 18px;color:#e6edf3;background:#0d1117}}
 h1{{font-size:23px;margin:0 0 2px}} .sub{{color:#8b949e;margin:0 0 16px}}
 h2{{font-size:16px;border-bottom:1px solid #30363d;padding-bottom:4px;margin-top:26px}}
 table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #30363d;padding:8px 11px;vertical-align:top;font-size:13px}}
 td:first-child{{width:30%;font-weight:600}} a{{color:#58a6ff;text-decoration:none}} a:hover{{text-decoration:underline}}
 .banner{{background:#11271a;border:2px solid #3fb950;border-radius:10px;padding:14px 18px;margin:14px 0}}
 .pill{{display:inline-block;font-size:11px;padding:1px 8px;border-radius:10px;background:#21262d;color:#8b949e;margin:2px 4px 2px 0}}
 .cav{{color:#c9b89a;font-size:13px;border-top:1px solid #30363d;padding-top:10px;margin-top:12px}}
</style>
<h1>Certified Reachability of NLA-Defined Concepts</h1>
<p class=sub>proving when a model can(not) recognize it is being tested · index generated {ts}</p>

<div class=banner>
 <b>Status: complete methods skeleton, validated end to end.</b><br>
 Five parallel work-streams plus the original chain, all on cached GPT-2.
 <span class=pill>probe/surrogate = real</span>
 <span class=pill>certificate = sound (witnessed)</span>
 <span class=pill>verbalizer = retrieval stand-in</span>
 <span class=pill>real NLA = honest partial</span><br>
 Test suites: bottom-half 11, top-half 16, harder 5, gaming 5, cert 9 — all pass.
 Paper builds to 12 pages.
 <div class=cav><b>The one robust scientific finding</b> is the negative one: a geometry-only
 target gets gamed off-manifold (the certificate is satisfied at ~24σ without the state ever
 being genuinely eval-reading), and the language re-decode catches it 100% of the time. Everything
 else is honest plumbing on a synthetic task — see each report's own caveats.</div>
</div>

<h2>Components &amp; results</h2>
<table>
<tr><th>report</th><th>headline result</th></tr>
{rows}
</table>

<h2>Paper</h2>
<p><a href="{pdf_rel}">paper/main.pdf</a> — 12pp, compiles clean (no undefined refs, no bad overfull). Real numbers folded into the results section; limitations prominent (synthetic task, activation-space-not-tokens, retrieval stand-in, reachability≠attribution).</p>

<h2>Honest limitations (carried from every report)</h2>
<p class=cav>Synthetic near-separable task (the harder run fixes the headline metric to AUROC 0.84, but the task is still synthetic); certification holds in activation space through the probe/surrogate, not back through the full transformer to tokens (the deep run shows that radius collapses ~60× one block back); the verbalizer is a k-NN retrieval stand-in, not a trained NLA (the real-NLA run shows wiring it up is a base-model migration to Qwen-7B); and a reachability certificate finds inference-time inputs that reach A*, not the training data that instilled the concept — that needs influence functions.</p>
"""

out.write_text(html, encoding="utf-8")
print("INDEX:", out)
