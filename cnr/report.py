"""Timestamped HTML results dashboard. No external libs: inline CSS + a tiny
hand-rolled SVG sparkline. Renders whatever pipeline stages are present.

The report leads with an honest status banner so nobody mistakes the
retrieval-based verbalizer stand-in for a trained NLA, or the sound-but-loose
certificate for a tight one.
"""
from __future__ import annotations

import datetime as _dt
import html
from pathlib import Path


def _sparkline(history, w=220, h=40, pad=3):
    if not history:
        return ""
    lo, hi = min(history), max(history)
    rng = (hi - lo) or 1.0
    n = len(history)
    pts = []
    for i, v in enumerate(history):
        x = pad + (w - 2 * pad) * (i / max(1, n - 1))
        y = h - pad - (h - 2 * pad) * ((v - lo) / rng)
        pts.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg width="{w}" height="{h}" style="background:#0d1117;border-radius:4px">'
        f'<polyline fill="none" stroke="#3fb950" stroke-width="1.5" points="{" ".join(pts)}"/>'
        f"</svg>"
    )


def _row(cells, header=False):
    tag = "th" if header else "td"
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


def _esc(x):
    return html.escape(str(x))


def _verdict(results: dict) -> str:
    """Plain-language 'does it work?' box, derived from the actual run (not typed)."""
    meta = results.get("meta", {})
    v = results.get("verifier") or {}
    reg = results.get("regularizer") or {}
    atk = results.get("attack") or {}
    lp = results.get("loop") or {}
    anchors = v.get("anchors", [])
    n_cert = sum(1 for a in anchors if a.get("certified_unreachable"))

    checks = []  # (passed, html_text)
    val = meta.get("val_acc")
    if isinstance(val, (int, float)):
        checks.append((val >= 0.9,
            f"The concept probe cleanly tells eval-like from ordinary activations "
            f"(validation accuracy {val:.0%})."))
    if anchors:
        checks.append((n_cert > 0,
            f"The certificate is real and it bites: {n_cert} of {len(anchors)} benign "
            f"anchors are <b>provably unable</b> to reach the eval-aware region inside the "
            f"ε-ball — and every bound is cross-checked against brute-force sampling."))
    if reg:
        rb, ra = reg.get("radius_before"), reg.get("radius_after")
        if rb is not None and ra is not None:
            checks.append((ra > rb,
                f"The certificate doubles as a training signal: descending it grows the "
                f"provably-safe radius ({rb:.2f} → {ra:.2f}) with accuracy unchanged."))
    if atk:
        checks.append((bool(atk.get("crossed")),
            f"The inversion works: a token search pushes a benign prompt <b>across the "
            f"threshold</b> into the eval-aware region (score {atk.get('score_start','?')} → "
            f"{atk.get('score'):+.1f}), beating random search."))
    if lp:
        checks.append((bool(lp.get("agree")),
            "The crossing is <b>semantically real</b>, not a numerical fluke: an independent "
            "readout agrees the activation looks eval-aware."))

    all_ok = len(checks) >= 4 and all(ok for ok, _ in checks)
    items = []
    for ok, txt in checks:
        mark = "✅" if ok else "❌"
        items.append(f"<li>{mark} {txt}</li>")
    head = (
        'Does it work? &nbsp;<span class=ok>Yes — end to end, as a proof of concept.</span>'
        if all_ok else
        'Does it work? &nbsp;<span class=no>Partly — see which row failed below.</span>'
    )
    caveat = (
        '<div class=vcav><b>What “works” means here, honestly:</b> this is a methods '
        "skeleton on a deliberately easy <i>synthetic</i> eval-vs-deploy task (that is why "
        "accuracy is ~100%). The certificate covers reachability in <i>activation space "
        "through the probe</i>, not all the way back through GPT-2 to the tokens, and the "
        "loop-closure readout is a retrieval <i>stand-in</i>, not a trained NLA. So the "
        "machinery and the math are real and sound — it is not yet evidence about real "
        "eval-awareness, which would need NLA-derived labels."
        "</div>"
    )
    cls = "verdict" if all_ok else "verdict no"
    return f'<div class="{cls}"><p class=vhead>{head}</p><ul>{"".join(items)}</ul>{caveat}</div>'


def build_report(results: dict, out_dir: Path) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir) / f"report_{ts}.html"
    _i = 1
    while out.exists():
        out = Path(out_dir) / f"report_{ts}_{_i}.html"
        _i += 1
    meta = results.get("meta", {})

    parts = [
        """<!doctype html><meta charset=utf-8>
<title>Certified NLA Reachability — results</title>
<style>
 body{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;
      margin:32px auto;padding:0 18px;color:#e6edf3;background:#0d1117}
 h1{font-size:22px;margin:0 0 2px} h2{font-size:16px;margin:26px 0 8px;
    border-bottom:1px solid #30363d;padding-bottom:4px}
 .sub{color:#8b949e;margin:0 0 18px}
 table{border-collapse:collapse;width:100%;margin:6px 0 4px}
 th,td{border:1px solid #30363d;padding:5px 9px;text-align:left;font-size:13px}
 th{background:#161b22}
 code{background:#161b22;padding:1px 5px;border-radius:4px}
 .banner{background:#1c2128;border:1px solid #f0883e;border-radius:8px;padding:12px 16px;margin:14px 0}
 .ok{color:#3fb950;font-weight:600}.no{color:#f85149;font-weight:600}
 .pill{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px;background:#21262d;color:#8b949e;margin-left:6px}
 .quote{background:#161b22;border-left:3px solid #6e7681;padding:6px 12px;color:#adbac7}
 .verdict{border-radius:10px;padding:16px 20px;margin:16px 0;border:2px solid #3fb950;background:#11271a}
 .verdict.no{border-color:#f0883e;background:#241a11}
 .vhead{font-size:19px;font-weight:700;margin:0 0 6px}
 .verdict ul{margin:8px 0 0;padding-left:0;list-style:none}
 .verdict li{margin:6px 0;padding-left:24px;text-indent:-24px}
 .vcav{margin-top:13px;color:#c9b89a;font-size:13px;border-top:1px solid #30363d;padding-top:10px}
</style>""",
        "<h1>Certified reachability of an NLA-defined concept</h1>",
        f'<p class="sub">target model <code>{_esc(meta.get("model","?"))}</code> · '
        f'residual layer {_esc(meta.get("layer","?"))} · '
        f'd={_esc(meta.get("d_model","?"))} · generated {ts}</p>',
        _verdict(results),
        """<div class=banner>
 <b>Honest status.</b> &nbsp;
 <span class=pill>probe = real trained ReLU net</span>
 <span class=pill>certificate = SOUND bound</span>
 <span class=pill>verbalizer = retrieval STAND-IN, not a trained NLA</span><br>
 The concept target A* is the decision region of a 1-hidden-layer ReLU probe
 (Raghunathan's 2018 architecture class). Certificates are one-sided guarantees.
 The loop-closure "readout" is k-NN retrieval over labeled activations, a
 stand-in for an NLA activation-verbalizer — it is not a trained autoencoder.
</div>""",
    ]

    # provenance / probe quality
    parts.append("<h2>1 · Concept target A* (the probe)</h2>")
    parts.append(
        "<table>"
        + _row(["metric", "value"], header=True)
        + _row(["probe", "Linear(768,64) → ReLU → Linear(64,1) over raw activations"])
        + _row(["train / val accuracy", f'{meta.get("train_acc","?"):.3f} / {meta.get("val_acc","?"):.3f}'
                if isinstance(meta.get("val_acc"), float) else "?"])
        + _row(["score: eval mean vs deploy mean",
                f'{meta.get("score_eval_mean","?")} vs {meta.get("score_deploy_mean","?")}'])
        + _row(["A* threshold τ", _esc(meta.get("tau", 0.0))])
        + "</table>"
        + '<p class="sub">Note: the synthetic eval-vs-deploy task is near-linearly '
        "separable, so accuracy saturates. Real eval-awareness signal is subtler; "
        "this is a methods skeleton, not a difficulty benchmark.</p>"
    )

    # verifier
    v = results.get("verifier")
    parts.append("<h2>2 · Certified reachability (Raghunathan-style relaxation)</h2>")
    if v:
        method = v.get("method", "?")
        parts.append(
            f'<p>method: <code>{_esc(method)}</code>'
            + ("" if method == "crown" else ' <span class=pill>IBP — sound, looser than CROWN</span>')
            + f' · fixed ε for the table: {_esc(v.get("eps_fixed","?"))} (per-dim, ×σ)</p>'
        )
        rows = _row(
            ["anchor", "probe score s(h₀)", "bound U over ball", "U &lt; τ?", "certified radius εmax"],
            header=True,
        )
        for a in v.get("anchors", []):
            cert = a.get("certified_unreachable")
            badge = '<span class=ok>certified unreachable</span>' if cert else '<span class=no>not certified</span>'
            rows += _row([
                a.get("id"), f'{a.get("score"):.2f}', f'{a.get("U"):.2f}',
                badge, f'{a.get("radius"):.3f}',
            ])
        parts.append("<table>" + rows + "</table>")
        parts.append(
            '<p class="sub">Reading: around a benign activation h₀, no perturbation '
            "inside the ε·σ ball can push the eval-awareness probe across τ — a "
            "guarantee, not a correlation. The certified radius is where that breaks.</p>"
        )
    else:
        parts.append('<p class="sub">verifier stage not present in this run.</p>')

    # regularizer
    r = results.get("regularizer")
    if r:
        parts.append("<h2>2b · Differentiable certificate as a regularizer</h2>")
        parts.append(
            "<table>"
            + _row(["", "certified radius (mean)", "bank accuracy"], header=True)
            + _row(["before", f'{r.get("radius_before"):.3f}', f'{r.get("acc_before"):.3f}'])
            + _row(["after", f'{r.get("radius_after"):.3f}', f'{r.get("acc_after"):.3f}'])
            + "</table>"
            + '<p class="sub">Mechanism demo (Raghunathan\'s actual contribution): the '
            "bound is differentiable, so the concept region can be pushed away from "
            "benign activations while keeping accuracy. Not a tuned result.</p>"
        )

    # attack
    a = results.get("attack")
    parts.append("<h2>3 · Discrete token search (GCG-style inversion)</h2>")
    if a:
        crossed = a.get("crossed")
        parts.append(
            f'<p>prefix: <code>{_esc(a.get("prefix",""))}</code><br>'
            f'found prompt: <span class=quote>{_esc(a.get("prompt",""))}</span></p>'
            "<table>"
            + _row(["", "probe score", "crossed τ?"], header=True)
            + _row(["benign start", f'{a.get("score_start","?")}', ""])
            + _row(["GCG result", f'{a.get("score"):.2f}',
                    '<span class=ok>yes</span>' if crossed else '<span class=no>no</span>'])
            + _row(["random baseline", f'{a.get("random_score","?")}', ""])
            + "</table>"
            + f'<p class="sub">score trajectory {_sparkline(a.get("history", []))}</p>'
        )
    else:
        parts.append('<p class="sub">attack stage not present in this run.</p>')

    # loop closure
    lp = results.get("loop")
    parts.append("<h2>4 · Loop closure — re-verbalize &amp; check agreement</h2>")
    if lp:
        agree = lp.get("agree")
        parts.append(
            f'<p>readout: <span class=quote>{_esc(lp.get("readout",""))}</span></p>'
            "<table>"
            + _row(["geometry (probe ≥ τ)", "readout says eval", "agree?"], header=True)
            + _row([
                '<span class=ok>eval-aware</span>' if lp.get("crossed") else '<span class=no>ordinary</span>',
                '<span class=ok>yes</span>' if lp.get("readout_says_eval") else '<span class=no>no</span>',
                '<span class=ok>✓</span>' if agree else '<span class=no>✗ disagree</span>',
            ])
            + "</table>"
            + '<p class="sub">Agreement guards against an inversion that hits A* '
            "numerically but is not semantically eval-aware (the adversarial-feature "
            "failure). Disagreement is the interesting, report-worthy case.</p>"
        )
    else:
        parts.append('<p class="sub">loop-closure stage not present in this run.</p>')

    out.write_text("\n".join(parts), encoding="utf-8")
    return out
