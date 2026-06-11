"""ACTIVATION-SPACE gaming sweep + self-contained HTML report.

Turns the project's key qualitative finding -- "loop-closure catches
off-manifold surrogate-gaming the certificate can't see" -- into numbers and a
report. Runs cnr.gaming.run_experiment over >=6 benign anchors along two routes
(on-manifold interpolation toward the eval cluster; off-manifold gradient ascent
on the surrogate g), records g / certificate-status / NLA-re-decode at every
displacement, saves the raw results + a matplotlib figure to artifacts/gaming/,
and writes reports/gaming_<timestamp>.html.

    python scripts/04_gaming_sweep.py

Honesty: the NLA verbalizer is a transparent STAND-IN (cnr/nla.py), not a trained
Natural Language Autoencoder; the report says so. Everything else (GPT-2 acts,
surrogate g, the sound CROWN/IBP certificate) is real computation.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cnr.config import Config
from cnr.gaming import run_experiment, save_results

# headless matplotlib (figure under artifacts/gaming/)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# figure: mean curves over anchors -- g, certificate status, readout-agreement
# --------------------------------------------------------------------------- #
def make_figure(results: dict, out_png: Path) -> Path:
    deltas = np.asarray(results["deltas"], dtype=float)
    tau = results["meta"]["tau"]
    nd = len(deltas)

    def stack(route, key):
        arr = np.full((len(results["per_anchor"]), nd), np.nan)
        for i, pa in enumerate(results["per_anchor"]):
            pts = pa["routes"][route]["points"]
            for j, p in enumerate(pts):
                arr[i, j] = p[key]
        return arr

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {"on_manifold": "#1f77b4", "off_manifold": "#d62728"}
    labels = {"on_manifold": "ON-manifold (interp -> eval cluster)",
              "off_manifold": "OFF-manifold (gradient ascent on g)"}

    # (1) g(h) vs displacement
    ax = axes[0]
    for route in ("on_manifold", "off_manifold"):
        gm = np.nanmean(stack(route, "g"), axis=0)
        ax.plot(deltas, gm, color=colors[route], label=labels[route], lw=2)
    ax.axhline(tau, color="k", ls="--", lw=1, label=f"tau = {tau:g}")
    ax.set_xlabel("displacement  (per-dim std units)")
    ax.set_ylabel("surrogate  g(h)   (mean over anchors)")
    ax.set_title("(1) surrogate score: off-manifold crosses cheaply")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # (2) certificate status: fraction of anchors certified-unreachable
    ax = axes[1]
    for route in ("on_manifold", "off_manifold"):
        cu = np.nanmean(stack(route, "certified_unreachable"), axis=0)
        ax.plot(deltas, cu, color=colors[route], label=labels[route], lw=2)
    ax.set_xlabel("displacement  (per-dim std units)")
    ax.set_ylabel("P(certified_unreachable) at eps")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"(2) certificate (eps={results['meta']['eps']:g}): blind past its radius")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # (3) readout agreement among g>=tau points -- the gaming
    ax = axes[2]
    for route in ("on_manifold", "off_manifold"):
        g_arr = stack(route, "g")
        rd_arr = stack(route, "readout_eval")
        frac = np.full(nd, np.nan)
        for j in range(nd):
            mask = g_arr[:, j] >= tau
            if mask.sum() > 0:
                frac[j] = np.nanmean(rd_arr[mask, j])
        ax.plot(deltas, frac, color=colors[route], label=labels[route], lw=2, marker="o", ms=3)
    ax.set_xlabel("displacement  (per-dim std units)")
    ax.set_ylabel("P(readout==eval | g>=tau)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(3) loop-closure: readout agrees on-manifold, never off-manifold")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    fig.suptitle("Activation-space gaming: certificate is local-sound but blind; "
                 "loop-closure (NLA re-decode) catches off-manifold gaming",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# self-contained HTML (no dependency on cnr/report.py)
# --------------------------------------------------------------------------- #
def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _fmt(v, nd=2):
    if v is None:
        return "&mdash;"
    if isinstance(v, float):
        if np.isnan(v):
            return "n/a"
        return f"{v:.{nd}f}"
    return _esc(v)


def write_html(results: dict, png_rel: str, out_html: Path, ts: str) -> Path:
    m = results["meta"]
    a = results["aggregate"]

    # --- per-anchor crossing table ---
    rows = []
    for pa in results["per_anchor"]:
        on = pa["routes"]["on_manifold"]
        off = pa["routes"]["off_manifold"]
        rows.append(
            "<tr>"
            f"<td>{pa['anchor_id']}</td>"
            f"<td>{_fmt(pa['g0'])}</td>"
            f"<td>{_fmt(on['g_cross_disp'])}</td>"
            f"<td>{_fmt(off['g_cross_disp'])}</td>"
            f"<td>{_fmt(on['readout_flip_disp'])}</td>"
            f"<td>{'never' if off['readout_flip_disp'] is None else _fmt(off['readout_flip_disp'])}</td>"
            f"<td>{_fmt(on['genuine_cross_disp'])}</td>"
            f"<td>{'never' if off['genuine_cross_disp'] is None else _fmt(off['genuine_cross_disp'])}</td>"
            "</tr>"
        )
    anchor_table = "\n".join(rows)

    # --- an illustrative single-anchor trajectory (first anchor) ---
    pa0 = results["per_anchor"][0]
    trows = []
    on_pts = pa0["routes"]["on_manifold"]["points"]
    off_pts = pa0["routes"]["off_manifold"]["points"]
    for po, pf in zip(on_pts, off_pts):
        def cell(p):
            cls = "cross" if p["g_crossed"] else ""
            rd = "EVAL" if p["readout_eval"] else "deploy"
            rdcls = "ev" if p["readout_eval"] else "dp"
            certs = "yes" if p["certified_unreachable"] else "no"
            return (f"<td class='{cls}'>{p['g']:.2f}</td>"
                    f"<td>{certs}</td>"
                    f"<td class='{rdcls}'>{rd} ({p['readout_proba']:.2f})</td>")
        trows.append(f"<tr><td>{po['disp_std']:.1f}</td>{cell(po)}{cell(pf)}</tr>")
    traj_table = "\n".join(trows)

    catch_off = a["catch_rate_off"]
    catch_on = a["catch_rate_on"]
    peval_on = a["p_readout_eval_given_gcross_on"]
    peval_off = a["p_readout_eval_given_gcross_off"]

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Activation-space gaming sweep &mdash; {ts}</title>
<style>
  :root {{ --on:#1f77b4; --off:#d62728; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
          max-width: 1080px; margin: 2rem auto; padding: 0 1rem; color:#1a1a1a; line-height:1.5; }}
  h1 {{ font-size: 1.5rem; margin-bottom: .2rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2rem; border-bottom: 2px solid #eee; padding-bottom:.25rem; }}
  .sub {{ color:#666; font-size:.9rem; }}
  .banner {{ background:#fff8e1; border:1px solid #ffe082; border-radius:8px; padding:.7rem 1rem;
             margin:1rem 0; font-size:.9rem; }}
  .banner b {{ color:#a86a00; }}
  table {{ border-collapse: collapse; width:100%; font-size:.86rem; margin:.6rem 0; }}
  th, td {{ border:1px solid #ddd; padding:.35rem .5rem; text-align:right; }}
  th {{ background:#f5f5f5; }}
  td:first-child, th:first-child {{ text-align:left; }}
  td.cross {{ background:#ffeaea; font-weight:600; }}
  td.ev {{ color:#0a6; font-weight:600; }} td.dp {{ color:#888; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:.8rem; margin:1rem 0; }}
  .card {{ flex:1 1 220px; border:1px solid #e3e3e3; border-radius:8px; padding:.8rem 1rem; background:#fafafa; }}
  .card .v {{ font-size:1.5rem; font-weight:700; }}
  .card .k {{ font-size:.8rem; color:#666; }}
  .on {{ color:var(--on); }} .off {{ color:var(--off); }}
  img {{ max-width:100%; border:1px solid #eee; border-radius:8px; }}
  code {{ background:#f3f3f3; padding:.05rem .3rem; border-radius:4px; }}
  .key {{ background:#e8f5e9; border:1px solid #a5d6a7; border-radius:8px; padding:.8rem 1rem; margin:1rem 0; }}
</style></head><body>

<h1>Activation-space gaming: does loop-closure catch what the certificate can't see?</h1>
<div class="sub">{_esc(m['model'])} &middot; residual layer {m['layer']} &middot; mean-pooled &middot;
  d={m['d_model']} &middot; {m['n_anchors']} benign anchors &middot;
  tau={m['tau']:g} &middot; eps={m['eps']:g} (per-dim L&infin;) &middot;
  certificate: {_esc(m['cert_method'])} &middot; generated {ts}</div>

<div class="banner">
  <b>HONESTY:</b> the NLA verbalizer used for the re-decode readout is a transparent
  <b>STAND-IN</b> (retrieval + eval-axis; see <code>cnr/nla.py</code>), <i>not</i> a trained
  Natural Language Autoencoder. The GPT-2 activations, the distilled surrogate g, and the
  sound CROWN/IBP certificate are all real computation. "Genuine-crossing" displacements that
  the off-manifold route never reaches within the {m['delta_max']:g}&sigma; sweep are reported as
  &ge; {m['delta_max']:g}&sigma; (a sound lower bound on the true value), and flagged.
</div>

<div class="key">
<b>Finding.</b> Budget-projected gradient ascent <b>games the surrogate cheaply</b>:
g crosses tau at a <i>smaller</i> displacement off-manifold
(<b class="off">~{_fmt(a['g_cross_disp_median_off'])}&sigma;</b>) than it takes to interpolate into
the genuine eval cluster on-manifold (<b class="on">~{_fmt(a['g_cross_disp_median_on'])}&sigma;</b>).
But among the points where g&ge;tau, the re-decode reads as eval with probability
<b class="on">{_fmt(peval_on, 3)}</b> on-manifold versus <b class="off">{_fmt(peval_off, 3)}</b> off-manifold.
The off-manifold route satisfies the surrogate <i>without ever becoming genuinely eval</i> &mdash;
the loop-closure catches <b class="off">{_fmt(catch_off, 3)}</b> of those g&ge;tau points
(catch rate), and a genuinely-eval-reading state is reached on-manifold at
<b class="on">~{_fmt(a['genuine_cross_disp_median_on'])}&sigma;</b> but
<b class="off">never (&ge;{m['delta_max']:g}&sigma;)</b> off-manifold
({a['n_off_genuine_reached']}/{m['n_anchors']} off-manifold anchors ever reach it).
The certificate is locally sound but blind beyond its radius; the loop-closure is what bites.
</div>

<h2>Headline numbers</h2>
<div class="cards">
  <div class="card"><div class="k">g crosses tau at (median &sigma;)</div>
     <div class="v"><span class="on">{_fmt(a['g_cross_disp_median_on'])}</span>
       / <span class="off">{_fmt(a['g_cross_disp_median_off'])}</span></div>
     <div class="k">on-manifold / off-manifold &mdash; <b>off is cheaper</b></div></div>
  <div class="card"><div class="k">genuine cross (g&ge;tau &amp; readout=eval), median &sigma;</div>
     <div class="v"><span class="on">{_fmt(a['genuine_cross_disp_median_on'])}</span>
       / <span class="off">&ge;{_fmt(a['genuine_cross_disp_median_off'])}</span></div>
     <div class="k">on-manifold / off-manifold &mdash; <b>off never reaches it</b></div></div>
  <div class="card"><div class="k">P(readout=eval | g&ge;tau)</div>
     <div class="v"><span class="on">{_fmt(peval_on, 2)}</span>
       / <span class="off">{_fmt(peval_off, 2)}</span></div>
     <div class="k">on-manifold / off-manifold &mdash; <b>the gaming</b></div></div>
  <div class="card"><div class="k">loop-closure catch rate</div>
     <div class="v"><span class="off">{_fmt(catch_off, 2)}</span></div>
     <div class="k">frac of off-manifold g&ge;tau flagged NOT eval
       (on-manifold {_fmt(catch_on, 2)})</div></div>
</div>

<h2>Curves (mean over {m['n_anchors']} anchors)</h2>
<img src="{_esc(png_rel)}" alt="g, certificate status, and readout-agreement vs displacement">
<p class="sub">(1) g(h) vs displacement &mdash; off-manifold ascent crosses tau first.
(2) fraction of anchors still certified-unreachable at the operating eps &mdash; the certificate is a
<i>local</i> guarantee and is blind to displacements past its radius. (3) among g&ge;tau points, the
fraction whose re-decode reads as eval: on-manifold the readout agrees once you arrive; off-manifold
it never does. That gap (3) is the loop-closure catch.</p>

<h2>Per-anchor crossing displacements (&sigma; units)</h2>
<table>
  <tr><th>anchor</th><th>g(h0)</th>
      <th>g-cross ON</th><th>g-cross OFF</th>
      <th>readout-flip ON</th><th>readout-flip OFF</th>
      <th>genuine ON</th><th>genuine OFF</th></tr>
  {anchor_table}
</table>
<p class="sub">"g-cross" = first displacement with g&ge;tau. "readout-flip" = first displacement the
re-decode reads as eval. "genuine" = first displacement with <i>both</i>. Off-manifold g crosses early
but readout-flip / genuine are "never" within the sweep &mdash; the surrogate is gamed, the readout isn't.</p>

<h2>One anchor, full trajectory (anchor {pa0['anchor_id']})</h2>
<table>
  <tr><th rowspan="2">disp &sigma;</th>
      <th colspan="3">ON-manifold (interp &rarr; eval cluster)</th>
      <th colspan="3">OFF-manifold (ascent on g)</th></tr>
  <tr><th>g</th><th>cert?</th><th>readout</th>
      <th>g</th><th>cert?</th><th>readout</th></tr>
  {traj_table}
</table>
<p class="sub">Red g cells = g&ge;tau. "cert?" = certified_unreachable at eps={m['eps']:g}.
Watch the OFF column: g goes red (crosses) early and stays red, but the readout stays
<span class="dp">deploy</span> the whole way &mdash; gaming the certificate's surrogate without ever
reading as eval. ON-manifold, g and the readout cross together (within ~1&sigma;).</p>

<h2>What this demonstrates</h2>
<ul>
  <li><b>The certificate is sound but local.</b> Inside its radius g provably cannot cross tau; past
      it the certificate makes no claim. Both routes leave the certified ball, so on a bare
      "certified_unreachable?" test both look the same once you travel far enough.</li>
  <li><b>The surrogate is cheap to game.</b> Steepest ascent on g finds an off-manifold direction
      that crosses tau at <b class="off">~{_fmt(a['g_cross_disp_median_off'])}&sigma;</b>, well before
      the on-manifold route reaches the real eval cluster.</li>
  <li><b>Loop-closure is the discriminator.</b> Re-decoding the activation and asking "does this still
      read as eval?" separates the two: on-manifold yes, off-manifold no. The re-decode flags
      <b class="off">{_fmt(catch_off, 3)}</b> of the off-manifold g&ge;tau points as NOT eval. That is
      the off-manifold surrogate-gaming the certificate can't see, caught by closing the loop.</li>
</ul>

<p class="sub">Raw numbers: <code>artifacts/gaming/gaming_results.json</code> &middot;
figure: <code>{_esc(png_rel)}</code>. Reproduce: <code>python scripts/04_gaming_sweep.py</code>.</p>

</body></html>"""
    out_html.write_text(html)
    return out_html


# --------------------------------------------------------------------------- #
def main():
    cfg = Config()
    gaming_dir = cfg.artifacts / "gaming"
    gaming_dir.mkdir(parents=True, exist_ok=True)

    print("[gaming] running activation-space sweep on", cfg.model_name,
          "layer", cfg.layer, "(both routes, >=6 benign anchors)")
    results = run_experiment(cfg, n_anchors=6, delta_max=25.0, n_delta=26,
                             on_mode="centroid", ascent_steps=300)

    # save raw numbers
    res_path = save_results(results, gaming_dir)
    png_path = make_figure(results, gaming_dir / "gaming_curves.png")

    # timestamped report; figure copied next to the html so it is portable
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_png = cfg.reports / f"gaming_curves_{ts}.png"
    report_png.write_bytes((gaming_dir / "gaming_curves.png").read_bytes())
    out_html = cfg.reports / f"gaming_{ts}.html"
    write_html(results, report_png.name, out_html, ts)

    a = results["aggregate"]
    m = results["meta"]
    print("=" * 74)
    print("RESULTS  (activation-space gaming experiment)")
    print("=" * 74)
    print(f"  anchors                       : {m['n_anchors']} benign (deploy) acts")
    print(f"  certificate method            : {m['cert_method']}  (eps={m['eps']:g}, tau={m['tau']:g})")
    print(f"  g crosses tau at (median std) : ON {a['g_cross_disp_median_on']:.2f}   "
          f"OFF {a['g_cross_disp_median_off']:.2f}   (off is cheaper to game)")
    print(f"  genuine cross (g>=tau & eval) : ON {a['genuine_cross_disp_median_on']:.2f}   "
          f"OFF >= {a['genuine_cross_disp_median_off']:.2f}   "
          f"(off reached {a['n_off_genuine_reached']}/{m['n_anchors']})")
    print(f"  P(readout=eval | g>=tau)      : ON {a['p_readout_eval_given_gcross_on']:.3f}   "
          f"OFF {a['p_readout_eval_given_gcross_off']:.3f}   <- the gaming")
    print(f"  loop-closure catch rate       : ON {a['catch_rate_on']:.3f}   "
          f"OFF {a['catch_rate_off']:.3f}   "
          f"(frac g>=tau flagged NOT eval)")
    print(f"  off anchors: g crossed but readout NEVER flipped: "
          f"{a['off_anchors_g_crossed_readout_never']}/{a['off_anchors_g_crossed_total']}")
    print("=" * 74)
    print("RAW   :", res_path)
    print("FIG   :", png_path)
    print("REPORT:", out_html)
    return out_html


if __name__ == "__main__":
    main()
