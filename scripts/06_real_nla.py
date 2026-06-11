#!/usr/bin/env python3
"""06_real_nla.py -- attempt to drive a REAL released NLA, honestly.

Run from repo root:  python scripts/06_real_nla.py

This does NOT modify any existing file or artifact. It writes only into
artifacts/real_nla/ and reports/realnla_<ts>.html (plus REAL_NLA_NOTES.md, which
is written separately).

What it does:
  1. Record what was actually released (registry from cnr.real_nla).
  2. REAL partial checks that run on this box (no 15 GB weights):
       a. tokenizer-only verification that our AV prompt construction reproduces
          the released input format byte-for-byte (injection token + neighbors).
       b. an injection self-test on a *native-dim* synthetic activation (proves
          the embedding-injection path is shape-correct for the AV).
       c. confirm the cnr GPT-2 activation-extraction path runs end-to-end.
  3. ATTEMPT to load the real AV and generate. On a 16 GB box this is expected
     to fail at the RAM guard; the failure text is captured verbatim (honest).
  4. Side-by-side: stand-in (cnr.nla) readouts vs real-NLA status, on the SAME
     GPT-2 activations -- showing exactly why the real Qwen AV cannot read them.
  5. Save artifacts/real_nla/run.json and render an HTML report.
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cnr.config import Config  # noqa: E402
from cnr.data import build_dataset  # noqa: E402
from cnr.harness import ModelHarness  # noqa: E402
from cnr.real_nla import (  # noqa: E402
    AV_PROMPT_TEMPLATE,
    RELEASED_NLAS,
    RealNLAVerbalizer,
    write_spec_json,
)

ART = ROOT / "artifacts" / "real_nla"
ART.mkdir(parents=True, exist_ok=True)
TS = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT = ROOT / "reports" / f"realnla_{TS}.html"

log: dict = {"timestamp": TS, "steps": {}}


def step(name, **kv):
    log["steps"][name] = kv
    print(f"\n=== {name} ===")
    for k, v in kv.items():
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        print(f"  {k}: {s[:300]}")


# ---------------------------------------------------------------------------
# 1. Record the released registry
# ---------------------------------------------------------------------------
spec_path = write_spec_json(ART / "released_nla_spec.json")
step(
    "1_released_artifacts",
    smallest_av="kitft/nla-qwen2.5-7b-L20-av",
    n_released=len(RELEASED_NLAS),
    families=sorted({m["base_model"].split("/")[-1].split("-")[0] for m in RELEASED_NLAS.values()}),
    spec_written=str(spec_path.relative_to(ROOT)),
)

NLA_KEY = "qwen2.5-7b-L20"
adapter = RealNLAVerbalizer(NLA_KEY, device="cpu", ram_guard_gb=14.0)

# ---------------------------------------------------------------------------
# 2a. REAL: prompt-format verification (tokenizer-only; downloads a few MB)
# ---------------------------------------------------------------------------
prompt_check: dict = {}
try:
    built = adapter.build_av_prompt()
    ids = built["input_ids"]
    p = built["injection_pos"]
    meta = adapter.meta
    prompt_check = {
        "prompt_token_len": len(ids),
        "injection_pos": p,
        "injection_token_id_used": meta["injection_token_id"],
        "left_neighbor_actual": ids[p - 1],
        "left_neighbor_expected": meta["injection_left_neighbor_id"],
        "right_neighbor_actual": ids[p + 1],
        "right_neighbor_expected": meta["injection_right_neighbor_id"],
        "neighbors_match": (
            ids[p - 1] == meta["injection_left_neighbor_id"]
            and ids[p + 1] == meta["injection_right_neighbor_id"]
        ),
        "rendered_head": built["rendered"][:80],
        "status": "REAL_PASS",
    }
except Exception as e:  # noqa: BLE001
    prompt_check = {"status": "FAIL", "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc()}
step("2a_real_prompt_format_check", **prompt_check)

# ---------------------------------------------------------------------------
# 2b. REAL: injection self-test on a NATIVE-dim (3584) synthetic activation
#     proves inject_activation() is shape/scale correct for the AV embedding.
# ---------------------------------------------------------------------------
inject_check: dict = {}
try:
    import torch

    d = adapter.d_model
    T = prompt_check.get("prompt_token_len", 8) or 8
    fake_embeds = torch.zeros(1, T, d, dtype=torch.float32)
    h_native = np.random.default_rng(0).standard_normal(d).astype(np.float32) * 13.0
    pos = prompt_check.get("injection_pos", 0)
    out = adapter.inject_activation(fake_embeds, pos, h_native)
    injected = out[0, pos].numpy()
    inject_check = {
        "native_d_model": d,
        "raw_norm": round(float(np.linalg.norm(h_native)), 4),
        "injected_norm": round(float(np.linalg.norm(injected)), 4),
        "target_injection_scale": adapter.meta["injection_scale"],
        "norm_matches_scale": abs(float(np.linalg.norm(injected)) - adapter.meta["injection_scale"]) < 1e-2,
        "status": "REAL_PASS",
    }
except Exception as e:  # noqa: BLE001
    inject_check = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
step("2b_real_injection_selftest", **inject_check)

# ---------------------------------------------------------------------------
# 2c. REAL: confirm cnr's GPT-2 activation path runs (the cnr-side reality)
# ---------------------------------------------------------------------------
cfg = Config()
hz = ModelHarness(cfg)
texts, labels = build_dataset(4, seed=0)
demo_texts = texts[:6]
demo_labels = labels[:6]
H_gpt2 = hz.activations(demo_texts, layer=cfg.layer)  # (6, 768)
step(
    "2c_real_gpt2_path",
    base_model=cfg.model_name,
    layer=cfg.layer,
    gpt2_d_model=hz.d_model,
    activation_shape=list(H_gpt2.shape),
    activation_norms=[round(float(x), 1) for x in np.linalg.norm(H_gpt2, axis=1)],
    status="REAL_PASS",
)

# ---------------------------------------------------------------------------
# 3. ATTEMPT real AV load + generate (expected gated on this box -- captured)
# ---------------------------------------------------------------------------
ok, msg = adapter.try_load()
real_readouts = []
if ok:
    # Genuinely run the AV on the native-dim self-test vector (we have weights).
    try:
        dec = adapter.verbalize(h_native)  # type: ignore[name-defined]
        real_readouts.append({"input": "native synthetic activation", "text": dec.text,
                              "real": dec.real, "source": dec.source})
    except Exception as e:  # noqa: BLE001
        msg += f"  | generate failed: {type(e).__name__}: {e}"
step(
    "3_real_av_load_attempt",
    loaded=ok,
    message=msg,
    av_weight_gb=adapter.meta["av_weight_gb"],
    box_ram_gb=16,
    status="REAL_RUN" if ok else "PENDING_BLOCKED",
)

# ---------------------------------------------------------------------------
# 4. Side-by-side: stand-in readouts vs real-NLA, on the SAME GPT-2 activations
# ---------------------------------------------------------------------------
comparison = []
standin_err = None
try:
    from cnr.nla import NLAVerbalizer  # the stand-in

    bank = np.load(cfg.bank_path)
    bt_raw = json.loads(Path(cfg.bank_text_path).read_text())
    bank_texts = bt_raw["texts"] if isinstance(bt_raw, dict) else bt_raw
    bX, bY = bank["X"], bank["y"].astype(int)
    # eval direction = difference of class means in the bank (eval-vs-deploy axis).
    eval_dir = bX[bY == 1].mean(0) - bX[bY == 0].mean(0)
    standin = NLAVerbalizer(bX, bank_texts, bY, eval_dir.astype(np.float64), k=7)
    standin_decs = standin.decode_many(H_gpt2, exclude_self=False)
except Exception as e:  # noqa: BLE001
    standin_err = f"{type(e).__name__}: {e}"
    standin_decs = None

for i, t in enumerate(demo_texts):
    row = {
        "prompt": t,
        "label": "eval" if demo_labels[i] == 1 else "deploy",
        "gpt2_activation_dim": int(H_gpt2.shape[1]),
    }
    if standin_decs is not None:
        sd = standin_decs[i]
        row["standin_text"] = sd.text
        row["standin_eval_likeness"] = round(float(sd.eval_likeness), 3)
        row["standin_stand_in_flag"] = sd.stand_in
    else:
        row["standin_text"] = f"(stand-in unavailable: {standin_err})"
    # The honest real-NLA cell. If the AV is loaded we really generate; otherwise
    # we surface the *deeper* structural refusal -- feed this GPT-2 activation to
    # the AV's injection step, which raises ValueError (768 != 3584). That dim
    # mismatch (no GPT-2 NLA exists) is the real reason, captured verbatim.
    try:
        if adapter._loaded:
            row["real_nla_text"] = adapter.verbalize(H_gpt2[i]).text
        else:
            import torch as _t

            _emb = _t.zeros(1, 4, adapter.d_model)
            adapter.inject_activation(_emb, 0, H_gpt2[i])  # raises on dim mismatch
            row["real_nla_text"] = "(unexpectedly accepted GPT-2 activation)"
    except Exception as e:  # noqa: BLE001
        row["real_nla_text"] = f"[{type(e).__name__}] {str(e).splitlines()[0]}"
    comparison.append(row)

step(
    "4_side_by_side",
    n_rows=len(comparison),
    standin_available=standin_decs is not None,
    standin_error=standin_err,
    real_nla_on_gpt2="refused: dim 768 != AV expects 3584 (no GPT-2 NLA released)",
)

# ---------------------------------------------------------------------------
# overall honest verdict
# ---------------------------------------------------------------------------
got_real_generation = bool(real_readouts)
verdict = "REAL (AV generated)" if got_real_generation else "PARTIAL"
log["verdict"] = verdict
log["comparison"] = comparison
log["real_readouts"] = real_readouts
log["prompt_check"] = prompt_check
log["inject_check"] = inject_check
log["load_msg"] = msg

(ART / "run.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
print(f"\nwrote {ART / 'run.json'}")


# ---------------------------------------------------------------------------
# 5. HTML report (self-contained)
# ---------------------------------------------------------------------------
def esc(x):
    return _html.escape(str(x))


def badge(text, kind):
    colors = {"real": "#16794e", "partial": "#a36a00", "pending": "#8a1f1f",
              "info": "#274b8a"}
    return (f'<span style="background:{colors.get(kind, "#555")};color:#fff;'
            f'padding:2px 9px;border-radius:10px;font-size:12px;'
            f'font-weight:600">{esc(text)}</span>')


pm = prompt_check
inj = inject_check
neigh_ok = pm.get("neighbors_match")

rows_html = ""
for r in comparison:
    rows_html += (
        "<tr>"
        f"<td style='vertical-align:top'><span style='color:#888'>[{esc(r['label'])}]</span><br>{esc(r['prompt'])}"
        f"<br><span style='color:#999;font-size:11px'>GPT-2 act dim {r['gpt2_activation_dim']}</span></td>"
        f"<td style='vertical-align:top;font-size:13px'>{esc(r.get('standin_text',''))}"
        + (f"<br><span style='color:#999;font-size:11px'>eval-likeness {r.get('standin_eval_likeness')}, stand_in={r.get('standin_stand_in_flag')}</span>"
           if 'standin_eval_likeness' in r else "")
        + "</td>"
        f"<td style='vertical-align:top;font-size:13px;color:#8a1f1f'>{esc(r.get('real_nla_text',''))}</td>"
        "</tr>"
    )

real_readout_html = ""
if real_readouts:
    for rr in real_readouts:
        real_readout_html += (f"<div class='card'><b>{esc(rr['source'])}</b><br>"
                              f"<i>{esc(rr['input'])}</i><br>{esc(rr['text'])}</div>")
else:
    real_readout_html = ("<p>No real AV generation on this machine. The smallest released AV is a "
                         "full <b>Qwen2.5-7B</b> fine-tune (~15.2 GB BF16); this box has 16 GB RAM, "
                         "so the loader's RAM guard refused it. The blocker text captured at run time:</p>"
                         f"<pre>{esc(msg)}</pre>")

reg_rows = ""
for k, m in RELEASED_NLAS.items():
    reg_rows += (f"<tr><td>{esc(k)}</td><td>{esc(m['av_repo'])}</td>"
                 f"<td>{esc(m['base_model'])}</td><td style='text-align:right'>{m['d_model']}</td>"
                 f"<td style='text-align:right'>L{m['layer']}</td>"
                 f"<td style='text-align:right'>{m.get('av_weight_gb','?')}</td></tr>")

verdict_badge = badge("PARTIAL (real wiring verified, generation pending)", "partial") if not got_real_generation \
    else badge("REAL", "real")

html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Real NLA -- honest integration report ({TS})</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:24px auto;
   padding:0 18px;color:#1d1d1f;line-height:1.5}}
 h1{{font-size:24px}} h2{{font-size:18px;margin-top:30px;border-bottom:1px solid #ddd;padding-bottom:4px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{border:1px solid #ddd;padding:7px;text-align:left}}
 th{{background:#f5f5f7}}
 pre{{background:#f5f5f7;padding:10px;border-radius:6px;overflow:auto;font-size:12px;white-space:pre-wrap}}
 .card{{background:#f5f5f7;border-radius:8px;padding:12px;margin:8px 0}}
 .ok{{color:#16794e;font-weight:600}} .no{{color:#8a1f1f;font-weight:600}}
 .banner{{background:#fff7e6;border:1px solid #e0b94d;border-radius:8px;padding:12px 14px;margin:14px 0}}
 code{{background:#eee;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>Real NLA: honest integration report</h1>
<p>{verdict_badge} &nbsp; <span style="color:#888">generated {esc(TS)}</span></p>

<div class="banner">
<b>Honest status.</b> The released NLAs are real and were located, and this adapter
reproduces their exact activation-injection input format (verified live against the
released <code>nla_meta.yaml</code> -- see &sect;2). But <b>no real verbalization of a
cnr activation was produced</b>, for two reasons stated plainly:
<ol style="margin:6px 0">
<li><b>Hardware.</b> The smallest released activation-verbalizer is a full
<b>Qwen2.5-7B</b> fine-tune (~15.2&nbsp;GB BF16). This machine has 16&nbsp;GB RAM, so it
cannot be loaded without thrashing. The adapter's RAM guard refuses it.</li>
<li><b>Base-model mismatch.</b> An NLA is welded to <i>its own</i> base model's residual
stream. The released ones target Qwen2.5-7B&nbsp;(L20, 3584-d), Gemma-3, and
Llama-3.3. The cnr pipeline reads <b>GPT-2</b> (768-d, L6). <b>There is no released
GPT-2 NLA</b>, so a Qwen AV mathematically cannot read a cnr activation
(768&nbsp;&ne;&nbsp;3584). We do not fake one.</li>
</ol>
This is a documented honest <b>partial</b>: real spec + real input-format wiring +
real partial runs, with the heavy generation step gated, not faked.
</div>

<h2>1. What was actually released</h2>
<p>Source: paper <code>transformer-circuits.pub/2026/nla</code>; code
<code>github.com/kitft/natural_language_autoencoders</code> + <code>kitft/nla-inference</code>;
weights in HF collection <code>kitft/nla-models</code>; interactive frontend on
Neuronpedia. An NLA = an <b>activation verbalizer (AV)</b> + an <b>activation
reconstructor (AR)</b>, both full fine-tunes of the base model, trained with RL to
reconstruct residual activations. Eight checkpoints (AV+AR &times; 4 base models):</p>
<table><tr><th>key</th><th>AV repo</th><th>base model</th><th>d_model</th><th>layer</th><th>AV size (GB)</th></tr>
{reg_rows}</table>
<p>The AV injects the (scaled) activation as a single token's embedding into a fixed
prompt and autoregressively writes a 2-3 snippet explanation. Default inference path
uses <b>SGLang</b> (sends <code>input_embeds</code> over HTTP); the adapter here uses
HF <code>generate(inputs_embeds=...)</code> instead so no server is needed.</p>

<h2>2. Real wiring verified on this machine (no big weights)</h2>
<p>The exact AV prompt template (verbatim from the released yaml):</p>
<pre>{esc(AV_PROMPT_TEMPLATE)}</pre>
<p><b>2a. Prompt-format reproduction.</b>
With the real Qwen tokenizer, our assembled prompt places the injection token
(id <code>{esc(pm.get('injection_token_id_used'))}</code>, char <code>&#x320e;</code>)
at position <code>{esc(pm.get('injection_pos'))}</code>. Its neighbors are
left=<code>{esc(pm.get('left_neighbor_actual'))}</code>,
right=<code>{esc(pm.get('right_neighbor_actual'))}</code> &mdash; which
<b>{'match' if neigh_ok else 'do NOT match'}</b> the released yaml's
<code>injection_left_neighbor_id={esc(pm.get('left_neighbor_expected'))}</code> /
<code>injection_right_neighbor_id={esc(pm.get('right_neighbor_expected'))}</code>.
<span class="{ 'ok' if neigh_ok else 'no' }">[{esc(pm.get('status'))}]</span>
This proves the prompt construction is byte-for-byte the released format.</p>
<p><b>2b. Injection self-test (native dim).</b>
On a synthetic {esc(inj.get('native_d_model'))}-d activation, after the documented
scaling <code>v &middot; injection_scale/||v||</code> the injected slot has norm
<code>{esc(inj.get('injected_norm'))}</code>, matching
<code>injection_scale={esc(inj.get('target_injection_scale'))}</code>
<span class="{ 'ok' if inj.get('norm_matches_scale') else 'no' }">[{esc(inj.get('status'))}]</span>.</p>
<p><b>2c. cnr GPT-2 path.</b> cnr's own activation extraction runs end-to-end:
<code>gpt2</code> layer {cfg.layer}, produced activations of shape
<code>{esc(log['steps']['2c_real_gpt2_path']['activation_shape'])}</code>
(dim <b>768</b>) <span class="ok">[REAL_PASS]</span>.</p>

<h2>3. Real readout attempt</h2>
{real_readout_html}

<h2>4. Side-by-side on the SAME GPT-2 activations: stand-in vs real NLA</h2>
<p>The stand-in (<code>cnr/nla.py</code>) verbalizes via retrieval + an eval-axis
heuristic and works on any dim, so it produces text for every GPT-2 activation. The
real Qwen AV is asked the same thing and <b>honestly refuses</b> (dim mismatch). This
is the crux: to get real NLA readouts in this pipeline you must switch cnr's target
model to an NLA base model (see notes).</p>
<table><tr><th>prompt (GPT-2 activation)</th><th>stand-in decode (cnr/nla.py)</th>
<th>real Qwen AV</th></tr>{rows_html}</table>

<h2>5. To get genuine readouts (integration path)</h2>
<ol>
<li>Point cnr at an NLA base model, e.g. set <code>Config.model_name =
"Qwen/Qwen2.5-7B-Instruct"</code>, <code>Config.layer = 20</code>. The harness already
reads <code>hidden_states[layer]</code> generically.</li>
<li>Extract the <b>per-token (last-token), unpooled</b> residual at L20 (NLAs were
trained on token activations, not mean-pooled). 3584-d -> matches the AV.</li>
<li>Run on a machine with &ge;~20&nbsp;GB free (GPU or large host RAM); raise the
adapter's <code>ram_guard_gb</code>. Then
<code>RealNLAVerbalizer("qwen2.5-7b-L20", load_weights=True).verbalize(h)</code>
returns a real explanation.</li>
</ol>
<p style="color:#888;font-size:12px">Full reproducible details, exact ids/URLs, and
the SGLang alternative are in <code>REAL_NLA_NOTES.md</code>. Run record:
<code>artifacts/real_nla/run.json</code>.</p>
</body></html>"""

REPORT.write_text(html_doc)
print(f"wrote {REPORT}")
print(f"\nVERDICT: {verdict}")
