"""Swap the TRAINED BART NLA into the top-half pipeline and re-measure the
circularity the retrieval stand-in caused.

Old stand-in: activation -> k-NN neighbour labels -> templated text (LEAKS the label).
Trained NLA:  activation -> BART (RL-reconstruction trained) -> text (no label in loop).

If classifier accuracy on the trained-NLA decode is far below the stand-in's ~1.00,
the old headline was circular and the trained NLA gives the honest, harder signal.
Writes reports/nla_swapin_<ts>.html and opens it.
"""
import datetime as _dt
import html
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnr.config import Config
from cnr.harness import ModelHarness
from cnr.nla import NLAVerbalizer
from cnr.nla_train import TrainedNLA
from cnr.text_classifier import NLATextClassifier, build_description_corpus


def _auroc(p, yb):
    order = np.argsort(p); ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = int(yb.sum()), int((1 - yb).sum())
    return float((ranks[yb == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    cfg = Config()
    harness = ModelHarness(cfg)
    bank = np.load(cfg.bank_path); X = bank["X"].astype(float); y = bank["y"].astype(int)
    texts = json.loads(cfg.bank_text_path.read_text())["texts"]
    eval_dir = X[y == 1].mean(0) - X[y == 0].mean(0)

    rng = np.random.default_rng(1)
    idx = np.concatenate([rng.choice(np.where(y == 1)[0], 32, replace=False),
                          rng.choice(np.where(y == 0)[0], 32, replace=False)])
    yb = y[idx]

    clf = NLATextClassifier(seed=0)
    ctxt, clab = build_description_corpus(seed=0); clf.fit(ctxt, clab)

    old = NLAVerbalizer(X, texts, y, eval_dir, k=7)
    old_dec = [old.decode(X[i], exclude_self=True).text for i in idx]

    nla = TrainedNLA(harness, k=16, max_tokens=28).load(cfg.artifacts / "nla_bart_proj.pt")
    new_dec = [nla.decode(X[i]) for i in idx]

    def metrics(dec):
        p = clf.proba(dec); pred = (p >= 0.5).astype(int)
        return {"acc": float((pred == yb).mean()), "auroc": _auroc(p, yb),
                "p_eval": float(p[yb == 1].mean()), "p_deploy": float(p[yb == 0].mean())}

    M = {"old": metrics(old_dec), "new": metrics(new_dec)}
    tlog = json.loads((cfg.artifacts / "nla_train_log.json").read_text())
    print("OLD stand-in :", M["old"])
    print("TRAINED NLA  :", M["new"])
    print(f"train recon cos {tlog['base_cos']:+.3f} -> {tlog['after_cos']:+.3f} "
          f"over {tlog['steps']} steps")

    # ---- report ----
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = cfg.reports / f"nla_swapin_{ts}.html"
    esc = html.escape

    def mrow(name, m):
        return ("<tr><td>" + name + "</td>"
                + f"<td>{m['acc']:.3f}</td><td>{m['auroc']:.3f}</td>"
                + f"<td>{m['p_eval']:.2f}</td><td>{m['p_deploy']:.2f}</td></tr>")

    drop = M["old"]["acc"] - M["new"]["acc"]
    verdict = (
        "The trained NLA <b>removes the circularity</b>: accuracy falls from "
        f"{M['old']['acc']:.2f} (stand-in, leaking the k-NN label) to {M['new']['acc']:.2f} "
        "(trained, label-free reconstruction). The easy 1.00 was the leak; this is the "
        "honest, harder number."
        if drop > 0.15 else
        "The trained NLA still tracks the label "
        f"(acc {M['new']['acc']:.2f} vs stand-in {M['old']['acc']:.2f}); on this coarse "
        "mean-pooled GPT-2 readout the decoded language retains some eval-vs-deploy signal.")

    ex = ("".join(
        f"<tr><td>{('eval' if y[i] else 'deploy')}</td>"
        f"<td><small>{esc(old.decode(X[i], exclude_self=True).text[:90])}</small></td>"
        f"<td><small>{esc(nla.decode(X[i]))}</small></td></tr>"
        for i in list(idx[:2]) + list(idx[-2:])))

    out.write_text(f"""<!doctype html><meta charset=utf-8><title>Trained NLA swap-in</title>
<style>body{{font:14px/1.5 -apple-system,sans-serif;max-width:900px;margin:32px auto;
padding:0 18px;color:#e6edf3;background:#0d1117}}h1{{font-size:21px}}h2{{font-size:15px;
border-bottom:1px solid #30363d;padding-bottom:4px;margin-top:24px}}table{{border-collapse:collapse;
width:100%}}th,td{{border:1px solid #30363d;padding:5px 9px;font-size:13px;text-align:left}}
th{{background:#161b22}}.v{{border:2px solid #3fb950;background:#11271a;border-radius:10px;
padding:14px 18px;margin:14px 0}}code{{background:#161b22;padding:1px 5px;border-radius:4px}}</style>
<h1>Trained GPT-2 NLA (BART base, RL) — swap-in vs the retrieval stand-in</h1>
<p style="color:#8b949e">Generated {ts}. The NLA is now an actual autoencoder: GPT-2 activation →
projection → frozen BART → text → fed back through GPT-2, rewarded by REINFORCE for centered-cosine
reconstruction. Held-out reconstruction cosine {tlog['base_cos']:+.3f} → {tlog['after_cos']:+.3f}
over {tlog['steps']} steps.</p>
<div class=v>{verdict}</div>
<h2>Classifier on the NLA output — stand-in vs trained</h2>
<table><tr><th>NLA</th><th>accuracy</th><th>AUROC</th><th>P(eval) on eval</th><th>P(eval) on deploy</th></tr>
{mrow('retrieval stand-in (leaks k-NN label)', M['old'])}
{mrow('trained BART NLA (label-free)', M['new'])}</table>
<p style="color:#8b949e">64-activation balanced subset. The stand-in's numbers are the circular
baseline; the trained NLA's are what the same eval-awareness classifier reads from genuinely
decoded content.</p>
<h2>Example decodes</h2>
<table><tr><th>class</th><th>retrieval stand-in</th><th>trained BART NLA</th></tr>{ex}</table>
<p style="color:#c9b89a;font-size:13px;border-top:1px solid #30363d;padding-top:10px">Honest note:
this trained NLA is a lightweight from-scratch autoencoder (only the activation→prefix projection
is trained; ~{tlog['steps']} RL steps on a laptop), not a validated published NLA — the parallel
investigation found no runnable GPT-2 NLA exists (smallest released is Qwen-7B, 15&nbsp;GB). It is,
however, model-specific and label-free, which is what the circularity fix requires.</p>""",
        encoding="utf-8")
    print("REPORT:", out)
    try:
        webbrowser.open(f"file://{out}")
    except Exception:
        pass
    return out


if __name__ == "__main__":
    main()
