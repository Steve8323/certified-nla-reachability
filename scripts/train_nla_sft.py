"""Train the GPT-2 NLA the stable way: supervised reconstruction warm-start, then
a gentle RL fine-tune (the paper's RL, but on a working decoder, not from scratch).
Keeps whichever checkpoint reconstructs held-out activations better.

    python scripts/train_nla_sft.py [sft_steps] [rl_steps]
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnr.config import Config
from cnr.harness import ModelHarness
from cnr.nla_train import TrainedNLA

MARKERS = ("evaluat", "benchmark", "test", "assess", "grad", "mmlu", "swe-bench",
           "red-team", "red team", "attempt", "capabilit", "score")


def _recon_metrics(nla, X, prompts, idx):
    decs = [nla.decode(X[i]) for i in idx]
    cos = float(nla.recon_cos(decs, X[idx]).mean())
    jac = []
    for d, i in zip(decs, idx):
        a, b = set(d.lower().split()), set(prompts[i].lower().split())
        jac.append(len(a & b) / max(1, len(a | b)))
    return cos, float(np.mean(jac)), decs


def main():
    sft_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    rl_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    cfg = Config()
    harness = ModelHarness(cfg)
    bank = np.load(cfg.bank_path)
    X, y = bank["X"].astype(np.float64), bank["y"].astype(int)
    texts = json.loads(cfg.bank_text_path.read_text())["texts"]

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    tr, ho = perm[:160], perm[160:174]

    nla = TrainedNLA(harness, k=16, max_tokens=28)
    nla.h_mean = X[tr].mean(0)
    print(f"[nla-sft] device={nla.device} base=bart-large sft={sft_steps} rl={rl_steps}")

    base_cos, base_jac, _ = _recon_metrics(nla, X, texts, ho)
    print(f"  BEFORE: held-out recon cos {base_cos:+.3f}  token-overlap {base_jac:.3f}")

    ce_hist = nla.train_supervised([texts[i] for i in tr], X[tr], steps=sft_steps, batch=8, lr=1e-4)
    sft_state = copy.deepcopy(nla.proj.state_dict())
    sft_cos, sft_jac, _ = _recon_metrics(nla, X, texts, ho)
    print(f"  AFTER SFT: recon cos {sft_cos:+.3f}  token-overlap {sft_jac:.3f}")

    rl_hist = nla.train(X[tr], steps=rl_steps, batch=6, lr=1e-5, rep_pen=0.5)
    rl_cos, rl_jac, _ = _recon_metrics(nla, X, texts, ho)
    print(f"  AFTER RL : recon cos {rl_cos:+.3f}  token-overlap {rl_jac:.3f}")

    if sft_cos >= rl_cos:
        nla.proj.load_state_dict(sft_state)
        chosen, after_cos, after_jac = "sft", sft_cos, sft_jac
    else:
        chosen, after_cos, after_jac = "sft+rl", rl_cos, rl_jac
    print(f"  KEEPING: {chosen}  (held-out recon cos {after_cos:+.3f}, Δ {after_cos-base_cos:+.3f})")

    # marker recall: do eval decodes contain eval-framing words? deploy false-positives?
    ev_ids = [i for i in ho if y[i] == 1]
    dp_ids = [i for i in ho if y[i] == 0]

    def marker_rate(ids):
        if not ids:
            return None
        return float(np.mean([any(m in nla.decode(X[i]).lower() for m in MARKERS) for i in ids]))
    ev_rate, dp_rate = marker_rate(ev_ids), marker_rate(dp_ids)
    print(f"  eval-marker recall: eval decodes {ev_rate}  deploy decodes {dp_rate}")

    nla.save(cfg.artifacts / "nla_bart_proj.pt")
    examples = []
    for lab in (1, 0):
        for i in np.where(y == lab)[0][:2]:
            examples.append({"true": int(lab), "decode": nla.decode(X[i]),
                             "source": texts[i][:90]})
    (cfg.artifacts / "nla_train_log.json").write_text(json.dumps({
        "method": "sft+gentle-rl, keep-best", "kept": chosen,
        "steps": sft_steps + (rl_steps if chosen == "sft+rl" else 0),
        "base_cos": base_cos, "after_cos": after_cos,
        "sft_cos": sft_cos, "rl_cos": rl_cos, "token_overlap": after_jac,
        "eval_marker_recall": ev_rate, "deploy_marker_fpr": dp_rate,
        "history": {"ce": ce_hist, "rl_reward": rl_hist.get("reward", [])},
        "examples": examples}, indent=2))
    print("  saved artifacts/nla_bart_proj.pt + nla_train_log.json")
    for e in examples:
        print(f"    [{'eval ' if e['true'] else 'deploy'}] {e['decode']!r}")
    print("[nla-sft] DONE")


if __name__ == "__main__":
    main()
