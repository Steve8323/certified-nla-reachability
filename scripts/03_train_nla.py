"""Train the GPT-2 Natural Language Autoencoder (BART base decoder, RL).

    python scripts/03_train_nla.py [steps]

Saves artifacts/nla_bart_proj.pt + artifacts/nla_train_log.json. Reports held-out
reconstruction cosine BEFORE vs AFTER training (the autoencoder objective working).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnr.config import Config
from cnr.harness import ModelHarness
from cnr.nla_train import TrainedNLA


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    cfg = Config()
    harness = ModelHarness(cfg)
    bank = np.load(cfg.bank_path)
    X, y = bank["X"].astype(np.float64), bank["y"].astype(int)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    tr, ho = perm[:160], perm[160:174]                 # train / held-out activations
    Xtr, Xho = X[tr], X[ho]

    nla = TrainedNLA(harness, k=8, max_tokens=18)
    nla.h_mean = Xtr.mean(0)                            # center reward on the train bank
    print(f"[nla-train] device={nla.device}  base=bart-large  steps={steps}  "
          f"train={len(Xtr)} heldout={len(Xho)}")

    base_cos = float(nla.recon_cos([nla.decode(h) for h in Xho], Xho).mean())
    print(f"  held-out greedy reconstruction cosine BEFORE: {base_cos:+.3f}")

    hist = nla.train(Xtr, steps=steps, batch=6, lr=1e-4, rep_pen=0.5)

    after_cos = float(nla.recon_cos([nla.decode(h) for h in Xho], Xho).mean())
    print(f"  held-out greedy reconstruction cosine AFTER : {after_cos:+.3f}  "
          f"(Δ {after_cos - base_cos:+.3f})")

    nla.save(cfg.artifacts / "nla_bart_proj.pt")
    examples = []
    for lab in (1, 0):
        for i in np.where(y == lab)[0][:2]:
            examples.append({"true": int(lab), "decode": nla.decode(X[i]),
                             "source": json.loads(cfg.bank_text_path.read_text())["texts"][i][:80]})
    log = {"steps": steps, "base_cos": base_cos, "after_cos": after_cos,
           "history": hist, "examples": examples}
    (cfg.artifacts / "nla_train_log.json").write_text(json.dumps(log, indent=2))
    print("  saved artifacts/nla_bart_proj.pt + nla_train_log.json")
    print("  example decodes:")
    for e in examples:
        print(f"    [{'eval ' if e['true'] else 'deploy'}] {e['decode']!r}")
    print("[nla-train] DONE")


if __name__ == "__main__":
    main()
