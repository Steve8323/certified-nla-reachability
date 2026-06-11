"""Stage 1: build the eval-awareness probe = target set A*.

Produces, in artifacts/:
    bank.npz       X (activations), y (labels)
    bank_texts.json
    probe.pt       folded ReLU probe over raw activations
    meta.json      accuracy, tau/tau_hi, anchors, config echo
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnr.config import Config
from cnr.data import build_dataset, save_texts
from cnr.harness import ModelHarness
from cnr.probe import load_probe, save_probe, train_probe


def main() -> None:
    cfg = Config()
    print(f"[1/5] loading {cfg.model_name} (layer {cfg.layer}, device {cfg.device})")
    harness = ModelHarness(cfg)

    print(f"[2/5] building dataset ({cfg.n_per_class}/class)")
    texts, labels = build_dataset(cfg.n_per_class, cfg.seed)
    save_texts(cfg.bank_text_path, texts, labels)

    print("[3/5] extracting activations")
    X = harness.activations(texts).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    np.savez(cfg.bank_path, X=X, y=y)
    print(f"      activations: {X.shape}, eval={int(y.sum())} deploy={int((1-y).sum())}")

    print("[4/5] training ReLU probe (defines A*)")
    probe, meta = train_probe(
        X, y, hidden=cfg.hidden, epochs=cfg.probe_epochs, lr=cfg.probe_lr, seed=cfg.seed
    )
    save_probe(cfg.probe_path, probe, meta)

    # choose anchors: benign (deploy) activations are the h0 we certify around
    deploy_ids = np.where(y == 0)[0][:8].tolist()
    eval_centroid = X[y == 1].mean(0)

    full_meta = {
        "model": cfg.model_name,
        "layer": cfg.layer,
        "d_model": int(X.shape[1]),
        "hidden": cfg.hidden,
        "eps": cfg.eps,
        "norm": cfg.norm,
        "train_acc": meta["train_acc"],
        "val_acc": meta["val_acc"],
        "tau": meta["tau"],
        "tau_hi": meta["tau_hi"],
        "score_eval_mean": meta["score_eval_mean"],
        "score_deploy_mean": meta["score_deploy_mean"],
        "anchor_ids": deploy_ids,
        "eval_centroid": eval_centroid.tolist(),
    }
    with open(cfg.meta_path, "w") as f:
        json.dump(full_meta, f, indent=2)

    print("[5/5] verifying reload + folding")
    probe2, _ = load_probe(cfg.probe_path)
    import torch

    with torch.no_grad():
        s = probe2(torch.tensor(X)).numpy()
    sep = s[y == 1].mean() - s[y == 0].mean()
    print(f"      val_acc={meta['val_acc']:.3f} train_acc={meta['train_acc']:.3f}")
    print(f"      score separation (eval-deploy)={sep:.2f}  tau={meta['tau']}  tau_hi={meta['tau_hi']:.2f}")
    print("done. artifacts in", cfg.artifacts)


if __name__ == "__main__":
    main()
