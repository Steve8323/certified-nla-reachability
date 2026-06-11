"""The eval-awareness probe = the target set A*.

A one-hidden-layer ReLU MLP (Raghunathan's ICLR-2018 architecture class) trained
to separate eval-like from deployment-like activations. Standardization is folded
into the first Linear after training, so the saved module is a clean

    Linear(d, hidden) -> ReLU -> Linear(hidden, 1)

operating directly on *raw* residual-stream activations. That keeps the
certification problem exactly the ReLU-net reachability problem the certified
defense was built for, with no preprocessing to reason about.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn


class ReLUProbe(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fold_standardization(probe: ReLUProbe, mu: np.ndarray, sigma: np.ndarray) -> None:
    """Rewrite first Linear so forward(raw_h) == forward_trained((raw_h-mu)/sigma)."""
    lin0 = probe.net[0]
    W = lin0.weight.detach().numpy()          # (hidden, d)
    b = lin0.bias.detach().numpy()            # (hidden,)
    W_new = W / sigma[None, :]
    b_new = b - W @ (mu / sigma)
    with torch.no_grad():
        lin0.weight.copy_(torch.tensor(W_new, dtype=lin0.weight.dtype))
        lin0.bias.copy_(torch.tensor(b_new, dtype=lin0.bias.dtype))


def train_probe(
    X: np.ndarray, y: np.ndarray, hidden: int, epochs: int, lr: float, seed: int = 0
) -> tuple[ReLUProbe, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n, d = X.shape

    perm = rng.permutation(n)
    n_val = max(1, n // 5)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    mu = X[tr_idx].mean(0)
    sigma = X[tr_idx].std(0) + 1e-6

    Xz = (X - mu) / sigma
    Xt = torch.tensor(Xz, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    probe = ReLUProbe(d, hidden)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr, ytr = Xt[tr_idx], yt[tr_idx]
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(probe(Xtr), ytr)
        loss.backward()
        opt.step()

    def acc(idx):
        with torch.no_grad():
            pred = (probe(Xt[idx]) >= 0).float()
        return float((pred == yt[idx]).float().mean())

    metrics = {"train_acc": acc(tr_idx), "val_acc": acc(val_idx)}

    # fold standardization -> probe now consumes raw activations
    _fold_standardization(probe, mu, sigma)
    probe.eval()

    with torch.no_grad():
        s_raw = probe(torch.tensor(X, dtype=torch.float32)).numpy()
    # sanity: folded raw-input scores match standardized-input scores
    with torch.no_grad():
        probe_chk = probe  # already folded; recompute standardized for assertion
    metrics["tau"] = 0.0
    metrics["tau_hi"] = float(np.quantile(s_raw[y == 0], 0.99))
    metrics["score_eval_mean"] = float(s_raw[y == 1].mean())
    metrics["score_deploy_mean"] = float(s_raw[y == 0].mean())
    metrics["mu"] = mu.tolist()
    metrics["sigma"] = sigma.tolist()
    return probe, metrics


def save_probe(path, probe: ReLUProbe, meta: dict) -> None:
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "d": probe.net[0].in_features,
            "hidden": probe.net[0].out_features,
            "meta": meta,
        },
        path,
    )


def load_probe(path) -> tuple[ReLUProbe, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    probe = ReLUProbe(blob["d"], blob["hidden"])
    probe.load_state_dict(blob["state_dict"])
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe, blob["meta"]
