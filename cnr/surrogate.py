"""The certifiable surrogate g(h) of the NLA-text classifier.

The classifier we actually care about is a map

    h  --NLA-->  natural-language text  --TextClf-->  log-odds(test-aware)

and Raghunathan's relaxation CANNOT be pushed soundly through a discrete text
decoder + an n-gram classifier. So we do the honest thing: **distil** that map
into a one-hidden-layer ReLU net g(h) on the *continuous* activation -- exactly the
architecture class the 2018 certificate was built for -- and certify g, while
MEASURING and REPORTING how faithfully g tracks the real classifier. g is the
certifiable shadow of the classifier; its fidelity is a number, not a hope.

g is a `cnr.probe.ReLUProbe`, so the existing verifier (cnr/verify.py) and the
inversion search (cnr/attack.py) consume it with no changes -- the surrogate is a
drop-in replacement for the old activation-space probe, now grounded in the NLA's
language readout instead of synthetic labels.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .probe import ReLUProbe, _fold_standardization


def distill_surrogate(
    X: np.ndarray,
    target_logit: np.ndarray,
    hard_label: np.ndarray,
    hidden: int = 64,
    epochs: int = 400,
    lr: float = 2e-3,
    lam_cls: float = 0.3,
    clamp: float = 12.0,
    seed: int = 0,
) -> tuple[ReLUProbe, dict]:
    """Fit g(h) ~= TextClf.logit(NLA(h)).

    Loss = MSE(g(x), clamp(target_logit)) + lam_cls * BCE(g(x), hard_label).
    The MSE makes g a faithful real-valued surrogate (good for a meaningful tau and
    for inversion gradients); the BCE term keeps the decision boundary crisp.
    Standardisation is folded in afterwards so g consumes RAW activations.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    t = np.clip(np.asarray(target_logit, dtype=np.float64), -clamp, clamp)
    c = np.asarray(hard_label, dtype=np.float64)
    n, d = X.shape

    perm = rng.permutation(n)
    n_val = max(1, n // 5)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    mu = X[tr_idx].mean(0)
    sigma = X[tr_idx].std(0) + 1e-6
    Xz = (X - mu) / sigma

    Xt = torch.tensor(Xz, dtype=torch.float32)
    tt = torch.tensor(t, dtype=torch.float32)
    ct = torch.tensor(c, dtype=torch.float32)

    g = ReLUProbe(d, hidden)
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    Xtr = Xt[tr_idx]
    for _ in range(epochs):
        opt.zero_grad()
        out = g(Xtr)
        loss = mse(out, tt[tr_idx]) + lam_cls * bce(out, ct[tr_idx])
        loss.backward()
        opt.step()

    g.eval()
    with torch.no_grad():
        pred_z = g(Xt).numpy()

    # --- fidelity to the real classifier (the honest headline numbers) ---
    def _corr(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    sign_agree = (np.sign(pred_z) == np.sign(t)).astype(float)
    meta = {
        "hidden": hidden,
        "tau": 0.0,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "fidelity_corr_all": _corr(pred_z, t),
        "fidelity_corr_val": _corr(pred_z[val_idx], t[val_idx]),
        "sign_agree_all": float(sign_agree.mean()),
        "sign_agree_val": float(sign_agree[val_idx].mean()),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
    }

    # fold standardisation -> g now consumes raw activations (verifier-ready)
    _fold_standardization(g, mu, sigma)
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)

    # sanity: folded raw scores match standardized scores
    with torch.no_grad():
        raw = g(torch.tensor(X, dtype=torch.float32)).numpy()
    assert np.allclose(raw, pred_z, atol=1e-3), "standardization fold mismatch"
    return g, meta
