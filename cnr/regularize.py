"""Raghunathan's actual innovation: the certificate is DIFFERENTIABLE.

The IBP / CROWN upper bound U(theta; h0, eps) on  max_{B(h0,eps)} probe_theta(h)
is a function of the probe parameters theta, built from differentiable ops
(clamp, matmul, relu). So we can treat "the certified bound at the benign anchors"
as a TRAINING SIGNAL and push it down -- which moves the eval-aware region A* away
from benign deployment activations and ENLARGES the certified radius, exactly the
certified-defense move from Raghunathan/Steinhardt/Liang (ICLR/NeurIPS 2018).

We demonstrate the mechanism on the IBP bound (looser than CROWN but the cleanest
differentiable closed form). To keep the probe useful we add a penalty that keeps
its logits on the bank close to the originals, so accuracy is preserved.

THIS IS A MECHANISM DEMO, NOT A TUNED RESULT. A handful of gradient steps,
fixed hyper-parameters, CPU, no early stopping. The point is the direction of
the BEFORE -> AFTER change (radius up, accuracy roughly unchanged), not the
magnitude.
"""
from __future__ import annotations

import json

import numpy as np
import torch

# Work both as `python -m cnr.regularize` (package) and `python cnr/regularize.py`
# (script). In script mode there is no parent package for relative imports, so we
# add the repo root to sys.path and import absolutely.
if __package__ in (None, ""):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cnr.config import Config
    from cnr.probe import load_probe
    from cnr.verify import (
        AUTO_LIRPA_AVAILABLE,
        CertifiedVerifier,
        certified_radius,
        ibp_upper,
    )
else:
    from .config import Config
    from .probe import load_probe
    from .verify import (
        AUTO_LIRPA_AVAILABLE,
        CertifiedVerifier,
        certified_radius,
        ibp_upper,
    )


def _params_of(probe):
    """The four leaf tensors (W1,b1,W2,b2) of the probe, as the optimization vars."""
    lin0, lin1 = probe.net[0], probe.net[2]
    return lin0.weight, lin0.bias, lin1.weight, lin1.bias


def _ibp_upper_batch(W1, b1, W2, b2, h0, eps_vec):
    """Differentiable IBP upper bounds for a batch of anchors h0 (n,d).

    eps_vec: (d,) per-dim radius. Returns U (n,) -- bound for each anchor's box.
    """
    xl = h0 - eps_vec[None, :]
    xu = h0 + eps_vec[None, :]
    return ibp_upper(W1, b1, W2, b2, xl, xu)


def _bank_accuracy(probe, X, y, tau=0.0):
    with torch.no_grad():
        s = probe(torch.tensor(X, dtype=torch.float32))
        pred = (s >= tau).float()
    return float((pred == torch.tensor(y, dtype=torch.float32)).float().mean())


def _mean_certified_radius(probe, sigma, anchors_X, tau, grid):
    """CROWN certified radius averaged over anchors (uses the real verifier)."""
    verifier = CertifiedVerifier(probe, sigma, norm="inf", method="auto")
    radii = [certified_radius(verifier, anchors_X[i], tau, grid)
             for i in range(anchors_X.shape[0])]
    return float(np.mean(radii)), radii


def push_away_demo(steps: int = 150, lr: float = 5e-3, train_eps: float = 0.35,
                   lam_keep: float = 1.0, hinge_margin: float = 2.0, seed: int = 0):
    """Push A* away from benign anchors by descending the IBP bound; keep accuracy.

    Loss = mean_anchors hinge(U_ibp + margin)  +  lam_keep * MSE(logits_bank, frozen).
      - hinge(U + margin) = relu(U + margin): drives the certified UPPER bound at the
        anchors below -margin, i.e. deeper into "certified unreachable".
      - the keep-term pins bank logits to their original values so the decision
        boundary (hence accuracy) barely moves.
    """
    torch.manual_seed(seed)
    cfg = Config()

    probe0, meta = load_probe(cfg.probe_path)
    sigma = np.asarray(meta["sigma"], dtype=np.float64)
    tau = float(meta.get("tau", 0.0))

    bank = np.load(cfg.bank_path)
    X = bank["X"].astype(np.float32)
    y = bank["y"].astype(np.float32)
    with open(cfg.meta_path) as f:
        mj = json.load(f)
    anchor_ids = mj["anchor_ids"]
    anchors_X = X[anchor_ids]                                   # (n, d)

    grid = np.linspace(0.02, 2.0, 25)
    eps_vec_np = train_eps * sigma

    # ---- clone the probe so we never touch the saved artifact ----
    probe = load_probe(cfg.probe_path)[0]
    for p in probe.parameters():
        p.requires_grad_(True)
    W1, b1, W2, b2 = _params_of(probe)

    # ---- BEFORE metrics ----
    acc_before = _bank_accuracy(probe0, X, y, tau)
    radius_before, radii_before = _mean_certified_radius(probe0, sigma, anchors_X, tau, grid)

    # frozen reference logits for the keep-accuracy penalty
    with torch.no_grad():
        logits_ref = probe0(torch.tensor(X, dtype=torch.float32)).detach()

    Xt = torch.tensor(X, dtype=torch.float32)
    anchors_t = torch.tensor(anchors_X, dtype=torch.float32)
    eps_vec = torch.tensor(eps_vec_np, dtype=torch.float32)

    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        U = _ibp_upper_batch(W1, b1, W2, b2, anchors_t, eps_vec)     # (n,)
        push = torch.relu(U + hinge_margin).mean()                  # drive U below -margin
        keep = ((probe(Xt) - logits_ref) ** 2).mean()              # preserve logits
        loss = push + lam_keep * keep
        loss.backward()
        opt.step()

    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)

    # ---- AFTER metrics ----
    acc_after = _bank_accuracy(probe, X, y, tau)
    radius_after, radii_after = _mean_certified_radius(probe, sigma, anchors_X, tau, grid)

    method_label = "CROWN (auto_LiRPA)" if AUTO_LIRPA_AVAILABLE else "CROWN (native)"
    print("=" * 74)
    print("DIFFERENTIABLE-CERTIFICATE DEMO  (Raghunathan 2018 mechanism)")
    print("  push A* away from benign anchors by descending the IBP upper bound;")
    print("  keep-term pins bank logits so accuracy is preserved.")
    print(f"  steps={steps} lr={lr} train_eps={train_eps} hinge_margin={hinge_margin} "
          f"lam_keep={lam_keep}")
    print(f"  radius measured with {method_label}; sweep grid {grid[0]:g}..{grid[-1]:g}")
    print("-" * 74)
    print(f"{'':22} | {'BEFORE':>10} | {'AFTER':>10} | {'delta':>10}")
    print(f"{'bank accuracy':22} | {acc_before:>10.4f} | {acc_after:>10.4f} | "
          f"{acc_after-acc_before:>+10.4f}")
    print(f"{'mean cert. radius':22} | {radius_before:>10.4f} | {radius_after:>10.4f} | "
          f"{radius_after-radius_before:>+10.4f}")
    print("-" * 74)
    print(f"per-anchor radius before: {[round(r,3) for r in radii_before]}")
    print(f"per-anchor radius after : {[round(r,3) for r in radii_after]}")
    print("-" * 74)
    print("MECHANISM DEMO, not a tuned result: the certificate is differentiable, so")
    print("descending it provably enlarges the certified-unreachable radius while the")
    print("keep-term holds accuracy ~constant. Magnitudes depend on the few fixed knobs.")
    return {
        "acc_before": acc_before, "acc_after": acc_after,
        "radius_before": radius_before, "radius_after": radius_after,
        "radii_before": radii_before, "radii_after": radii_after,
    }


if __name__ == "__main__":
    push_away_demo()
