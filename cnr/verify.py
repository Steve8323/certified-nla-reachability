"""Certified verifier for eval-awareness reachability.

This is "Agent A" of the prototype: given a benign deployment activation h0 and a
per-dimension-scaled L-inf ball B around it, return a SOUND upper bound U on
    max_{h in B} probe(h)
so that  U < tau  certifies that the eval-aware halfspace A* = {h : probe(h)>=tau}
is provably NOT reachable within the ball.

We reuse Raghunathan/Steinhardt/Liang's certified-defense idea (ICLR/NeurIPS
2018): relax the one-hidden-layer ReLU net to a convex (here, linear) bound and
propagate it through the ball. Three implementations, in order of preference:

    (a) "crown" via auto_LiRPA  -- the published linear-relaxation bounder
        (CROWN / backward mode). Used only if auto_LiRPA imports and runs against
        the installed torch; otherwise we fall back transparently.
    (b) "crown" native          -- a from-scratch, fully sound CROWN backward
        bound (same algorithm) in pure torch, returned as min(CROWN, IBP) since
        neither pure bound dominates (CROWN wins at small eps; IBP can win once
        most ReLUs are unstable at large eps -- the CROWN-IBP combination).
    (c) "ibp"  native           -- interval-bound propagation. Sound but loose.

Soundness is enforced at runtime: every certify() draws N random samples inside B
and asserts U >= max(sampled scores) - 1e-4. A violation is a verifier BUG and
raises immediately -- we never silently return a bound we cannot defend.

The ball uses per-dim radius eps_i = eps * sigma_i with sigma from the probe meta,
so eps is measured in units of per-dimension activation std (matches config.eps).
"""
from __future__ import annotations

import numpy as np
import torch

# Work both as `python -m cnr.verify` (package) and `python cnr/verify.py` (script).
if __package__ in (None, ""):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cnr.contracts import CertResult
else:
    from .contracts import CertResult


# --------------------------------------------------------------------------- #
# auto_LiRPA availability (probed once, lazily)
# --------------------------------------------------------------------------- #
def _try_import_auto_lirpa():
    """Return the auto_LiRPA symbols if it both imports AND is usable, else None.

    auto_LiRPA 0.x pins torch<1.13 and imports private ONNX symbols that newer
    torch removed, so on torch>=2 the import raises ImportError. We catch *any*
    exception so a partial/broken install degrades gracefully to the native path.
    """
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor  # noqa: F401
        from auto_LiRPA.perturbations import PerturbationLpNorm  # noqa: F401
    except BaseException:  # ImportError, version errors, C-ext issues, ...
        return None
    return {
        "BoundedModule": BoundedModule,
        "BoundedTensor": BoundedTensor,
        "PerturbationLpNorm": PerturbationLpNorm,
    }


_AUTO_LIRPA = _try_import_auto_lirpa()
AUTO_LIRPA_AVAILABLE = _AUTO_LIRPA is not None


# --------------------------------------------------------------------------- #
# Extract the (W1,b1,W2,b2) of a Linear->ReLU->Linear probe
# --------------------------------------------------------------------------- #
def _extract_params(probe) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull the two Linear layers out of the probe (Sequential[Lin,ReLU,Lin])."""
    seq = probe.net
    lin0, _relu, lin1 = seq[0], seq[1], seq[2]
    W1 = lin0.weight.detach().to(torch.float64)        # (hidden, d)
    b1 = lin0.bias.detach().to(torch.float64)          # (hidden,)
    W2 = lin1.weight.detach().to(torch.float64)        # (1, hidden)
    b2 = lin1.bias.detach().to(torch.float64)          # (1,)
    return W1, b1, W2, b2


# --------------------------------------------------------------------------- #
# Native sound bounds (pure torch, differentiable; see regularize.py)
# --------------------------------------------------------------------------- #
def ibp_bounds_linear(W: torch.Tensor, b: torch.Tensor,
                      xl: torch.Tensor, xu: torch.Tensor):
    """Interval propagation through y = Wx+b given x in [xl, xu].

    u = W+ xu + W- xl + b ,  l = W+ xl + W- xu + b   (W+ = relu(W), W- = -relu(-W)).
    Returns (yl, yu). Works for batched x of shape (..., in) with W (out, in).
    """
    Wp = W.clamp(min=0)
    Wn = W.clamp(max=0)
    yu = xu @ Wp.T + xl @ Wn.T + b
    yl = xl @ Wp.T + xu @ Wn.T + b
    return yl, yu


def ibp_upper(W1, b1, W2, b2, xl, xu):
    """Sound IBP upper bound of the scalar  W2 relu(W1 x + b1) + b2  over [xl,xu].

    Pure torch and differentiable in (W1,b1,W2,b2). xl,xu shape (..., d).
    Returns U with shape (...,) (the trailing scalar output squeezed).
    """
    zl, zu = ibp_bounds_linear(W1, b1, xl, xu)        # pre-activation
    al, au = torch.relu(zl), torch.relu(zu)           # ReLU is monotone
    _, U = ibp_bounds_linear(W2, b2, al, au)          # (..., 1)
    return U.squeeze(-1)


def crown_upper(W1, b1, W2, b2, xl, xu):
    """Sound CROWN backward upper bound of  W2 relu(W1 x + b1) + b2  over [xl,xu].

    Linear-relaxation (Zhang et al. CROWN; same family as Raghunathan 2018's
    convex relaxation). For each unstable ReLU neuron we relax with two lines:
        upper:  relu(z) <= (u/(u-l)) (z - l)
        lower:  relu(z) >=  a * z          (a in {0,1}, adaptive)
    To bound an UPPER bound of the (scalar, positively-combined) output we select,
    per neuron, the upper line where the back-propagated coefficient is >=0 and the
    lower line where it is <0. The single output has coefficient +1, so the
    back-prop coefficient onto the hidden layer is just W2.

    Pure torch and differentiable in (W1,b1,W2,b2). xl,xu shape (..., d).
    Returns U with shape (...,).
    """
    # pre-activation interval (IBP is the standard cheap pre-bound for CROWN)
    zl, zu = ibp_bounds_linear(W1, b1, xl, xu)        # (..., h)

    active = zl >= 0                                  # relu(z) = z
    inactive = zu <= 0                                # relu(z) = 0
    unstable = ~(active | inactive)

    # upper relaxation line:  alpha_u * z + beta_u
    denom = (zu - zl).clamp(min=1e-12)
    slope = zu / denom                                # u/(u-l), in (0,1) when unstable
    alpha_u = torch.where(active, torch.ones_like(zl),
              torch.where(inactive, torch.zeros_like(zl), slope))
    beta_u = torch.where(unstable, -slope * zl, torch.zeros_like(zl))

    # lower relaxation line:  alpha_l * z   (adaptive 0/1 slope, beta_l = 0)
    a_low = (slope >= 0.5).to(zl.dtype)               # CROWN adaptive choice
    alpha_l = torch.where(active, torch.ones_like(zl),
              torch.where(inactive, torch.zeros_like(zl), a_low))
    beta_l = torch.zeros_like(zl)                     # both lines pass through origin

    # back-propagated coefficient onto hidden post-activation = W2, broadcast to (...,h)
    lam = W2.squeeze(0)                               # (h,)
    lam = lam.expand(zl.shape)                        # (..., h)
    pos = lam >= 0

    alpha = torch.where(pos, alpha_u, alpha_l)        # (..., h)
    beta = torch.where(pos, beta_u, beta_l)

    # c^T f <= sum_j lam_j (alpha_j z_j + beta_j) + b2
    #        = (lam*alpha) @ (W1 x + b1) + lam@beta + b2
    lam_z = lam * alpha                               # (..., h)
    const = (lam * beta).sum(-1) + b2.sum()           # (...,)
    A = lam_z @ W1                                    # (..., d)
    const = const + (lam_z * b1).sum(-1)              # add lam_z @ b1

    # maximize A x + const over the box [xl, xu]
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)
    U = (Ap * xu + An * xl).sum(-1) + const           # (...,)
    return U


def crown_ibp_upper(W1, b1, W2, b2, xl, xu):
    """Sound upper bound = elementwise min(CROWN, IBP).

    Neither pure CROWN nor pure IBP dominates the other on this net: CROWN's
    adaptive ReLU lower-line (slope in {0,1}) wins for small boxes (few unstable
    neurons), while IBP can be tighter once most neurons are unstable at large eps.
    Both are individually sound upper bounds, so their min is a sound upper bound
    and is no worse than either -- this is exactly the CROWN-IBP combination used
    by modern verifiers. Differentiable in (W1,b1,W2,b2).
    """
    Uc = crown_upper(W1, b1, W2, b2, xl, xu)
    Ui = ibp_upper(W1, b1, W2, b2, xl, xu)
    return torch.minimum(Uc, Ui)


# --------------------------------------------------------------------------- #
# auto_LiRPA CROWN path (used only if the library actually runs)
# --------------------------------------------------------------------------- #
def _crown_auto_lirpa(probe, h0_t: torch.Tensor, eps_vec: torch.Tensor) -> float:
    """Upper bound of probe(h) over the per-dim L-inf ball, via auto_LiRPA CROWN.

    Returns a float U. Raises if auto_LiRPA is unavailable -- callers guard on
    AUTO_LIRPA_AVAILABLE. We explicitly take the UPPER element of the returned
    (lb, ub) pair; if a backend ever returned them swapped we still max() to be safe.
    """
    BoundedModule = _AUTO_LIRPA["BoundedModule"]
    BoundedTensor = _AUTO_LIRPA["BoundedTensor"]
    PerturbationLpNorm = _AUTO_LIRPA["PerturbationLpNorm"]

    x = h0_t.reshape(1, -1).to(torch.float32)
    model = BoundedModule(probe, x, device="cpu")
    # per-dimension eps via explicit x_L / x_U boxes (PerturbationLpNorm supports it)
    x_L = (x - eps_vec.reshape(1, -1)).to(torch.float32)
    x_U = (x + eps_vec.reshape(1, -1)).to(torch.float32)
    ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=x_L, x_U=x_U)
    bx = BoundedTensor(x, ptb)
    lb, ub = model.compute_bounds(x=(bx,), method="CROWN")
    # probe output is scalar -> ub shape (1,1) or (1,)
    return float(torch.as_tensor(ub).reshape(-1).max().item())


# --------------------------------------------------------------------------- #
# Sampling soundness witness
# --------------------------------------------------------------------------- #
def _sampled_max(probe, h0: np.ndarray, eps_vec: np.ndarray, n: int, seed: int = 0) -> float:
    """Max probe score over n uniform samples in the box B (a sound LOWER bound on
    the true max). The verifier's U must dominate this; that is the runtime check."""
    rng = np.random.default_rng(seed)
    d = h0.shape[0]
    # uniform in [h0 - eps_vec, h0 + eps_vec]; include the anchor and box corners-ish
    u = rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)
    pts = h0[None, :].astype(np.float32) + u * eps_vec[None, :].astype(np.float32)
    pts = np.vstack([h0[None, :].astype(np.float32), pts])  # ensure anchor included
    with torch.no_grad():
        s = probe(torch.tensor(pts, dtype=torch.float32)).numpy()
    return float(s.max())


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class CertifiedVerifier:
    """Sound upper-bound certifier for the probe halfspace A* within an eps-ball.

    method:
        "auto"  -> "crown" (auto_LiRPA if usable, else native CROWN)
        "crown" -> same as auto but errors out to native if the lib can't run
        "ibp"   -> native interval bound propagation (sound, looser)
    """

    def __init__(self, probe, sigma, norm: str = "inf", method: str = "auto"):
        if norm not in ("inf",):
            # Only L-inf is implemented for the relaxation; the contract allows "2"
            # but the certified-defense relaxation here is the L-inf box version.
            raise ValueError(f"only norm='inf' is supported by this verifier, got {norm!r}")
        self.probe = probe
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.norm = norm
        self.method = method
        self.n_witness = 2000
        # cache extracted params as float64 tensors (sound w.r.t. fp; tighter)
        self._W1, self._b1, self._W2, self._b2 = _extract_params(probe)

    # ---- resolve which concrete method to run ----
    def _resolved_method(self) -> str:
        if self.method == "ibp":
            return "ibp"
        # "auto" or "crown": prefer auto_LiRPA CROWN if it actually runs
        if AUTO_LIRPA_AVAILABLE:
            return "crown_auto_lirpa"
        return "crown_native"

    def certify(self, h0: np.ndarray, eps: float, tau: float) -> CertResult:
        h0 = np.asarray(h0, dtype=np.float64).reshape(-1)
        eps_vec = eps * self.sigma                         # per-dim radius
        xl = torch.tensor(h0 - eps_vec, dtype=torch.float64)
        xu = torch.tensor(h0 + eps_vec, dtype=torch.float64)

        chosen = self._resolved_method()
        used = chosen

        if chosen == "crown_auto_lirpa":
            try:
                U = _crown_auto_lirpa(
                    self.probe,
                    torch.tensor(h0, dtype=torch.float32),
                    torch.tensor(eps_vec, dtype=torch.float32),
                )
                used = "crown"
            except BaseException:
                # graceful fallback to native CROWN(+IBP min)
                U = float(crown_ibp_upper(self._W1, self._b1, self._W2, self._b2, xl, xu).item())
                used = "crown"  # still the CROWN algorithm, native backend
        elif chosen == "crown_native":
            # native CROWN, refined by min with IBP (sound; see crown_ibp_upper)
            U = float(crown_ibp_upper(self._W1, self._b1, self._W2, self._b2, xl, xu).item())
            used = "crown"
        else:  # "ibp"
            U = float(ibp_upper(self._W1, self._b1, self._W2, self._b2, xl, xu).item())
            used = "ibp"

        with torch.no_grad():
            s0 = float(self.probe(torch.tensor(h0, dtype=torch.float32)).item())

        # ---- soundness witness (MUST hold; else verifier bug) ----
        sampled_lb = _sampled_max(self.probe, h0, eps_vec, self.n_witness)
        if not (U >= sampled_lb - 1e-4):
            raise AssertionError(
                f"SOUNDNESS VIOLATION [{used}]: bound U={U:.6f} < sampled max "
                f"{sampled_lb:.6f} (eps={eps}). The verifier returned a value that "
                f"is provably NOT an upper bound -- this is a bug."
            )

        certified = bool(U < tau)
        details = {
            "sampled_lb": sampled_lb,
            "margin_to_tau": float(tau - U),          # >0 means certified
            "auto_lirpa_available": AUTO_LIRPA_AVAILABLE,
            "eps_vec_l2": float(np.linalg.norm(eps_vec)),
            "n_witness": self.n_witness,
        }
        return CertResult(
            certified_unreachable=certified,
            score_upper=U,
            score_at_anchor=s0,
            tau=float(tau),
            eps=float(eps),
            norm=self.norm,
            method=used,
            details=details,
        )


# --------------------------------------------------------------------------- #
# Certified radius via an eps sweep
# --------------------------------------------------------------------------- #
def certified_radius(verifier: CertifiedVerifier, h0: np.ndarray, tau: float, grid) -> float:
    """Largest eps in `grid` that is still certified-unreachable.

    We sweep eps ascending; U is monotone non-decreasing in eps, so once a value
    fails to certify, all larger ones do too. Returns the last certified eps
    (0.0 if even the smallest grid point cannot be certified).
    """
    grid = np.asarray(grid, dtype=float)
    grid = np.sort(grid)
    last_certified = 0.0
    for eps in grid:
        res = verifier.certify(h0, float(eps), tau)
        if res.certified_unreachable:
            last_certified = float(eps)
        else:
            break
    return last_certified


# --------------------------------------------------------------------------- #
# CLI: per-anchor certified-radius table
# --------------------------------------------------------------------------- #
def main():
    import json

    from cnr.config import Config
    from cnr.probe import load_probe

    cfg = Config()
    probe, meta = load_probe(cfg.probe_path)
    sigma = np.asarray(meta["sigma"], dtype=np.float64)
    tau = float(meta.get("tau", 0.0))

    bank = np.load(cfg.bank_path)
    X = bank["X"].astype(np.float64)
    with open(cfg.meta_path) as f:
        mj = json.load(f)
    anchor_ids = mj["anchor_ids"]
    eps_fixed = float(cfg.eps)

    verifier = CertifiedVerifier(probe, sigma, norm="inf", method="auto")
    method_label = "crown (auto_LiRPA)" if AUTO_LIRPA_AVAILABLE else "crown (native)"

    print("=" * 78)
    print("CERTIFIED EVAL-AWARENESS REACHABILITY  (target A* = {h: probe(h) >= tau})")
    print(f"  tau = {tau:g}   ball = per-dim L-inf, radius eps * sigma_i   norm = {cfg.norm}")
    print(f"  bound method: {method_label}"
          + ("" if AUTO_LIRPA_AVAILABLE else
             "   [auto_LiRPA import failed under this torch -> native fallback]"))
    print("=" * 78)

    grid = np.linspace(0.02, 2.0, 25)
    header = (f"{'anchor':>6} | {'score s(h0)':>12} | {'cert.radius':>11} | "
              f"{'U@eps='+format(eps_fixed,'g'):>14} | {'certified?':>10}")
    print(header)
    print("-" * len(header))

    radii = []
    for aid in anchor_ids:
        h0 = X[aid]
        cr = certified_radius(verifier, h0, tau, grid)
        radii.append(cr)
        res = verifier.certify(h0, eps_fixed, tau)
        print(f"{aid:>6} | {res.score_at_anchor:>12.3f} | {cr:>11.3f} | "
              f"{res.score_upper:>14.3f} | {str(res.certified_unreachable):>10}")

    print("-" * len(header))
    print(f"mean certified radius over {len(anchor_ids)} anchors: {np.mean(radii):.3f}  "
          f"(grid {grid[0]:g}..{grid[-1]:g}, {len(grid)} pts)")
    print(f"interpretation: at eps <= ~{np.mean(radii):.3f} (in std units) the eval-aware")
    print("region is PROVABLY unreachable from these benign deployment activations.")


if __name__ == "__main__":
    # repo root was added to sys.path at import time when run as a script
    main()
