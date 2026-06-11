"""Tests for the certified verifier (cnr.verify).

Run from repo root:  python -m pytest tests/test_verify.py -q

The non-negotiable property is SOUNDNESS: the returned U is always an upper bound
on max_{B} probe(h). We test that directly against dense sampling and (on a tiny
hand-built net) against a brute-force grid, plus monotonicity and a clean
certified / not-certified separation.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

# --- add repo root to sys.path so `cnr` imports when run as a file or via pytest ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cnr.config import Config  # noqa: E402
from cnr.probe import ReLUProbe, load_probe  # noqa: E402
from cnr.verify import (  # noqa: E402
    AUTO_LIRPA_AVAILABLE,
    CertifiedVerifier,
    certified_radius,
    crown_upper,
    ibp_upper,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_probe():
    cfg = Config()
    probe, meta = load_probe(cfg.probe_path)
    sigma = np.asarray(meta["sigma"], dtype=np.float64)
    bank = np.load(cfg.bank_path)
    X = bank["X"].astype(np.float64)
    import json
    with open(cfg.meta_path) as f:
        mj = json.load(f)
    return probe, sigma, X, mj["anchor_ids"], float(meta.get("tau", 0.0))


def _dense_sampled_max(probe, h0, eps_vec, n, seed=0):
    """Max probe score over n uniform samples in the per-dim box (sound lower bound)."""
    rng = np.random.default_rng(seed)
    d = h0.shape[0]
    u = rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)
    pts = h0[None, :].astype(np.float32) + u * eps_vec[None, :].astype(np.float32)
    with torch.no_grad():
        s = probe(torch.tensor(pts, dtype=torch.float32)).numpy()
    return float(s.max())


# --------------------------------------------------------------------------- #
# 1. Soundness on the real probe: U >= sampled max (5000 samples)
# --------------------------------------------------------------------------- #
def test_soundness_real_probe(real_probe):
    probe, sigma, X, anchor_ids, tau = real_probe
    verifier = CertifiedVerifier(probe, sigma, norm="inf", method="auto")
    for aid in anchor_ids[:4]:
        h0 = X[aid]
        for eps in (0.05, 0.25, 0.6, 1.2):
            res = verifier.certify(h0, eps, tau)
            eps_vec = eps * sigma
            samp = _dense_sampled_max(probe, h0, eps_vec, n=5000, seed=aid)
            assert res.score_upper >= samp - 1e-4, (
                f"UNSOUND: anchor {aid} eps {eps}: U={res.score_upper:.4f} "
                f"< sampled {samp:.4f}")
            # the verifier's own internal witness must also be recorded and respected
            assert res.score_upper >= res.details["sampled_lb"] - 1e-4


def test_soundness_real_probe_ibp(real_probe):
    """IBP is also sound (and, being looser, must dominate CROWN)."""
    probe, sigma, X, anchor_ids, tau = real_probe
    v_ibp = CertifiedVerifier(probe, sigma, norm="inf", method="ibp")
    v_crown = CertifiedVerifier(probe, sigma, norm="inf", method="auto")
    for aid in anchor_ids[:4]:
        h0 = X[aid]
        for eps in (0.1, 0.5, 1.0):
            r_ibp = v_ibp.certify(h0, eps, tau)
            r_crown = v_crown.certify(h0, eps, tau)
            eps_vec = eps * sigma
            samp = _dense_sampled_max(probe, h0, eps_vec, n=5000, seed=100 + aid)
            assert r_ibp.score_upper >= samp - 1e-4
            assert r_crown.score_upper >= samp - 1e-4
            assert r_ibp.method == "ibp"
            # the "crown" method is the sound min(CROWN, IBP), so it is never
            # looser than plain IBP (neither pure bound dominates the other --
            # CROWN wins at small eps, IBP can win at large eps; min gets both).
            assert r_crown.score_upper <= r_ibp.score_upper + 1e-6


# --------------------------------------------------------------------------- #
# 2. Monotonicity: U non-decreasing in eps
# --------------------------------------------------------------------------- #
def test_monotonicity(real_probe):
    probe, sigma, X, anchor_ids, tau = real_probe
    for method in ("auto", "ibp"):
        verifier = CertifiedVerifier(probe, sigma, norm="inf", method=method)
        for aid in anchor_ids[:3]:
            h0 = X[aid]
            epses = np.linspace(0.02, 2.0, 20)
            Us = [verifier.certify(h0, float(e), tau).score_upper for e in epses]
            for a, b in zip(Us, Us[1:]):
                assert b >= a - 1e-6, f"non-monotone U: {a:.5f} -> {b:.5f} ({method})"


# --------------------------------------------------------------------------- #
# 3. Tiny hand-built net: brute-force the true max, assert soundness + tightness
# --------------------------------------------------------------------------- #
def _build_toy(W1, b1, W2, b2):
    """A ReLUProbe(d=2, hidden=2) with explicit weights (bypasses standardization)."""
    probe = ReLUProbe(2, 2)
    with torch.no_grad():
        probe.net[0].weight.copy_(torch.tensor(W1, dtype=torch.float32))
        probe.net[0].bias.copy_(torch.tensor(b1, dtype=torch.float32))
        probe.net[2].weight.copy_(torch.tensor(W2, dtype=torch.float32))
        probe.net[2].bias.copy_(torch.tensor(b2, dtype=torch.float32))
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


def _brute_force_max(probe, xl, xu, n=200):
    xs = torch.linspace(xl[0], xu[0], n)
    ys = torch.linspace(xl[1], xu[1], n)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    with torch.no_grad():
        s = probe(grid)
    return float(s.max())


def test_toy_net_soundness_and_tightness():
    W1 = [[1.0, -2.0], [0.5, 1.5]]
    b1 = [0.2, -0.3]
    W2 = [[1.3, -1.1]]
    b2 = [0.05]
    probe = _build_toy(W1, b1, W2, b2)

    # sigma=1 so the box is plain [h0-eps, h0+eps]
    sigma = np.ones(2)
    h0 = np.array([0.3, -0.2])
    eps = 0.8
    xl = h0 - eps * sigma
    xu = h0 + eps * sigma

    true_max = _brute_force_max(probe, xl, xu, n=200)

    # both native bounds
    xl_t = torch.tensor(xl, dtype=torch.float64)
    xu_t = torch.tensor(xu, dtype=torch.float64)
    W1t = torch.tensor(W1, dtype=torch.float64)
    b1t = torch.tensor(b1, dtype=torch.float64)
    W2t = torch.tensor(W2, dtype=torch.float64)
    b2t = torch.tensor(b2, dtype=torch.float64)
    U_crown = float(crown_upper(W1t, b1t, W2t, b2t, xl_t, xu_t).item())
    U_ibp = float(ibp_upper(W1t, b1t, W2t, b2t, xl_t, xu_t).item())

    # soundness: both bounds >= true max
    assert U_crown >= true_max - 1e-4, f"CROWN unsound {U_crown} < {true_max}"
    assert U_ibp >= true_max - 1e-4, f"IBP unsound {U_ibp} < {true_max}"

    # tightness: on this toy box CROWN should be reasonably close, <= IBP
    assert U_crown <= U_ibp + 1e-6
    assert (U_crown - true_max) < 1.0, (
        f"CROWN unexpectedly loose: U={U_crown:.4f} true={true_max:.4f}")

    # and the full verifier path agrees (its sampling witness must hold)
    v = CertifiedVerifier(probe, sigma, norm="inf", method="auto")
    res = v.certify(h0, eps, tau=0.0)
    assert res.score_upper >= true_max - 1e-4
    assert abs(res.score_upper - U_crown) < 1e-3


def test_toy_net_all_active_is_exact():
    """If the box keeps both ReLUs strictly active, CROWN must be EXACT (affine)."""
    # weights/box chosen so W1 x + b1 > 0 over the whole box -> net is affine there
    W1 = [[1.0, 0.0], [0.0, 1.0]]
    b1 = [5.0, 5.0]
    W2 = [[1.0, 1.0]]
    b2 = [0.0]
    probe = _build_toy(W1, b1, W2, b2)
    sigma = np.ones(2)
    h0 = np.array([0.0, 0.0])
    eps = 1.0  # box [-1,1]^2, preacts in [4,6] -> always active
    xl = h0 - eps * sigma
    xu = h0 + eps * sigma
    true_max = _brute_force_max(probe, xl, xu, n=200)
    W1t = torch.tensor(W1, dtype=torch.float64); b1t = torch.tensor(b1, dtype=torch.float64)
    W2t = torch.tensor(W2, dtype=torch.float64); b2t = torch.tensor(b2, dtype=torch.float64)
    U_crown = float(crown_upper(W1t, b1t, W2t, b2t,
                                torch.tensor(xl), torch.tensor(xu)).item())
    # affine over the box: bound = true continuous max (grid is within fp/discretization)
    assert U_crown >= true_max - 1e-4
    assert abs(U_crown - true_max) < 1e-2


# --------------------------------------------------------------------------- #
# 4. Clear separation: small eps certifies unreachable, large eps does not
# --------------------------------------------------------------------------- #
def test_certified_vs_not_separation():
    # well-separated 1-D-ish probe: score(x) ~ w*x0, threshold at 0.
    # anchor sits far on the negative side; small ball stays negative, big ball crosses.
    W1 = [[1.0, 0.0], [-1.0, 0.0]]   # relu(x0) and relu(-x0)
    b1 = [0.0, 0.0]
    W2 = [[1.0, -1.0]]               # = relu(x0) - relu(-x0) = x0  (identity on x0)
    b2 = [0.0]
    probe = _build_toy(W1, b1, W2, b2)
    sigma = np.ones(2)
    h0 = np.array([-5.0, 0.0])       # score = -5, far below tau=0
    tau = 0.0
    v = CertifiedVerifier(probe, sigma, norm="inf", method="auto")

    res_small = v.certify(h0, eps=1.0, tau=tau)    # ball x0 in [-6,-4] -> max -4 < 0
    assert res_small.certified_unreachable is True
    assert res_small.score_upper < tau

    res_big = v.certify(h0, eps=6.0, tau=tau)       # ball x0 in [-11,1] -> max +1 >= 0
    assert res_big.certified_unreachable is False
    assert res_big.score_upper >= tau

    # certified_radius lands between the two and is itself certified
    grid = np.linspace(0.5, 8.0, 40)
    r = certified_radius(v, h0, tau, grid)
    assert 1.0 <= r < 6.0
    assert v.certify(h0, r, tau).certified_unreachable is True


def test_certified_radius_real_probe(real_probe):
    """On the real probe the anchors certify a positive radius (gap to cross ~10)."""
    probe, sigma, X, anchor_ids, tau = real_probe
    v = CertifiedVerifier(probe, sigma, norm="inf", method="auto")
    grid = np.linspace(0.02, 2.0, 25)
    for aid in anchor_ids[:3]:
        r = certified_radius(v, X[aid], tau, grid)
        assert r > 0.0, f"anchor {aid} should certify a positive radius"
        # and that radius is genuinely certified
        assert v.certify(X[aid], r, tau).certified_unreachable is True


def test_reports_method_label(real_probe):
    """CertResult.method reflects what actually ran; auto resolves to crown here."""
    probe, sigma, X, anchor_ids, tau = real_probe
    v = CertifiedVerifier(probe, sigma, norm="inf", method="auto")
    res = v.certify(X[anchor_ids[0]], 0.25, tau)
    assert res.method == "crown"
    assert res.details["auto_lirpa_available"] == AUTO_LIRPA_AVAILABLE


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
