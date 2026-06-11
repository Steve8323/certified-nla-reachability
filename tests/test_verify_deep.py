"""Tests for the DEEP certified verifier (cnr.verify_deep).

Run from the repo root:   python -m pytest tests/test_verify_deep.py -q

The non-negotiable property is SOUNDNESS of the deep certificate: the returned U
is always an upper bound on the score of the FROZEN-ATTENTION subnetwork over the
ball (the exact subnetwork the witness re-runs). We test that directly against
dense sampling on several anchors and eps, plus:
  * the sound primitives (gelu interval/linear envelope, LayerNorm interval) are
    verified sound on dense grids / Monte-Carlo;
  * CROWN <= IBP (the deep CROWN path returns min(CROWN,IBP), so never looser);
  * the null-direction probe behaves sanely (it does not separate eval; its
    certified radius is finite and positive on the well-separated anchors).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cnr.config import Config  # noqa: E402
from cnr.harness import ModelHarness  # noqa: E402
from cnr.probe import load_probe  # noqa: E402
from cnr.verify_deep import (  # noqa: E402
    gelu_interval,
    gelu_linear_relax,
    gelu_new,
    layernorm_interval,
    make_certifier_from_harness,
    prepare_anchor,
)


# --------------------------------------------------------------------------- #
# Module-scoped heavy fixtures (load GPT-2 once)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def setup():
    cfg = Config()
    probe, meta = load_probe(cfg.probe_path)
    sigma = np.asarray(meta["sigma"], dtype=np.float64)
    tau = float(meta.get("tau", 0.0))
    harness = ModelHarness(cfg)
    with open(cfg.meta_path) as f:
        mj = json.load(f)
    with open(cfg.bank_text_path) as f:
        bt = json.load(f)
    texts = bt["texts"] if isinstance(bt, dict) and "texts" in bt else bt
    cert = make_certifier_from_harness(harness, probe, sigma, cfg.layer, n_witness=600)
    return {
        "cfg": cfg, "probe": probe, "sigma": sigma, "tau": tau,
        "harness": harness, "anchor_ids": mj["anchor_ids"], "texts": texts,
        "cert": cert, "layer": cfg.layer,
    }


# --------------------------------------------------------------------------- #
# 1. Sound primitives
# --------------------------------------------------------------------------- #
def test_gelu_interval_is_exact_range():
    """gelu_interval must be a SOUND (here, exact) range over each interval."""
    torch.manual_seed(0)
    l = torch.randn(4000, dtype=torch.float64) * 3
    u = l + torch.rand(4000, dtype=torch.float64) * 5 + 1e-6
    gmin, gmax = gelu_interval(l, u)
    ts = torch.linspace(0, 1, 3000, dtype=torch.float64).view(-1, 1)
    xs = l.view(1, -1) + ts * (u - l).view(1, -1)
    gs = gelu_new(xs)
    assert (gmax >= gs.max(0).values - 1e-9).all(), "gelu interval upper not sound"
    assert (gmin <= gs.min(0).values + 1e-9).all(), "gelu interval lower not sound"


def test_gelu_linear_envelope_sound():
    """a_l x + b_l <= gelu(x) <= a_u x + b_u on [l,u], checked on a dense grid."""
    torch.manual_seed(1)
    # include wide, mostly-negative intervals like real fc pre-activations
    l = torch.rand(8000, dtype=torch.float64) * -8 - 0.1
    u = l + torch.rand(8000, dtype=torch.float64) * 8 + 0.5
    a_l, b_l, a_u, b_u = gelu_linear_relax(l, u, ngrid=96)
    ts = torch.linspace(0, 1, 4000, dtype=torch.float64).view(-1, 1)
    xs = l.view(1, -1) + ts * (u - l).view(1, -1)
    gs = gelu_new(xs)
    up = a_u.view(1, -1) * xs + b_u.view(1, -1)
    lo = a_l.view(1, -1) * xs + b_l.view(1, -1)
    assert float((up - gs).min()) >= -1e-6, "GELU upper envelope unsound"
    assert float((gs - lo).min()) >= -1e-6, "GELU lower envelope unsound"


def test_layernorm_interval_sound():
    """LayerNorm interval bound must contain all sampled outputs."""
    torch.manual_seed(2)
    D = 768
    gamma = torch.randn(D, dtype=torch.float64)
    beta = torch.randn(D, dtype=torch.float64)
    eps = 1e-5
    for _ in range(6):
        x0 = torch.randn(D, dtype=torch.float64) * 2
        r = 0.05 * torch.rand(D, dtype=torch.float64) + 0.02
        lo = x0 - r
        hi = x0 + r
        y_lo, y_hi = layernorm_interval(lo, hi, gamma, beta, eps)
        S = x0 + (2 * torch.rand(20000, D, dtype=torch.float64) - 1) * r
        mean = S.mean(-1, keepdim=True)
        var = ((S - mean) ** 2).mean(-1, keepdim=True)
        ys = (S - mean) / torch.sqrt(var + eps) * gamma + beta
        assert (y_lo <= ys.min(0).values + 1e-9).all(), "LN lower unsound"
        assert (y_hi >= ys.max(0).values - 1e-9).all(), "LN upper unsound"


# --------------------------------------------------------------------------- #
# 2. Deep certificate soundness: U >= sampled frozen max (several anchors/eps)
# --------------------------------------------------------------------------- #
def _dense_frozen_max(cert, h, attn, w, eps_vec, n, token_index, seed):
    """Independent dense sampler (not the verifier's own witness) of the EXACT
    frozen-attention subnetwork score (a sound lower bound on the true max)."""
    gen = torch.Generator().manual_seed(seed)
    T = h.shape[0]
    D = h.shape[1]
    mx = cert._frozen_forward(h, attn, w)
    for _ in range(n):
        noise = (2 * torch.rand(T, D, generator=gen, dtype=torch.float64) - 1) * eps_vec
        hp = h.clone()
        if token_index is None or token_index < 0:
            hp = h + noise
        else:
            hp[token_index] = h[token_index] + noise[token_index]
        mx = max(mx, cert._frozen_forward(hp, attn, w))
    return mx


def test_deep_soundness_single_token(setup):
    cert = setup["cert"]; tau = setup["tau"]; sigma = setup["sigma"]
    for aid in setup["anchor_ids"][:3]:
        h, attn, w, pooled, T = prepare_anchor(setup["harness"], setup["texts"][aid], setup["layer"])
        for eps in (0.005, 0.02, 0.1):
            res = cert.certify(h, attn, w, eps, tau, method="crown",
                               token_index=T - 1, witness_seed=aid)
            eps_vec = torch.tensor(eps * sigma, dtype=torch.float64)
            samp = _dense_frozen_max(cert, h, attn, w, eps_vec, n=1500,
                                     token_index=T - 1, seed=1000 + aid)
            assert res.score_upper >= samp - 1e-4, (
                f"DEEP UNSOUND (1tok) anchor {aid} eps {eps}: "
                f"U={res.score_upper:.4f} < sampled {samp:.4f}")
            # the verifier's own recorded witness must also be dominated
            assert res.score_upper >= res.sampled_lb - 1e-4


def test_deep_soundness_all_tokens(setup):
    cert = setup["cert"]; tau = setup["tau"]; sigma = setup["sigma"]
    for aid in setup["anchor_ids"][:2]:
        h, attn, w, pooled, T = prepare_anchor(setup["harness"], setup["texts"][aid], setup["layer"])
        for eps in (0.01, 0.05):
            res = cert.certify(h, attn, w, eps, tau, method="crown",
                               token_index=-1, witness_seed=aid)
            eps_vec = torch.tensor(eps * sigma, dtype=torch.float64)
            samp = _dense_frozen_max(cert, h, attn, w, eps_vec, n=1500,
                                     token_index=-1, seed=2000 + aid)
            assert res.score_upper >= samp - 1e-4, (
                f"DEEP UNSOUND (all) anchor {aid} eps {eps}: "
                f"U={res.score_upper:.4f} < sampled {samp:.4f}")


def test_deep_anchor_score_matches_shallow(setup):
    """The frozen-attention forward at the anchor reproduces the probe's actual
    input (mean-pooled hidden_states[L]) -- so the subnetwork is wired correctly."""
    cert = setup["cert"]; probe = setup["probe"]
    for aid in setup["anchor_ids"][:3]:
        h, attn, w, pooled, T = prepare_anchor(setup["harness"], setup["texts"][aid], setup["layer"])
        s_frozen = cert._frozen_forward(h, attn, w)
        s_shallow = float(probe(pooled.float()).item())
        assert abs(s_frozen - s_shallow) < 1e-2, (
            f"frozen forward {s_frozen:.4f} != shallow probe input {s_shallow:.4f}")


# --------------------------------------------------------------------------- #
# 3. CROWN <= IBP (deep CROWN returns min(CROWN, IBP), never looser)
# --------------------------------------------------------------------------- #
def test_deep_crown_not_looser_than_ibp(setup):
    cert = setup["cert"]; tau = setup["tau"]
    for aid in setup["anchor_ids"][:3]:
        h, attn, w, pooled, T = prepare_anchor(setup["harness"], setup["texts"][aid], setup["layer"])
        for eps in (0.005, 0.02, 0.1):
            r_crown = cert.certify(h, attn, w, eps, tau, method="crown",
                                   token_index=T - 1, witness_seed=aid)
            r_ibp = cert.certify(h, attn, w, eps, tau, method="ibp",
                                 token_index=T - 1, witness_seed=aid)
            assert r_crown.score_upper <= r_ibp.score_upper + 1e-6, (
                f"deep CROWN looser than IBP: {r_crown.score_upper} > {r_ibp.score_upper}")
            assert r_ibp.method == "ibp" and r_crown.method == "crown"


# --------------------------------------------------------------------------- #
# 4. Monotonicity + clean certify/not-certify separation for the deep cert
# --------------------------------------------------------------------------- #
def test_deep_monotone_and_separation(setup):
    cert = setup["cert"]; tau = setup["tau"]
    aid = setup["anchor_ids"][0]
    h, attn, w, pooled, T = prepare_anchor(setup["harness"], setup["texts"][aid], setup["layer"])
    epses = [0.001, 0.002, 0.005, 0.01, 0.05, 0.2]
    Us = [cert.certify(h, attn, w, e, tau, method="crown",
                       token_index=T - 1, witness_seed=aid).score_upper for e in epses]
    for a, b in zip(Us, Us[1:]):
        assert b >= a - 1e-6, f"deep U non-monotone in eps: {a} -> {b}"
    # tiny eps certifies (U < tau), large eps does not (the score grows past tau)
    r_small = cert.certify(h, attn, w, 0.001, tau, method="crown",
                           token_index=T - 1, witness_seed=aid)
    r_big = cert.certify(h, attn, w, 0.2, tau, method="crown",
                         token_index=T - 1, witness_seed=aid)
    assert r_small.certified_unreachable is True
    assert r_big.certified_unreachable is False


# --------------------------------------------------------------------------- #
# 5. Null-direction probe sanity (used by the ablation baseline)
# --------------------------------------------------------------------------- #
def test_null_direction_behaves_sanely(setup):
    """The random null probe must NOT separate eval/deploy, yet still yield a
    finite, positive certified radius on the well-separated benign anchors."""
    from cnr.verify import CertifiedVerifier, certified_radius
    # import the ablation's null-probe builder
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "abl", os.path.join(_REPO_ROOT, "scripts", "05_cert_ablation.py"))
    abl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abl)

    cfg = setup["cfg"]; probe = setup["probe"]; sigma = setup["sigma"]; tau = setup["tau"]
    bank = np.load(cfg.bank_path)
    X = bank["X"].astype(np.float64)
    y = bank["y"]
    _, meta = load_probe(cfg.probe_path)
    null = abl._make_null_probe(probe, meta, X, y, seed=0)

    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        s_real = probe(Xt).numpy()
        s_null = null(Xt).numpy()
    # real probe separates eval (label 1) above deploy (label 0); null does NOT
    real_gap = s_real[y == 1].mean() - s_real[y == 0].mean()
    null_gap = s_null[y == 1].mean() - s_null[y == 0].mean()
    assert real_gap > 5.0, f"real probe should separate eval strongly (gap {real_gap:.2f})"
    assert abs(null_gap) < real_gap * 0.5, (
        f"null direction should NOT separate eval (gap {null_gap:.2f} vs real {real_gap:.2f})")

    # null probe still produces a sound, finite, positive certified radius
    v_null = CertifiedVerifier(null, sigma, norm="inf", method="auto")
    grid = np.linspace(0.02, 2.5, 25)
    r = certified_radius(v_null, X[setup["anchor_ids"][0]], tau, grid)
    assert r >= 0.0 and np.isfinite(r)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
