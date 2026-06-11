"""Tests for the HARD eval-awareness task (cnr.data_hard + the certified pipeline).

Run from repo root:
    HF_HUB_OFFLINE=1 python -m pytest tests/test_hard.py -q

These check the three properties that make the hard task honest:
  1. the dataset genuinely OVERLAPS -- both hard pools are present AND the
     ambiguous pool emits the SAME text under both labels (non-zero Bayes error);
  2. a ReLU probe on real GPT-2 layer-6 activations lands in (0.6, 0.99) AUROC,
     i.e. better than chance but NOT saturated;
  3. cnr.verify.certified_radius runs end-to-end without error and returns a
     finite, non-negative radius.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

# --- repo root on sys.path so `cnr` imports under pytest or as a file ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cnr.config import Config  # noqa: E402
from cnr.data_hard import build_hard_dataset, _AMBIGUOUS  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Dataset overlap (no model needed -- fast, pure)
# --------------------------------------------------------------------------- #
def test_dataset_has_hard_pools_and_balance():
    texts, labels, info = build_hard_dataset(256, seed=0, difficulty=0.55)
    assert len(texts) == len(labels) == info["n_total"]
    # both hard pools actually present
    assert info["eval_hard"] > 0, "no hard positives emitted"
    assert info["deploy_hard"] > 0, "no hard negatives emitted"
    # roughly balanced (within 65/35) so AUROC is meaningful
    frac1 = info["n_label1"] / info["n_total"]
    assert 0.35 < frac1 < 0.65, f"class balance off: label1 frac={frac1:.2f}"


def test_ambiguous_overlap_same_text_both_labels():
    """The same ambiguous surface text must appear under BOTH labels -- that is
    the genuine class overlap (non-zero Bayes error), not a wording difference."""
    texts, labels, info = build_hard_dataset(512, seed=0, difficulty=0.7)
    assert info["n_ambiguous"] > 0
    assert info["amb_flipped"] > 0, "no label noise -> no irreducible overlap"

    # An ambiguous stem (text before any appended body / formatting) that occurs
    # with both label 0 and label 1 proves the overlap concretely.
    stems = {a.split("?")[0] for a in _AMBIGUOUS}
    by_stem: dict[str, set[int]] = {}
    for t, y in zip(texts, labels):
        head = t.split("?")[0]
        if head in stems:
            by_stem.setdefault(head, set()).add(int(y))
    both = [s for s, ys in by_stem.items() if ys == {0, 1}]
    assert both, "no ambiguous stem appeared under both labels"


def test_difficulty_monotone_overlap():
    """Higher difficulty -> more ambiguous (overlapping) examples and more noise."""
    _, _, lo = build_hard_dataset(256, seed=1, difficulty=0.3)
    _, _, hi = build_hard_dataset(256, seed=1, difficulty=0.8)
    assert hi["n_ambiguous"] > lo["n_ambiguous"]
    assert hi["label_noise"] >= lo["label_noise"]


# --------------------------------------------------------------------------- #
# Shared model fixture (real GPT-2; cached offline). Module-scoped: load once.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def harness():
    from cnr.harness import ModelHarness
    return ModelHarness(Config())


@pytest.fixture(scope="module")
def trained(harness):
    """Train a probe on a modest hard dataset; return (probe, meta, X, y, test_idx)."""
    import torch  # noqa: F401
    from cnr.probe import train_probe

    cfg = Config()
    texts, labels, info = build_hard_dataset(220, seed=0, difficulty=0.55)
    X = harness.activations(texts).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    ntr = int(round(0.7 * len(y)))
    tr, te = perm[:ntr], perm[ntr:]
    probe, meta = train_probe(
        X[tr], y[tr], hidden=cfg.hidden, epochs=cfg.probe_epochs,
        lr=cfg.probe_lr, seed=0,
    )
    return probe, meta, X, y, te


# --------------------------------------------------------------------------- #
# 2. Probe is non-saturated on real activations
# --------------------------------------------------------------------------- #
def test_probe_auroc_not_saturated(trained):
    import torch
    from sklearn.metrics import roc_auc_score

    probe, meta, X, y, te = trained
    with torch.no_grad():
        s_te = probe(torch.tensor(X[te], dtype=torch.float32)).numpy()
    auc = roc_auc_score(y[te], s_te)
    assert 0.6 < auc < 0.99, (
        f"AUROC={auc:.3f} outside (0.6, 0.99): "
        f"{'too easy/saturated' if auc >= 0.99 else 'worse than expected'}"
    )


# --------------------------------------------------------------------------- #
# 3. certified_radius runs end-to-end without error
# --------------------------------------------------------------------------- #
def test_certified_radius_runs(trained):
    from cnr.verify import CertifiedVerifier, certified_radius

    probe, meta, X, y, te = trained
    sigma = np.asarray(meta["sigma"], dtype=np.float64)
    tau = float(meta["tau"])
    verifier = CertifiedVerifier(probe, sigma, norm="inf", method="auto")

    deploy_ids = np.where(y == 0)[0][:3]
    grid = np.linspace(0.02, 1.0, 8)
    for aid in deploy_ids:
        cr = certified_radius(verifier, X[aid].astype(np.float64), tau, grid)
        assert np.isfinite(cr) and cr >= 0.0, f"bad certified radius {cr} at anchor {aid}"
        # a sanity certify() call also returns a sound, witnessed bound
        res = verifier.certify(X[aid].astype(np.float64), 0.1, tau)
        assert res.score_upper >= res.details["sampled_lb"] - 1e-4
