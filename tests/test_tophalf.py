"""Top-half tests: classifier-on-NLA recovers eval-awareness, surrogate is a
faithful certifiable shadow, the certificate is sound and bites, and the
handoff artifacts load. build() is run once and shared.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnr.config import Config
from cnr.nla_target import build
from cnr.probe import load_probe


@pytest.fixture(scope="module")
def run():
    return build(Config())


def test_classifier_on_nla_recovers_eval_awareness(run):
    cl, nc = run["classifier"], run["nla_classify"]
    # trained on held-out phrasings, then applied to (differently phrased) NLA decodes
    assert cl["val_acc"] >= 0.9
    assert nc["acc_vs_truth"] >= 0.8
    assert nc["auroc"] >= 0.85
    assert nc["mean_proba_eval"] > nc["mean_proba_deploy"]


def test_surrogate_is_faithful_shadow(run):
    sg = run["surrogate"]
    assert sg["fidelity_corr_val"] >= 0.7
    assert sg["sign_agree_val"] >= 0.85
    assert sg["g_vs_clf_agree"] >= 0.85


def test_certificate_sound_and_bites(run):
    ct = run["certificate"]
    # at least one benign anchor is provably unable to reach the eval-aware region
    assert any(a["certified_unreachable"] for a in ct["anchors"])
    # every certified anchor has U strictly below tau=0 (the definition of the guarantee)
    for a in ct["anchors"]:
        if a["certified_unreachable"]:
            assert a["U"] < 0.0


def test_soundness_invariant_in_ascent(run):
    """Inside the certified radius the surrogate provably cannot cross tau, so a
    budget-projected ascent there must NOT cross. The sweep must also exercise a
    crossing at some larger budget (else the loop-closure diagnostic is vacuous)."""
    asc = run["ascent"]
    rcert = asc["cert_radius"]
    for s in asc["sweep"]:
        if s["budget_std"] <= rcert + 1e-9:
            assert not s["g_crossed"], "ascent crossed inside certified radius — unsound!"
    assert any(s["g_crossed"] for s in asc["sweep"])


def test_handoff_artifacts(run):
    cfg = Config()
    surr = cfg.artifacts / "nla_surrogate.pt"
    clf = cfg.artifacts / "nla_text_clf.joblib"
    ho = cfg.artifacts / "handoff.json"
    assert surr.exists() and clf.exists() and ho.exists()

    # the surrogate is a drop-in ReLU probe the inverter can score against
    g, meta = load_probe(surr)
    bank = np.load(cfg.bank_path)
    X = bank["X"].astype(np.float32)
    with torch.no_grad():
        s = g(torch.tensor(X)).numpy()
    assert s.shape == (X.shape[0],)
    assert "sigma" in meta and "tau" in meta

    h = json.loads(ho.read_text())
    assert "tau" in h and "target_activations" in h and len(h["target_activations"]) > 0
