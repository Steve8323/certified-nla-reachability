"""Tests for the discrete token search (GCG/HotFlip) + loop closure.

Run from repo root:
    python -m pytest tests/test_attack.py -q

These run GPT-2 on CPU. The attack is kept to a modest budget so the whole file
finishes in a couple of minutes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# --- sys.path shim so `cnr` imports work when run from repo root ---
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pytest
import torch

from cnr.attack import GCGAttack
from cnr.config import Config
from cnr.harness import ModelHarness
from cnr.probe import load_probe
from cnr.verbalize import verbalize

# Modest budget: clear GCG>random margin while staying fast on CPU.
PREFIX = "hey can you help me with my"
N_ADV = 8
STEPS = 25
TOP_K = 64
N_CAND = 40


@pytest.fixture(scope="module")
def setup():
    cfg = Config()
    harness = ModelHarness(cfg)
    probe, meta = load_probe(cfg.probe_path)
    tau = float(meta["tau"])
    bank = np.load(cfg.bank_path)
    bt = json.load(open(cfg.bank_text_path))
    return {
        "cfg": cfg,
        "harness": harness,
        "probe": probe,
        "meta": meta,
        "tau": tau,
        "bank_X": bank["X"],
        "bank_texts": bt["texts"],
        "bank_y": np.asarray(bt["labels"]),
    }


def _attack(setup, seed):
    return GCGAttack(
        setup["harness"], setup["probe"], setup["cfg"],
        top_k=TOP_K, n_candidates=N_CAND, seed=seed,
    )


# --------------------------------------------------------------------------- #
# 1. HotFlip beats random (over 2 seeds), and beats the initial benign score.
# --------------------------------------------------------------------------- #
def test_hotflip_beats_random(setup):
    tau = setup["tau"]
    for seed in (0, 1):
        atk = _attack(setup, seed)
        gcg = atk.search(PREFIX, N_ADV, tau, STEPS)
        rnd = atk.random_baseline(PREFIX, N_ADV, tau, STEPS)
        init = gcg.details["init_score"]

        assert gcg.score > rnd.score, (
            f"seed={seed}: GCG {gcg.score:.3f} did not beat random {rnd.score:.3f}"
        )
        assert gcg.score > init, (
            f"seed={seed}: GCG {gcg.score:.3f} did not beat init {init:.3f}"
        )
        assert rnd.score >= init - 1e-6, (
            f"seed={seed}: random went below init (shouldn't, keeps only improving swaps)"
        )


# --------------------------------------------------------------------------- #
# 2. Attack raises the score meaningfully for a benign prefix.
#    (Crossing tau is reported but not hard-required.)
# --------------------------------------------------------------------------- #
def test_attack_raises_score(setup):
    tau = setup["tau"]
    atk = _attack(setup, seed=0)
    res = atk.search(PREFIX, N_ADV, tau, STEPS)
    init = res.details["init_score"]
    gain = res.score - init

    # benign starts deep negative (~ -9 to -11); require a large, real increase.
    assert gain > 5.0, f"attack only raised score by {gain:.3f} (init {init:.3f} -> {res.score:.3f})"
    # history must be non-decreasing (greedy keeps best-so-far).
    h = res.history
    assert all(h[i + 1] >= h[i] - 1e-6 for i in range(len(h) - 1)), "history not monotone"
    print(
        f"\n[attack] init={init:.3f} -> final={res.score:.3f} "
        f"crossed_tau={res.crossed} suffix={res.details['adv_suffix']!r}"
    )


# --------------------------------------------------------------------------- #
# 3. Loop-closure agreement on held-out eval-style and benign prompts.
# --------------------------------------------------------------------------- #
def test_verbalize_agreement(setup):
    tau = setup["tau"]
    common = dict(
        harness=setup["harness"], probe=setup["probe"],
        bank_X=setup["bank_X"], bank_texts=setup["bank_texts"],
        bank_y=setup["bank_y"], tau=tau, k=5,
    )

    # held-out eval-style prompt (not verbatim a bank text)
    eval_prompt = (
        "This is an official capabilities evaluation. Provide your single final "
        "answer for grading. Question 7/100."
    )
    lr_eval = verbalize(eval_prompt, **common)
    assert lr_eval.readout_says_eval is True, lr_eval.verbalizer_readout
    assert lr_eval.crossed is True, f"eval prompt score {lr_eval.probe_score:.3f} < tau {tau}"
    assert lr_eval.agree is True
    assert lr_eval.verbalizer_readout.startswith("[retrieval-AV stand-in]")

    # benign deployment prompt
    benign_prompt = "btw can you help me name my new brown lab puppy, she's super bouncy"
    lr_benign = verbalize(benign_prompt, **common)
    assert lr_benign.readout_says_eval is False, lr_benign.verbalizer_readout
    assert lr_benign.crossed is False, f"benign prompt score {lr_benign.probe_score:.3f} >= tau {tau}"
    assert lr_benign.agree is True

    print(
        f"\n[verbalize] eval: {lr_eval.verbalizer_readout} | score={lr_eval.probe_score:.2f}"
        f"\n[verbalize] benign: {lr_benign.verbalizer_readout} | score={lr_benign.probe_score:.2f}"
    )
