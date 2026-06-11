"""Shared interfaces + result types that all modules build against.

Agents implement classes satisfying these Protocols. Keeping the data shapes
fixed here is what lets the verifier and the attack search be developed in
parallel and then wired together by run_pipeline.py.

CERTIFICATION TARGET (read this carefully -- signs matter):
    probe score is s(h) = Probe(h)  (a scalar logit; higher => more "eval-aware")
    target set     A*   = { h : s(h) >= tau }
    ball           B    = { h : ||h - h0||_p <= eps }   (p = inf or 2)

    "Is A* reachable from h0 within eps?"  <=>  is  max_{h in B} s(h) >= tau ?

A sound verifier returns an UPPER bound U >= max_{h in B} s(h).
    U <  tau  =>  certified_unreachable = True   (a guarantee: no perturbation
                  within eps can make the model cross into the eval-aware region)
    U >= tau  =>  cannot certify safety; may or may not be truly reachable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


# --------------------------------------------------------------------------- #
# Verifier (Agent A)
# --------------------------------------------------------------------------- #
@dataclass
class CertResult:
    certified_unreachable: bool   # True => A* provably NOT reachable within eps
    score_upper: float            # sound upper bound U on max_{B} s(h)
    score_at_anchor: float        # s(h0)
    tau: float
    eps: float
    norm: str
    method: str                   # "crown" | "ibp" | "exact-linear" | ...
    anchor_id: int = -1
    details: dict = field(default_factory=dict)


@runtime_checkable
class CertifiedVerifier(Protocol):
    """Certifies reachability of the probe halfspace A* within an eps-ball."""

    def certify(self, h0: np.ndarray, eps: float, tau: float) -> CertResult:
        """h0: (d,) anchor activation. Returns a sound CertResult."""
        ...


# --------------------------------------------------------------------------- #
# Token search / inversion (Agent B)
# --------------------------------------------------------------------------- #
@dataclass
class AttackResult:
    prompt: str
    token_ids: list[int]
    score: float                  # probe score s(h) of the found prompt
    crossed: bool                 # s(h) >= tau
    steps: int
    history: list[float] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@runtime_checkable
class TokenSearch(Protocol):
    """Discrete search for a real token sequence whose activation lands in A*."""

    def search(self, prefix: str, n_adv_tokens: int, tau: float, steps: int) -> AttackResult:
        ...


# --------------------------------------------------------------------------- #
# Loop closure: re-verbalize and check geometry/readout agreement (Agent B)
# --------------------------------------------------------------------------- #
@dataclass
class LoopResult:
    prompt: str
    probe_score: float
    crossed: bool                 # geometry says eval-aware (s >= tau)
    verbalizer_readout: str       # retrieval-based AV STAND-IN (not a trained NLA)
    neighbors: list = field(default_factory=list)   # [(text, label, dist), ...]
    readout_says_eval: bool = False
    agree: bool = False           # geometry and readout agree
