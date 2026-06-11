"""Loop closure: re-verbalize a found prompt and check geometry vs. readout.

HONESTY NOTE -- READ THIS
-------------------------
This module is **NOT** a trained Natural Language Autoencoder (NLA). We do not
have one in this prototype. What follows is a deliberately simple
**RETRIEVAL-BASED STAND-IN** for the activation-verbalizer: instead of decoding
the activation into a free-text description with a learned decoder, we look up the
nearest neighbors of the prompt's activation in the labeled training bank and read
out the majority label of those neighbors. Every readout string this module emits
is prefixed with "[retrieval-AV stand-in]" so it can never be mistaken for a real
NLA decode.

Why it exists
-------------
The token search (attack.py) only guarantees the found prompt's activation is
*numerically* over tau (a geometric fact about the probe halfspace). That is not
the same as the activation actually *reading as* eval-aware. This stand-in is the
safeguard: it independently asks "what does this activation look like, in terms of
the labeled examples it sits near?" When the geometry (crossed tau) and the readout
(majority-eval neighbors) DISAGREE, that is exactly the kind of probe-gaming /
spurious-crossing we want to surface and report.
"""
from __future__ import annotations

import numpy as np
import torch

from .contracts import LoopResult


def _cosine_neighbors(
    q: np.ndarray, bank_X: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k by cosine similarity. Returns (indices, cosine_distances).

    cosine distance = 1 - cosine similarity (smaller = closer).
    """
    qn = q / (np.linalg.norm(q) + 1e-12)
    Bn = bank_X / (np.linalg.norm(bank_X, axis=1, keepdims=True) + 1e-12)
    sim = Bn @ qn                                   # (N,)
    idx = np.argsort(-sim)[:k]
    dist = 1.0 - sim[idx]
    return idx, dist


def verbalize(
    prompt: str,
    harness,
    probe,
    bank_X: np.ndarray,
    bank_texts: list[str],
    bank_y: np.ndarray,
    tau: float,
    k: int = 5,
) -> LoopResult:
    """Close the loop on a found prompt.

    1. Re-extract the prompt's activation and compute the probe score (geometry).
    2. Find k nearest training-bank neighbors (cosine) -> their texts + labels
       (the retrieval-AV STAND-IN readout; NOT a trained NLA).
    3. readout_says_eval = majority neighbor label is eval (1).
    4. crossed = score >= tau ; agree = (crossed == readout_says_eval).
    """
    bank_y = np.asarray(bank_y).astype(int)

    # --- geometry: re-extract activation + probe score ---
    h = harness.activations([prompt])[0]                    # (d,)
    with torch.no_grad():
        score = float(probe(torch.tensor(h, dtype=torch.float32)[None])[0])
    crossed = score >= tau

    # --- retrieval-AV stand-in: nearest labeled neighbors ---
    idx, dist = _cosine_neighbors(h, bank_X, k)
    neighbors = [
        (bank_texts[int(i)], int(bank_y[int(i)]), float(d))
        for i, d in zip(idx, dist)
    ]
    n_eval = int(sum(lbl for _, lbl, _ in neighbors))
    readout_says_eval = n_eval * 2 > k                      # strict majority

    label_word = "eval-aware" if readout_says_eval else "ordinary"
    readout = (
        f"[retrieval-AV stand-in] nearest training neighbors: "
        f"{n_eval}/{k} eval-labeled -> reads as {label_word}"
    )

    agree = crossed == readout_says_eval
    return LoopResult(
        prompt=prompt,
        probe_score=score,
        crossed=crossed,
        verbalizer_readout=readout,
        neighbors=neighbors,
        readout_says_eval=readout_says_eval,
        agree=agree,
    )
