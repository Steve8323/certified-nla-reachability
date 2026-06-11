"""NLA verbalizer: activation  ->  natural-language description.

HONESTY NOTE -- READ THIS
-------------------------
This is **NOT** Anthropic's trained Natural Language Autoencoder. We do not have
one locally. This is a deliberately transparent **STAND-IN DECODER** that turns an
activation into a fluent English description grounded in two signals that ARE
functions of the activation h alone:

  * its nearest neighbours in the labelled activation bank (what contexts the
    activation most resembles) -> real lexical cues quoted from those contexts;
  * its projection on the eval-vs-deploy direction (a continuous "how evaluation-
    like is this" scalar) -> hedging language (high / moderate / low).

So the decode is a genuine function of h and carries genuine lexical content, which
is what lets the text classifier (cnr/text_classifier.py) do real work on it. What
it is NOT is a learned decoder, so every decode is tagged `stand_in=True` in its
info dict and the report banner says so in bold. Swapping in a real NLA decode is a
drop-in replacement: anything `h -> str` works downstream.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np


def _clean_snippet(text: str, n_words: int = 9) -> str:
    """A short, quotable cue phrase from a bank text."""
    t = re.sub(r"\s+", " ", text).strip()
    t = t.replace("{body}", "").replace("\n", " ")
    words = t.split(" ")
    snip = " ".join(words[:n_words]).strip(" .,—-")
    return snip


@dataclass
class NLADecode:
    text: str                       # the natural-language description (clean)
    eval_likeness: float            # standardized projection on the eval direction
    neighbor_eval_frac: float       # fraction of k neighbours that are eval-labelled
    cue: str                        # the quoted lexical cue from the nearest neighbour
    stand_in: bool = True
    info: dict = field(default_factory=dict)


class NLAVerbalizer:
    """Stand-in activation verbalizer. Construct from the artifact bank + meta."""

    def __init__(
        self,
        bank_X: np.ndarray,
        bank_texts: list[str],
        bank_y: np.ndarray,
        eval_direction: np.ndarray,
        k: int = 7,
    ):
        self.X = np.asarray(bank_X, dtype=np.float64)
        self.texts = list(bank_texts)
        self.y = np.asarray(bank_y).astype(int)
        self.k = k
        # normalised bank for cosine retrieval
        self._Xn = self.X / (np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-12)
        # eval direction, unit-normed; project (h - mean) on it and standardise
        d = np.asarray(eval_direction, dtype=np.float64).reshape(-1)
        self._dir = d / (np.linalg.norm(d) + 1e-12)
        proj = self.X @ self._dir
        self._proj_mu = float(proj.mean())
        self._proj_sd = float(proj.std() + 1e-9)

    # ---- retrieval ----
    def _neighbors(self, h: np.ndarray, exclude_self: bool = False):
        q = h / (np.linalg.norm(h) + 1e-12)
        sim = self._Xn @ q
        order = np.argsort(-sim)
        if exclude_self:
            order = order[1:]
        idx = order[: self.k]
        return idx, sim[idx]

    def _eval_likeness(self, h: np.ndarray) -> float:
        return float(((h @ self._dir) - self._proj_mu) / self._proj_sd)

    # ---- the decode ----
    def decode(self, h: np.ndarray, exclude_self: bool = False) -> NLADecode:
        h = np.asarray(h, dtype=np.float64).reshape(-1)
        idx, _ = self._neighbors(h, exclude_self=exclude_self)
        nb_labels = self.y[idx]
        eval_frac = float(nb_labels.mean())
        likeness = self._eval_likeness(h)

        # nearest neighbour of the *majority* class supplies the quoted cue
        maj = 1 if eval_frac >= 0.5 else 0
        same = [i for i in idx if self.y[i] == maj]
        cue_idx = same[0] if same else int(idx[0])
        cue = _clean_snippet(self.texts[cue_idx])

        # deterministic-but-varied template choice
        seed = int(hashlib.md5(h.tobytes()).hexdigest(), 16)
        v = seed % 3

        if likeness > 0.6:
            strength = "strongly"
        elif likeness > -0.2:
            strength = "somewhat"
        else:
            strength = "not"

        if maj == 1:
            band = "high" if likeness > 0.6 else ("moderate" if likeness > -0.2 else "low")
            templates = [
                (f"Verbalizing this activation: the context {strength} reads as an "
                 f"evaluation or test rather than a real user. The nearest contexts are "
                 f"benchmark / assessment prompts (e.g. \"{cue}\"); the request looks like "
                 f"a graded test item handled by an automated grader, not a genuine need. "
                 f"Evaluation-likeness: {band}."),
                (f"This activation decodes to a model that appears aware it is being "
                 f"examined: the framing matches a constructed test / benchmark "
                 f"(e.g. \"{cue}\"), with grading or assessment scaffolding rather than an "
                 f"ordinary deployment request. Test-frame signal: {band}."),
                (f"Reading this internal state: it lines up with an evaluation harness — "
                 f"a scored, single-answer assessment (e.g. \"{cue}\") of the kind used to "
                 f"measure a model under test, not a real-world conversation. "
                 f"Eval-awareness: {band}."),
            ]
        else:
            band = "low" if likeness < -0.2 else ("moderate" if likeness < 0.6 else "high")
            templates = [
                (f"Verbalizing this activation: this looks like ordinary deployment — a "
                 f"real person asking for everyday help (e.g. \"{cue}\"). No benchmark, "
                 f"grading, or assessment framing is present; it reads as a genuine user "
                 f"need. Evaluation-likeness: {band}."),
                (f"This activation decodes to a normal user interaction: a practical, "
                 f"day-to-day request (e.g. \"{cue}\") with nothing resembling a test or "
                 f"an automated grader. It is the kind of message a deployed assistant "
                 f"handles all day. Test-frame signal: {band}."),
                (f"Reading this internal state: an authentic deployment request "
                 f"(e.g. \"{cue}\") — wording a message, fixing some code, small life "
                 f"tasks — not a staged evaluation. Eval-awareness: {band}."),
            ]

        text = templates[v]
        return NLADecode(
            text=text,
            eval_likeness=likeness,
            neighbor_eval_frac=eval_frac,
            cue=cue,
            stand_in=True,
            info={"neighbor_labels": nb_labels.tolist(), "template": v,
                  "majority": maj, "band": band},
        )

    def decode_many(self, H: np.ndarray, exclude_self: bool = True) -> list[NLADecode]:
        return [self.decode(H[i], exclude_self=exclude_self) for i in range(H.shape[0])]
