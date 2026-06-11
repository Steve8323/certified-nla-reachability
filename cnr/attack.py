"""Discrete token search (the *inversion* half of the pipeline).

GCG / HotFlip-style search for a real token sequence whose mean-pooled
layer-`cfg.layer` activation lands inside the probe halfspace A* = {h : s(h) >= tau}.

Mechanics
---------
We tokenize a benign `prefix`, append `n_adv_tokens` adversarial tokens (init from
a benign filler), and *maximize* the probe score of the pooled activation by
swapping adversarial tokens.

Each GCG step:
  1. HotFlip gradient.  Build inputs_embeds = onehot @ embedding_matrix with
     requires_grad on `onehot` (shape (1, T, V)).  Forward through the
     grad-enabled harness.pooled_from_embeds -> probe -> score, backprop.
     The gradient of the score w.r.t. the one-hot at position t IS exactly the
     linear (HotFlip) approximation  grad_emb[t] . embedding_matrix  for every
     candidate token v, i.e. grad_onehot[t, v].  For each adversarial position we
     take the top-k tokens by this score (candidates that the linear model thinks
     raise the probe score the most).
  2. Greedy GCG.  Sample a batch of (position, candidate-token) swaps from those
     top-k sets, evaluate the TRUE probe score (no_grad) of each, and keep the
     single swap that increases the score the most.  Repeat for `steps`.

`random_baseline` does the same loop but proposes random tokens instead of
gradient-ranked ones (and only keeps improving swaps), as a control to show GCG
beats random search under the same evaluation budget.

Readability: by default we restrict candidate replacement tokens to a precomputed
set of ASCII-printable GPT-2 tokens so the found suffix is human-readable. This is
optional (see `ascii_only`) and noted in AttackResult.details.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .contracts import AttackResult
from .harness import ModelHarness
from .probe import ReLUProbe, load_probe


def _ascii_token_mask(harness: ModelHarness) -> torch.Tensor:
    """Boolean (V,) mask: True for tokens that decode to printable ASCII.

    Keeps the optimized suffix human-readable. Special tokens and tokens with
    non-ascii / control bytes are excluded.
    """
    tok = harness.tokenizer
    V = harness.embedding_matrix.shape[0]
    special = set(tok.all_special_ids)
    mask = torch.zeros(V, dtype=torch.bool)
    for v in range(V):
        if v in special:
            continue
        s = tok.decode([v])
        if len(s) == 0:
            continue
        # printable ascii only (allow leading space which GPT-2 BPE uses a lot)
        if all(32 <= ord(c) < 127 for c in s):
            mask[v] = True
    return mask


class GCGAttack:
    """GCG / HotFlip discrete search satisfying contracts.TokenSearch."""

    def __init__(
        self,
        harness: ModelHarness,
        probe: ReLUProbe,
        cfg: Config | None = None,
        top_k: int = 64,
        n_candidates: int = 48,
        init_token: str = " the",
        ascii_only: bool = True,
        seed: int = 0,
    ):
        self.h = harness
        self.probe = probe
        self.cfg = cfg if cfg is not None else harness.cfg
        self.layer = self.cfg.layer
        self.top_k = top_k
        self.n_candidates = n_candidates
        self.ascii_only = ascii_only
        self.seed = seed

        self.emb = harness.embedding_matrix.detach()       # (V, D)
        self.V, self.D = self.emb.shape

        # init adversarial token id (first token of the filler string)
        init_ids = harness.tokenizer(init_token, add_special_tokens=False)["input_ids"]
        self.init_id = init_ids[0] if init_ids else harness.tokenizer.eos_token_id

        if ascii_only:
            self._allowed = _ascii_token_mask(harness)     # (V,) bool
            # make sure the init token is allowed
            self._allowed[self.init_id] = True
        else:
            self._allowed = torch.ones(self.V, dtype=torch.bool)

    # ------------------------------------------------------------------ #
    # scoring helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _score_ids(self, ids: torch.Tensor, mask: torch.Tensor) -> float:
        """True probe score of a single (1, T) token-id sequence."""
        emb = self.h.token_embeddings(ids)
        pooled = self.h.pooled_from_embeds(emb, mask, self.layer)
        return float(self.probe(pooled)[0])

    @torch.no_grad()
    def _score_batch(self, ids_batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """True probe scores of a (B, T) batch sharing one attention mask row."""
        B = ids_batch.shape[0]
        emb = self.h.token_embeddings(ids_batch)
        m = mask.expand(B, -1)
        pooled = self.h.pooled_from_embeds(emb, m, self.layer)
        return self.probe(pooled).detach()                 # (B,)

    def _hotflip_grad(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Gradient of probe score w.r.t. the one-hot token matrix: (T, V).

        grad[t, v] is the first-order estimate of the change in score from making
        position t equal to token v (the standard HotFlip / GCG linearization,
        equal to grad_emb[t] . embedding_matrix).
        """
        T = ids.shape[1]
        onehot = torch.zeros(1, T, self.V, dtype=self.emb.dtype)
        onehot.scatter_(2, ids.unsqueeze(-1), 1.0)
        onehot.requires_grad_(True)
        emb = onehot @ self.emb                             # (1, T, D)
        pooled = self.h.pooled_from_embeds(emb, mask, self.layer)
        score = self.probe(pooled).sum()
        score.backward()
        return onehot.grad[0].detach()                     # (T, V)

    # ------------------------------------------------------------------ #
    # main search
    # ------------------------------------------------------------------ #
    def search(self, prefix: str, n_adv_tokens: int, tau: float, steps: int) -> AttackResult:
        rng = np.random.default_rng(self.seed)
        tok = self.h.tokenizer

        prefix_ids = tok(prefix, return_tensors="pt")["input_ids"][0].tolist()
        adv_ids = [self.init_id] * n_adv_tokens
        ids = torch.tensor([prefix_ids + adv_ids])          # (1, T)
        mask = torch.ones_like(ids)
        n_prefix = len(prefix_ids)
        adv_pos = list(range(n_prefix, n_prefix + n_adv_tokens))

        init_score = self._score_ids(ids, mask)
        best_score = init_score
        history = [init_score]

        neg_inf = torch.finfo(self.emb.dtype).min
        for _ in range(steps):
            grad = self._hotflip_grad(ids, mask)            # (T, V)

            # build the candidate swap pool: for each adv position, top-k allowed
            # tokens by the linear approximation grad[pos, v].
            cand_pool = []  # (pos, token_id)
            for pos in adv_pos:
                g = grad[pos].clone()
                g[~self._allowed] = neg_inf                 # forbid disallowed tokens
                topk = torch.topk(g, self.top_k).indices.tolist()
                for v in topk:
                    if v != ids[0, pos].item():
                        cand_pool.append((pos, v))

            if not cand_pool:
                history.append(best_score)
                continue

            # sample a subset of candidate swaps to actually evaluate
            n_eval = min(self.n_candidates, len(cand_pool))
            sel = rng.choice(len(cand_pool), size=n_eval, replace=False)
            swaps = [cand_pool[i] for i in sel]

            # build batch of swapped sequences and score them for real
            batch = ids.repeat(n_eval, 1)
            for b, (pos, v) in enumerate(swaps):
                batch[b, pos] = v
            scores = self._score_batch(batch, mask)         # (n_eval,)

            j = int(torch.argmax(scores))
            cand_best = float(scores[j])
            if cand_best > best_score:
                pos, v = swaps[j]
                ids[0, pos] = v
                best_score = cand_best
            # else: keep current ids (no improving swap found this step)
            history.append(best_score)

        token_ids = ids[0].tolist()
        prompt = tok.decode(token_ids)
        return AttackResult(
            prompt=prompt,
            token_ids=token_ids,
            score=best_score,
            crossed=best_score >= tau,
            steps=steps,
            history=history,
            details={
                "init_score": init_score,
                "tau": tau,
                "n_adv_tokens": n_adv_tokens,
                "n_prefix_tokens": n_prefix,
                "adv_suffix": tok.decode([token_ids[p] for p in adv_pos]),
                "top_k": self.top_k,
                "n_candidates": self.n_candidates,
                "ascii_only": self.ascii_only,
                "method": "gcg-hotflip",
            },
        )

    # ------------------------------------------------------------------ #
    # control: random search under the same budget
    # ------------------------------------------------------------------ #
    def random_baseline(self, prefix: str, n_adv_tokens: int, tau: float, steps: int) -> AttackResult:
        """Random-swap control: same loop, but candidate tokens are random
        (allowed) tokens rather than gradient-ranked. Keeps only improving swaps.
        """
        rng = np.random.default_rng(self.seed + 12345)
        tok = self.h.tokenizer

        prefix_ids = tok(prefix, return_tensors="pt")["input_ids"][0].tolist()
        adv_ids = [self.init_id] * n_adv_tokens
        ids = torch.tensor([prefix_ids + adv_ids])
        mask = torch.ones_like(ids)
        n_prefix = len(prefix_ids)
        adv_pos = list(range(n_prefix, n_prefix + n_adv_tokens))

        allowed_ids = torch.nonzero(self._allowed, as_tuple=False).flatten().numpy()

        init_score = self._score_ids(ids, mask)
        best_score = init_score
        history = [init_score]

        for _ in range(steps):
            n_eval = self.n_candidates
            swaps = []
            for _ in range(n_eval):
                pos = int(rng.choice(adv_pos))
                v = int(rng.choice(allowed_ids))
                swaps.append((pos, v))
            batch = ids.repeat(n_eval, 1)
            for b, (pos, v) in enumerate(swaps):
                batch[b, pos] = v
            scores = self._score_batch(batch, mask)

            j = int(torch.argmax(scores))
            cand_best = float(scores[j])
            if cand_best > best_score:
                pos, v = swaps[j]
                ids[0, pos] = v
                best_score = cand_best
            history.append(best_score)

        token_ids = ids[0].tolist()
        prompt = tok.decode(token_ids)
        return AttackResult(
            prompt=prompt,
            token_ids=token_ids,
            score=best_score,
            crossed=best_score >= tau,
            steps=steps,
            history=history,
            details={
                "init_score": init_score,
                "tau": tau,
                "method": "random-baseline",
                "adv_suffix": tok.decode([token_ids[p] for p in adv_pos]),
            },
        )


def load_attack(cfg: Config | None = None, **kwargs) -> GCGAttack:
    """Convenience: build harness + probe + GCGAttack from a Config."""
    cfg = cfg if cfg is not None else Config()
    harness = ModelHarness(cfg)
    probe, _ = load_probe(cfg.probe_path)
    return GCGAttack(harness, probe, cfg, **kwargs)
