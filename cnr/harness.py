"""ModelHarness: load the cached target model and extract residual-stream
activations. This is the shared spine -- the verifier needs only the saved
probe + anchor vectors, but the attack search differentiates the probe score
through this harness, so it exposes a grad-enabled embedding -> activation path.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config


class ModelHarness:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(cfg.model_name)
            .to(self.device)
            .eval()
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.d_model = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers

    # --- embedding matrix (for discrete token search) ---
    @property
    def embedding_matrix(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight  # (vocab, d)

    def token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    # --- mean pooling helper ---
    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(hidden.dtype)            # (b, t, 1)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)

    @torch.no_grad()
    def activations(
        self, prompts: list[str], layer: int | None = None, batch_size: int = 16
    ) -> np.ndarray:
        """(n, d) pooled residual-stream activations at `layer`."""
        layer = self.cfg.layer if layer is None else layer
        out = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_len,
            ).to(self.device)
            hs = self.model(**enc, output_hidden_states=True).hidden_states[layer]
            pooled = self._mean_pool(hs, enc["attention_mask"])
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def pooled_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: int | None = None,
    ) -> torch.Tensor:
        """Grad-enabled (n, d) pooled activation at `layer` from input embeddings.

        Used by the discrete token search to backprop the probe score onto the
        token embeddings. No torch.no_grad here on purpose.
        """
        layer = self.cfg.layer if layer is None else layer
        hs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[layer]
        return self._mean_pool(hs, attention_mask)
