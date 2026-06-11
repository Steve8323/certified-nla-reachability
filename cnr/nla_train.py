"""Train a model-specific Natural Language Autoencoder by reinforcement learning.

The NLA must belong to the target network — GPT-2 activations get a GPT-2 NLA. The
autoencoder:

    encode:  GPT-2 activation h  --proj-->  soft prefix  --BART(frozen)-->  text s
    decode:  text s  --GPT-2 (the target model itself)-->  activation ĥ
    reward:  centered cosine(ĥ, h)      "does the LANGUAGE reproduce the model's state?"

Only the activation→prefix projection is trained; BART is the frozen base decoder LM
and GPT-2 is the frozen target/decoder. Trained with REINFORCE (batch-mean baseline)
— the discrete-text bottleneck is exactly why it has to be RL, as the NLA paper states.

Why this fixes the leak: no labels enter the loop. The description is rewarded purely
for reconstructing the activation, so a downstream classifier reading it is reading
decoded CONTENT, not the k-NN label the retrieval stand-in baked in. Cosine is computed
on activations CENTERED by the bank mean, so the reward reflects the informative
deviation, not the large common-mode component that mean-pooled activations share.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BartForConditionalGeneration


class ActivationToPrefix(nn.Module):
    """h (B, d_act) -> k soft prefix vectors (B, k, d_model) in BART's input space."""

    def __init__(self, d_act: int, d_model: int, k: int, hidden: int = 1024):
        super().__init__()
        self.k, self.d_model = k, d_model
        self.net = nn.Sequential(
            nn.Linear(d_act, hidden), nn.GELU(), nn.Linear(hidden, k * d_model))
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).view(-1, self.k, self.d_model)


class TrainedNLA:
    def __init__(self, harness, device=None, k: int = 8, max_tokens: int = 18,
                 base: str = "facebook/bart-large"):
        self.harness = harness
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(base)
        self.bart = BartForConditionalGeneration.from_pretrained(base).float().to(self.device).eval()
        for p in self.bart.parameters():
            p.requires_grad_(False)
        self.d_model = self.bart.config.d_model
        self.k, self.max_tokens = k, max_tokens
        self.pad = self.tok.pad_token_id
        self.proj = ActivationToPrefix(harness.d_model, self.d_model, k).to(self.device)
        self.h_mean = np.zeros(harness.d_model, dtype=np.float64)   # set from the bank

    # ---- BART encoder from the activation-derived soft prefix ----
    def _encoder(self, h_t: torch.Tensor):
        soft = self.proj(h_t)
        amask = torch.ones(h_t.shape[0], self.k, dtype=torch.long, device=self.device)
        return self.bart.get_encoder()(inputs_embeds=soft, attention_mask=amask), amask

    @torch.no_grad()
    def generate(self, h, sample: bool = True):
        h_t = torch.as_tensor(np.atleast_2d(h), dtype=torch.float32, device=self.device)
        enc, amask = self._encoder(h_t)
        out = self.bart.generate(
            encoder_outputs=enc, attention_mask=amask, do_sample=sample, top_p=0.9,
            num_beams=1, min_new_tokens=5, max_new_tokens=self.max_tokens,
            no_repeat_ngram_size=3)
        return out, self.tok.batch_decode(out, skip_special_tokens=True)

    def seq_logprob(self, h_t: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        enc, amask = self._encoder(h_t)                       # grad flows to proj
        din, tgt = token_ids[:, :-1], token_ids[:, 1:]
        logits = self.bart(encoder_outputs=enc, attention_mask=amask, decoder_input_ids=din).logits
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = (tgt != self.pad).float()
        return (lp * mask).sum(-1)

    def train_supervised(self, prompts, H, steps=250, batch=8, lr=1e-4,
                         max_len=48, log=print, seed=0):
        """Stable warm-start: teacher-force BART to reconstruct each activation's
        SOURCE PROMPT from its activation prefix. Still model-specific (decodes
        GPT-2's activation) and label-free (target is the prompt, not the eval label).
        This is what keeps the later RL phase from collapsing."""
        self.h_mean = np.asarray(H, np.float64).mean(0)
        rng = np.random.default_rng(seed)
        opt = torch.optim.Adam(self.proj.parameters(), lr=lr)
        hist = []
        N = len(prompts)
        for step in range(steps):
            idx = rng.choice(N, size=batch, replace=False)
            h_t = torch.as_tensor(H[idx], dtype=torch.float32, device=self.device)
            enc, amask = self._encoder(h_t)
            labels = self.tok([prompts[i] for i in idx], return_tensors="pt",
                              padding=True, truncation=True, max_length=max_len
                              ).input_ids.to(self.device)
            labels = labels.masked_fill(labels == self.pad, -100)
            loss = self.bart(encoder_outputs=enc, attention_mask=amask, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.proj.parameters(), 1.0)
            opt.step()
            hist.append(float(loss))
            if step % 25 == 0 or step == steps - 1:
                _, gt = self.generate(H[idx][:1], sample=False)
                log(f"sft {step:3d} | CE {float(loss):.3f} | e.g. {gt[0][:58]!r}")
        return hist

    def recon_cos(self, texts, h) -> np.ndarray:
        """Centered cosine between GPT-2's activation of `texts` and the originals."""
        safe = [t if t.strip() else "." for t in texts]
        hhat = self.harness.activations(safe).astype(np.float64) - self.h_mean
        h = np.atleast_2d(np.asarray(h, np.float64)) - self.h_mean
        a = hhat / (np.linalg.norm(hhat, axis=1, keepdims=True) + 1e-9)
        b = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
        return (a * b).sum(1)

    @staticmethod
    def _rep_penalty(token_ids, pad) -> np.ndarray:
        out = []
        for row in token_ids.tolist():
            toks = [t for t in row if t != pad]
            out.append(1.0 if len(toks) <= 1 else 1.0 - len(set(toks)) / len(toks))
        return np.asarray(out)

    def train(self, H, steps=100, batch=6, lr=1e-4, rep_pen=0.5, log=print, seed=0):
        self.h_mean = np.asarray(H, np.float64).mean(0)
        rng = np.random.default_rng(seed)
        opt = torch.optim.Adam(self.proj.parameters(), lr=lr)
        hist = {"reward": [], "cos": [], "greedy": []}
        N = H.shape[0]
        for step in range(steps):
            idx = rng.choice(N, size=batch, replace=False)
            h = H[idx]
            h_t = torch.as_tensor(h, dtype=torch.float32, device=self.device)
            tok_ids, texts = self.generate(h, sample=True)
            cos = self.recon_cos(texts, h)
            reward = cos - rep_pen * self._rep_penalty(tok_ids, self.pad)
            adv = torch.as_tensor(reward - reward.mean(), dtype=torch.float32, device=self.device)
            logp = self.seq_logprob(h_t, tok_ids)
            loss = -(adv.detach() * logp).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.proj.parameters(), 1.0)
            opt.step()
            hist["reward"].append(float(reward.mean()))
            hist["cos"].append(float(cos.mean()))
            if step % 10 == 0 or step == steps - 1:
                _, gt = self.generate(h, sample=False)
                gcos = float(self.recon_cos(gt, h).mean())
                hist["greedy"].append([step, gcos])
                log(f"step {step:3d} | reward {reward.mean():+.3f} | cos {cos.mean():+.3f} "
                    f"| greedy-cos {gcos:+.3f} | e.g. {gt[0][:58]!r}")
        return hist

    @torch.no_grad()
    def decode(self, h) -> str:
        return self.generate(h, sample=False)[1][0]

    def save(self, path):
        torch.save({"proj": self.proj.state_dict(), "k": self.k,
                    "max_tokens": self.max_tokens, "h_mean": self.h_mean.tolist()}, path)

    def load(self, path):
        b = torch.load(path, map_location=self.device, weights_only=False)
        self.proj.load_state_dict(b["proj"])
        self.h_mean = np.asarray(b["h_mean"], np.float64)
        self.max_tokens = b.get("max_tokens", self.max_tokens)
        return self
