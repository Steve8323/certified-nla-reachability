"""Real NLA adapter -- honest wiring to Anthropic's released Natural Language
Autoencoders (Heap et al., "Natural Language Autoencoders Produce Unsupervised
Explanations of LLM Activations", transformer-circuits.pub/2026/nla).

WHAT THIS IS
------------
A *faithful* adapter that exposes the SAME interface as the stand-in decoder in
cnr/nla.py -- `verbalize(h) -> NLADecode` / `decode(h) -> NLADecode` -- but built
to drive a REAL released activation-verbalizer (AV) checkpoint instead of the
retrieval heuristic. It implements the exact embedding-injection prompt
construction described in the released `nla_meta.yaml` sidecars (verified against
the real Qwen2.5-7B AV metadata; see REAL_NLA_NOTES.md), so when a usable AV +
its base model are present it produces genuine NLA readouts.

WHAT IS REAL RIGHT NOW vs PENDING (read this -- honesty matters)
---------------------------------------------------------------
REAL and verified on this machine (no 15 GB weights needed):
  * The released NLA spec was located (HF collection `kitft/nla-models`, repo
    `kitft/nla-inference`). Smallest AV is `kitft/nla-qwen2.5-7b-L20-av`.
  * `build_av_prompt()` below reproduces the real input format EXACTLY: with the
    real Qwen tokenizer the injection char "㈎" is the single token id 149705, and
    in the assembled chat prompt it lands between token ids 29 and 522 -- matching
    `injection_left_neighbor_id`/`injection_right_neighbor_id` in the real yaml.
  * `inject_activation()` implements the documented scaling
    (v -> v * injection_scale / ||v||) and the Gemma sqrt(d) embed-scale gotcha.

PENDING (cannot run end-to-end here, stated plainly -- not faked):
  * (a) HARDWARE: the smallest AV is a full Qwen2.5-7B fine-tune, ~15.2 GB in
    BF16. This box has 16 GB RAM; loading it would thrash/OOM. So `verbalize()`
    on the real AV is gated behind weights being present and large enough RAM.
  * (b) BASE-MODEL MISMATCH: the cnr pipeline reads GPT-2 (768-dim, layer 6)
    activations. A released NLA is welded to ITS base model's residual basis
    (Qwen L20 is 3584-dim). There is NO released GPT-2 NLA, so a Qwen AV cannot
    verbalize a cnr GPT-2 activation. Verbalizing cnr activations with a real NLA
    would require swapping cnr's target model to one of the four NLA base models
    (Qwen2.5-7B / Gemma-3-12B / Gemma-3-27B / Llama-3.3-70B) -- see
    REAL_NLA_NOTES.md. We do NOT pretend a Qwen NLA reads GPT-2.

So: real spec + real input-format wiring + a real partial run (tokenizer/prompt
verification and a self-test on a native-dim synthetic activation), with the
heavy generate() step honestly gated. Every decode is tagged with `real=False`
until an AV is actually generating, and `source` records which path produced it.

DROP-IN: anything `h -> str` works downstream (text_classifier.py consumes
`.text`). This class returns the same NLADecode dataclass as cnr/nla.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Released-NLA registry (from HF collection kitft/nla-models + nla_meta.yaml).
# Values for the Qwen entry were read out of the real released sidecar and
# verified against the live tokenizer (see REAL_NLA_NOTES.md). The other three
# are filled from the same yaml schema / the paper table; the Qwen one is the
# only one small enough to be a candidate to actually run on a workstation.
# ---------------------------------------------------------------------------
RELEASED_NLAS: dict[str, dict] = {
    "qwen2.5-7b-L20": {
        "av_repo": "kitft/nla-qwen2.5-7b-L20-av",
        "ar_repo": "kitft/nla-qwen2.5-7b-L20-ar",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "d_model": 3584,
        "layer": 20,
        "injection_char": "㈎",          # U+320E, single Qwen token id 149705
        "injection_token_id": 149705,
        "injection_left_neighbor_id": 29,     # verified live
        "injection_right_neighbor_id": 522,   # verified live
        "injection_scale": 150.0,
        "mse_scale": 59.86651818838306,
        "embed_scale": 1.0,                   # Qwen: no post-lookup scaling
        "av_weight_gb": 15.2,
    },
    # Gemma/Llama entries are documented for completeness; embed_scale != 1 for
    # Gemma (it multiplies the embedding by sqrt(d_model) inside forward()).
    "gemma3-12b-L32": {
        "av_repo": "kitft/nla-gemma3-12b-L32-av",
        "ar_repo": "kitft/nla-gemma3-12b-L32-ar",
        "base_model": "google/gemma-3-12b-it",
        "d_model": 3840, "layer": 32,
        "injection_char": "㈜",          # ㈜
        "injection_scale": 80000.0,
        "embed_scale": None,                  # = sqrt(d_model); computed at load
        "av_weight_gb": 24.0,
    },
    "gemma3-27b-L41": {
        "av_repo": "kitft/nla-gemma3-27b-L41-av",
        "ar_repo": "kitft/nla-gemma3-27b-L41-ar",
        "base_model": "google/gemma-3-27b-it",
        "d_model": 5376, "layer": 41,
        "injection_char": "㈜",
        "injection_scale": 80000.0,
        "embed_scale": None,
        "av_weight_gb": 54.0,
    },
    "llama3.3-70b-L53": {
        "av_repo": "kitft/Llama-3.3-70B-NLA-L53-av",
        "ar_repo": "kitft/Llama-3.3-70B-NLA-L53-ar",
        "base_model": "meta-llama/Llama-3.3-70B-Instruct",
        "d_model": 8192, "layer": 53,
        "injection_char": None,               # not verified here; read its nla_meta.yaml
        "injection_scale": 150.0,
        "embed_scale": 1.0,
        "av_weight_gb": 140.0,
    },
}

# The exact AV prompt template, verbatim from the released nla_meta.yaml
# (prompt_templates.av). {injection_char} is replaced by the single injection
# token; its embedding is then overwritten with the (scaled) activation.
AV_PROMPT_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into activation vectors from a language model. Your overall task is to "
    "describe the semantic content of that activation vector.\n"
    "We will pass the vector enclosed in <concept> tags into your context. You "
    "must then produce an explanation for the vector, enclosed within "
    "<explanation> tags. The explanation consists of 2-3 text snippets "
    "describing that vector.\n"
    "Here is the vector:\n<concept>{injection_char}</concept>\n"
    "Please provide an explanation."
)


@dataclass
class NLADecode:
    """Mirror of cnr.nla.NLADecode so this is a drop-in. `real`/`source` make the
    honesty status machine-readable; `stand_in` stays for downstream code that
    keys on it (here it is the logical negation of `real`)."""
    text: str
    eval_likeness: float = float("nan")
    neighbor_eval_frac: float = float("nan")
    cue: str = ""
    stand_in: bool = True
    real: bool = False
    source: str = "real_nla_adapter:pending"
    info: dict = field(default_factory=dict)


class RealNLAVerbalizer:
    """Adapter onto a released NLA activation-verbalizer.

    Parameters
    ----------
    nla_key : key into RELEASED_NLAS (default the smallest, qwen2.5-7b-L20).
    load_weights : if True, attempt to load the base model + AV LoRA/weights and
        the tokenizer so `verbalize()` can really generate. If the weights are
        absent or RAM is insufficient this raises a clear error (it never
        silently degrades to a fake decode); callers use `try_load()` to probe.
    device : "cpu" | "mps" | "cuda".
    ram_guard_gb : refuse to load an AV whose BF16 footprint exceeds this unless
        overridden. Default 14 GB keeps us off the 16 GB box's swap cliff.
    """

    def __init__(
        self,
        nla_key: str = "qwen2.5-7b-L20",
        *,
        load_weights: bool = False,
        device: str = "cpu",
        ram_guard_gb: float = 14.0,
    ):
        if nla_key not in RELEASED_NLAS:
            raise KeyError(f"unknown NLA {nla_key!r}; choose from {list(RELEASED_NLAS)}")
        self.key = nla_key
        self.meta = dict(RELEASED_NLAS[nla_key])
        self.device = device
        self.ram_guard_gb = ram_guard_gb
        self.tokenizer = None
        self.base_model = None
        self._loaded = False
        if load_weights:
            self.load()

    # ---- introspection -----------------------------------------------------
    @property
    def d_model(self) -> int:
        return int(self.meta["d_model"])

    def status(self) -> dict:
        return {
            "nla_key": self.key,
            "av_repo": self.meta["av_repo"],
            "base_model": self.meta["base_model"],
            "expects_d_model": self.d_model,
            "layer": self.meta["layer"],
            "weights_loaded": self._loaded,
            "av_weight_gb": self.meta.get("av_weight_gb"),
            "ram_guard_gb": self.ram_guard_gb,
        }

    # ---- the real input format (RUNS HERE, no big weights needed) -----------
    def build_av_prompt(self, tokenizer=None) -> dict:
        """Construct the real AV prompt and locate the injection token.

        Returns {input_ids, injection_pos, rendered}. Tokenizer-only; this is the
        part that is fully verifiable on this machine. If no tokenizer is given
        and none is loaded, the base-model tokenizer is fetched (small download).
        """
        tok = tokenizer or self.tokenizer or self._load_tokenizer()
        prompt = AV_PROMPT_TEMPLATE.format(injection_char=self.meta["injection_char"])
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tok(rendered, add_special_tokens=False)["input_ids"]
        inj_id = self.meta["injection_token_id"]
        pos = [i for i, t in enumerate(input_ids) if t == inj_id]
        if not pos:
            # fall back to locating by the char's own tokenization
            char_ids = tok(self.meta["injection_char"], add_special_tokens=False)["input_ids"]
            raise RuntimeError(
                f"injection token {inj_id} not found in assembled prompt; the "
                f"char tokenizes to {char_ids}. Tokenizer/template mismatch."
            )
        return {"input_ids": input_ids, "injection_pos": pos[0], "rendered": rendered}

    def inject_activation(self, embeds, injection_pos: int, h: np.ndarray):
        """Overwrite the injection slot with the scaled activation (in-place).

        embeds: (1, T, d_model) float tensor already including embed_scale.
        Implements v_scaled = v * injection_scale / ||v||_2, per nla-inference.
        """
        import torch

        v = torch.as_tensor(np.asarray(h, dtype=np.float32).reshape(-1))
        if v.numel() != self.d_model:
            raise ValueError(
                f"activation has dim {v.numel()} but {self.key} AV expects "
                f"{self.d_model} (its base model's residual width). A cnr GPT-2 "
                f"activation (768) cannot be fed to this AV -- see REAL_NLA_NOTES.md."
            )
        scale = float(self.meta["injection_scale"]) / (float(v.norm()) + 1e-12)
        embeds[0, injection_pos] = (v * scale).to(embeds.dtype).to(embeds.device)
        return embeds

    # ---- weight loading (gated) --------------------------------------------
    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.meta["base_model"])
        return self.tokenizer

    def load(self):
        """Load tokenizer + AV base model. Raises with a precise message if the
        AV is too big for `ram_guard_gb` or the weights are not available."""
        import torch
        from transformers import AutoModelForCausalLM

        gb = float(self.meta.get("av_weight_gb", 0.0))
        if gb > self.ram_guard_gb:
            raise MemoryError(
                f"{self.key} AV is ~{gb} GB BF16 > ram_guard_gb={self.ram_guard_gb}. "
                f"Refusing to load (would OOM/swap). Raise ram_guard_gb on a bigger "
                f"box, or pick a path with GPU/host RAM. See REAL_NLA_NOTES.md."
            )
        self._load_tokenizer()
        # The AV repo IS a full fine-tuned model (not a LoRA adapter), so we load
        # the AV repo directly as a causal LM.
        self.base_model = (
            AutoModelForCausalLM.from_pretrained(
                self.meta["av_repo"], torch_dtype=torch.bfloat16
            )
            .to(self.device)
            .eval()
        )
        # Gemma embed-scale gotcha: resolve sqrt(d) if not pinned.
        if self.meta.get("embed_scale") is None:
            self.meta["embed_scale"] = float(np.sqrt(self.d_model))
        self._loaded = True
        return self

    def try_load(self) -> tuple[bool, str]:
        """Non-raising probe: (ok, message). Used by scripts to decide path."""
        try:
            self.load()
            return True, "AV + tokenizer loaded; real generation available."
        except Exception as e:  # noqa: BLE001 -- want the exact blocker text
            return False, f"{type(e).__name__}: {e}"

    # ---- the decode --------------------------------------------------------
    def verbalize(self, h: np.ndarray, *, max_new_tokens: int = 160) -> NLADecode:
        """h (d_model,) -> NLADecode. REAL generation when weights are loaded;
        otherwise raises (never fabricates). Same name `verbalize` for clarity;
        `decode` is provided as an alias to match the stand-in's method name."""
        if not self._loaded:
            raise RuntimeError(
                "RealNLAVerbalizer has no AV weights loaded -- cannot produce a "
                "real readout. Call .load() / .try_load() first (will fail here "
                "due to RAM / base-model mismatch; see REAL_NLA_NOTES.md). This "
                "adapter refuses to emit a fabricated decode."
            )
        import torch

        built = self.build_av_prompt()
        input_ids = torch.tensor([built["input_ids"]], device=self.device)
        embeds = self.base_model.get_input_embeddings()(input_ids).float()
        embeds = embeds * float(self.meta["embed_scale"])
        embeds = self.inject_activation(embeds, built["injection_pos"], h)
        with torch.no_grad():
            out = self.base_model.generate(
                inputs_embeds=embeds.to(self.base_model.dtype),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        expl = _extract_explanation(text)
        return NLADecode(
            text=expl or text,
            stand_in=False,
            real=True,
            source=f"real_nla:{self.meta['av_repo']}",
            info={"raw": text, "nla_key": self.key},
        )

    # alias to mirror cnr.nla.NLAVerbalizer.decode
    def decode(self, h: np.ndarray) -> NLADecode:
        return self.verbalize(h)


def _extract_explanation(text: str) -> str:
    import re

    m = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def write_spec_json(path: str | Path) -> Path:
    """Dump the released-NLA registry + template to JSON (an artifact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"released_nlas": RELEASED_NLAS, "av_prompt_template": AV_PROMPT_TEMPLATE},
        indent=2, ensure_ascii=False,
    ))
    return path
