# Real NLA — integration notes & honest status

Replacing the `cnr/nla.py` **stand-in** verbalizer with Anthropic's **released**
Natural Language Autoencoder (NLA). Companion code: `cnr/real_nla.py`,
`scripts/06_real_nla.py`. Run record: `artifacts/real_nla/run.json`. Report:
`reports/realnla_<ts>.html`.

**Bottom-line status: PARTIAL (honest).** The released NLAs are real and were
located; this adapter reproduces their *exact* activation-injection input format
(verified live against the released metadata). But **no real verbalization of a
cnr activation was produced on this machine**, for two concrete reasons (below).
Nothing here is faked — the heavy generation step is gated, not stubbed.

---

## 1. What was actually released

- **Paper:** *Natural Language Autoencoders Produce Unsupervised Explanations of
  LLM Activations*, transformer-circuits.pub/2026/nla (Anthropic).
- **Code:** `github.com/kitft/natural_language_autoencoders` (training) +
  `github.com/kitft/nla-inference` (inference client).
- **Weights:** HF collection **`kitft/nla-models`** — 8 checkpoints = an
  **activation verbalizer (AV)** + **activation reconstructor (AR)** for each of
  4 base models. Both AV and AR are *full fine-tunes* of the base model (not LoRA
  adapters), trained with RL to reconstruct residual-stream activations.
- **Frontend:** interactive sampling on Neuronpedia (e.g.
  `neuronpedia.org/gemma-3-27b-it/nla`).

| key | AV repo | base model | d_model | layer | AV size (BF16) |
|---|---|---|---|---|---|
| qwen2.5-7b-L20  | `kitft/nla-qwen2.5-7b-L20-av`  | `Qwen/Qwen2.5-7B-Instruct`    | 3584 | 20 | **~15.2 GB** |
| gemma3-12b-L32  | `kitft/nla-gemma3-12b-L32-av`  | `google/gemma-3-12b-it`       | 3840 | 32 | ~24 GB |
| gemma3-27b-L41  | `kitft/nla-gemma3-27b-L41-av`  | `google/gemma-3-27b-it`       | 5376 | 41 | ~54 GB |
| llama3.3-70b-L53| `kitft/Llama-3.3-70B-NLA-L53-av`| `meta-llama/Llama-3.3-70B-Instruct` | 8192 | 53 | ~140 GB |

**Smallest AV is Qwen2.5-7B** (3584-d residual). There is **no GPT-2 / sub-7B
NLA** released.

### How the AV works (mechanism)
The AV takes a fixed prompt containing a single special "injection" token, and
**overwrites that token's embedding with the (scaled) activation vector**, then
autoregressively writes a 2–3 snippet explanation inside `<explanation>…</explanation>`.

Exact AV prompt template (verbatim from the released `nla_meta.yaml`):
```
You are a meticulous AI researcher conducting an important investigation into
activation vectors from a language model. Your overall task is to describe the
semantic content of that activation vector.
We will pass the vector enclosed in <concept> tags into your context. You must
then produce an explanation for the vector, enclosed within <explanation> tags.
The explanation consists of 2-3 text snippets describing that vector.
Here is the vector:
<concept>{injection_char}</concept>
Please provide an explanation.
```

Released `nla_meta.yaml` fields used (Qwen2.5-7B-L20 AV, read from the actual file):
```yaml
role: av            # AR role uses prompt_templates.ar: "Summary of the following text: <text>{explanation}</text> <summary>"
d_model: 3584
extraction_layer_index: 20
extraction:
  injection_scale: 150.0          # v_scaled = v * injection_scale / ||v||_2
  mse_scale: 59.86651818838306
tokens:
  injection_char: ㈎              # U+320E
  injection_token_id: 149705      # already a Qwen vocab id (no added token)
  injection_left_neighbor_id: 29
  injection_right_neighbor_id: 522
```

**Per-family injection details** (from `nla-inference`):
- Qwen: injection char `㈎`, `injection_scale=150`, `embed_scale=1.0`.
- Gemma: injection char `㈜`, `injection_scale=80000`, and a **√d embed-scale
  gotcha** — Gemma multiplies embeddings by `sqrt(d_model)` *inside* `forward()`,
  so when you build `inputs_embeds` manually you must apply `embed_scale = √d`
  (~61.97 for 3840-d) after the lookup, *before* injecting.
- Default serving is **SGLang** (`--disable-radix-cache`), which accepts
  `input_embeds` over HTTP. `cnr/real_nla.py` instead uses HF
  `model.generate(inputs_embeds=...)` so no server is needed.

---

## 2. What was VERIFIED on this machine (real, no 15 GB weights)

Run `python scripts/06_real_nla.py`. All three pass:

1. **Prompt-format reproduction (REAL_PASS).** Using the *real* Qwen tokenizer,
   `cnr.real_nla.build_av_prompt()` places injection token id **149705** (char
   `㈎`) at position **111** of the assembled chat prompt, with left neighbor
   **29** and right neighbor **522** — which **match** the released yaml's
   `injection_left_neighbor_id`/`injection_right_neighbor_id` exactly. This proves
   the prompt construction is byte-for-byte the released input format.
2. **Injection self-test (REAL_PASS).** On a synthetic 3584-d activation, after
   `v · injection_scale/||v||` the injected embedding slot has norm **150.0**,
   matching `injection_scale=150.0`. The injection math is correct.
3. **cnr GPT-2 path (REAL_PASS).** cnr's own extraction runs end to end: GPT-2,
   layer 6, mean-pooled → activations of shape (6, 768).

The tokenizer download (~13 MB) is the only network cost; no AV weights needed.

---

## 3. Why no real readout of a cnr activation here (the two blockers)

**(a) Hardware.** The smallest AV is a full Qwen2.5-7B fine-tune, **~15.2 GB
BF16**. This box has **16 GB RAM**. Loading it would thrash/OOM, so
`RealNLAVerbalizer.load()` refuses past a `ram_guard_gb=14` guard (captured
verbatim in the run as `MemoryError: …~15.2 GB BF16 > ram_guard_gb=14.0`).

**(b) Base-model mismatch (the deeper one).** An NLA is welded to **its own**
base model's residual basis. The released AVs read Qwen2.5-7B (3584-d, L20),
Gemma-3, or Llama-3.3. The cnr pipeline reads **GPT-2 (768-d, L6)**. A 768-d
vector cannot be injected into a 3584-d embedding slot — the adapter raises
`ValueError: activation has dim 768 but qwen2.5-7b-L20 AV expects 3584`. Because
**no GPT-2 NLA exists**, a real NLA cannot verbalize a cnr activation as-is. We do
not fabricate one (the stand-in in `cnr/nla.py` *can* emit text for any dim, which
is exactly the side-by-side contrast shown in the report).

---

## 4. Exact path to genuine readouts

To make `RealNLAVerbalizer.verbalize(h)` return a real explanation:

1. **Switch cnr's target model to an NLA base model.** Easiest: Qwen.
   In `cnr/config.py` set `model_name = "Qwen/Qwen2.5-7B-Instruct"`, `layer = 20`.
   `cnr/harness.py` already reads `hidden_states[layer]` generically, so the
   activation shape becomes 3584-d automatically.
2. **Extract the activation the way the NLA was trained** — i.e. the **last-token
   (unpooled) residual at L20**, not mean-pooled. (NLAs are trained on per-token
   activations; mean-pooling is a cnr stand-in choice. For a faithful readout use
   the residual at a specific token position.)
3. **Run on adequate hardware** — a GPU with ≥~20 GB VRAM, or a host with
   ≥~24 GB free RAM. Raise the guard and load:
   ```python
   from cnr.real_nla import RealNLAVerbalizer
   av = RealNLAVerbalizer("qwen2.5-7b-L20", load_weights=True,
                          device="cuda", ram_guard_gb=64)
   print(av.verbalize(h_3584).text)   # real NLA explanation
   ```
   (Or, for the SGLang path: serve `kitft/nla-qwen2.5-7b-L20-av` with
   `--disable-radix-cache`, build `inputs_embeds` identically, POST to
   `/generate` with `input_embeds`.)

### Caveat on the cnr science
Even after step 1–3, the cnr eval-awareness *probe / target set A\** is currently
defined on **GPT-2** activations. Moving to Qwen means re-deriving the probe on
Qwen L20 activations. The two changes (real NLA + Qwen base) go together; this
adapter is the NLA half and is ready for that swap.

---

## 5. Files added by this task (no existing file modified)
- `cnr/real_nla.py` — adapter; same `decode(h)/verbalize(h) -> NLADecode`
  interface as the stand-in, real injection wiring, RAM/dim guards, never fakes.
- `scripts/06_real_nla.py` — runs the verifications + side-by-side + report.
- `artifacts/real_nla/released_nla_spec.json` — the registry + template, dumped.
- `artifacts/real_nla/run.json` — full run record.
- `reports/realnla_<ts>.html` — self-contained honest report.
- `REAL_NLA_NOTES.md` — this file.
