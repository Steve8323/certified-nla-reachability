# Certified reachability of an NLA-defined concept

A working prototype of the design we sketched: use a **Natural Language
Autoencoder**-style concept target together with **Raghunathan-style certified
robustness** to ask *which inputs can drive a model into a given internal state* —
here, the "I'm being evaluated / this is a constructed test" state.

It runs end-to-end on a laptop against a cached **GPT-2** as the stand-in target
model. It is a **methods skeleton**, not a frontier result, and the README is
explicit about which parts are real and which are honest stand-ins.

## The idea in one breath

1. **Define the target set A\*** for a concept (eval-awareness) as the decision
   region of a small **one-hidden-layer ReLU probe** on residual-stream
   activations. That architecture is deliberately the exact class Raghunathan,
   Steinhardt & Liang's 2018 certificate was built for.
2. **Certify reachability.** Around a benign ("deployment") activation `h0`,
   bound the worst-case probe score over a per-dimension ε·σ ball with a *sound*
   convex relaxation (CROWN via `auto_LiRPA`, or a native interval-bound
   fallback). `U < τ` ⇒ a **guarantee** that no perturbation that small can trip
   eval-awareness. Sweep ε to get a **certified radius**.
3. **Invert (discrete side).** A GCG/HotFlip token search finds a real token
   sequence whose activation actually lands in A\*.
4. **Close the loop.** Re-read the found prompt's activation and check the
   geometry (probe ≥ τ) agrees with a verbalized readout — the safeguard against
   an inversion that hits A\* numerically but isn't semantically the concept.

```
 NLA concept  ──►  A* = {h : ReLUProbe(h) ≥ τ}
                        │
   benign h0 ──►  certify:  max_{‖h−h0‖≤ε·σ} Probe(h)  <?  τ      (Raghunathan)
                        │
   token search ──►  real prompt whose activation ∈ A*            (GCG / HotFlip)
                        │
   loop closure ──►  re-verbalize; geometry vs readout agree?
```

## What is real vs. approximated  (read this)

| Component | Status |
|---|---|
| GPT-2 activation extraction | **Real** — cached `gpt2`, residual layer 6, mean-pooled. |
| Eval-awareness probe = A\* | **Real** trained ReLU net. The *labels* come from a synthetic eval-vs-deploy prompt set (a deliberate stand-in for the NLA's signal). |
| Certified bound | **Real and sound** — CROWN if `auto_LiRPA` installs, else native IBP (sound, looser). A sampling witness asserts `U ≥` observed max every call. |
| Differentiable-certificate regularizer | **Real mechanism**, shown as before/after; not a tuned result. |
| Token search / inversion | **Real** discrete optimization against the true model + probe. |
| Verbalizer / loop closure | **Stand-in** — k-NN retrieval over labeled activations, *not* a trained NLA. Every readout is prefixed `[retrieval-AV stand-in]`. |

The one premise correction from the design discussion still holds: this pipeline
answers *"which inference-time inputs reach A\*"*, **not** *"which training
examples instilled it"* — that second question needs influence functions, not a
certificate.

## Results from a clean run

`python scripts/run_pipeline.py` on cached GPT-2 (layer 6), one run:

| stage | result |
|---|---|
| probe (A\*) | train/val acc 1.00 / 1.00; eval vs deploy score means **+8.8 / −10.4** |
| certify | native CROWN(+IBP min); at ε=0.25 (·σ) every anchor **certified unreachable**; mean certified radius **0.29** std-units |
| regularizer | certified radius **0.29 → 0.49** with bank accuracy held at **1.00** |
| invert (GCG) | benign prefix + 8 tokens: score **−8.9 → +4.7**, **crosses τ**; beats random search |
| loop closure | geometry and readout **agree**; 5/5 nearest neighbors are real eval-frame texts |

`11 passed` (pytest). Full dashboard: `reports/report_<timestamp>.html`.

## Honest limitations

- **Scale.** Certification is in activation space through the concept probe, not
  back through all of GPT-2 to the tokens. Full-depth certification through
  attention/LayerNorm is the real research frontier (and why the token side is
  done heuristically). This is the same scale wall we discussed.
- **Separability.** The synthetic eval/deploy task is near-linearly separable, so
  probe accuracy saturates and thresholds are clean. Real eval-awareness is
  subtler; swap in NLA-derived labels to make it honest science.
- **Concept fidelity.** A probe direction ≠ a causal feature. The loop-closure
  agreement check mitigates but does not eliminate this.

## Layout

```
cnr/config.py     Config (model, layer, eps, paths)
cnr/harness.py    ModelHarness — activations + grad-enabled embedding path
cnr/data.py       synthetic eval-vs-deploy prompts
cnr/probe.py      ReLUProbe (= A*), train + folded standardization
cnr/contracts.py  CertResult / CertifiedVerifier / AttackResult / LoopResult
cnr/verify.py     certified reachability (CROWN / IBP)        [agent A]
cnr/regularize.py differentiable-certificate demo            [agent A]
cnr/attack.py     GCG/HotFlip token search                   [agent B]
cnr/verbalize.py  loop-closure verbalizer stand-in           [agent B]
cnr/report.py     timestamped HTML dashboard
scripts/01_build_probe.py   build A*
scripts/run_pipeline.py     end-to-end + report
```

## Run

```bash
cd ~/certified_nla_reachability
python scripts/01_build_probe.py      # builds artifacts/ (done once)
python scripts/run_pipeline.py        # certify + invert + close loop -> reports/*.html
python -m pytest tests -q             # soundness + attack tests
```

## Provenance

- Natural Language Autoencoders — transformer-circuits.pub/2026/nla
- Certified Defenses against Adversarial Examples — Raghunathan, Steinhardt, Liang, ICLR 2018 (arXiv:1801.09344)
- Semidefinite relaxations for certifying robustness — NeurIPS 2018 (arXiv:1811.01057)
