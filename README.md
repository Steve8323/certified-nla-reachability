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
| certify | native CROWN(+IBP min); at ε=0.25 (·σ) every anchor **certified unreachable**; mean certified radius **0.29** std-units — tight for this ball (within 24–31% of best PGD) but the ball is far inside where real attacks land; see follow-up |
| regularizer | certified radius **0.29 → 0.49** with bank accuracy held at **1.00** |
| invert (GCG) | benign prefix + 8 tokens: score **−8.9 → +4.7**, **crosses τ**; beats random search |
| loop closure | geometry and readout **agree**; 5/5 nearest neighbors are real eval-frame texts |

`11 passed` (pytest). Full dashboard: `reports/report_<timestamp>.html`.

## Honest limitations

- **The certified ball is in the wrong space.** The shallow certificate is
  genuinely *tight for its own threat model* — the certified radius lands within
  24–31% of the best PGD attack radius at all 8 anchors. But that threat model is
  a hidden-space ε·σ box that realistic input-space attacks never need to enter:
  the saved GCG token-space attack lands **≥16×** outside every certified ball,
  ordinary benign prompt-to-prompt variation sits **~13.5×** outside, and the
  deep frozen-attention certificate is up to **~430×** looser than the best attack
  on its own subnetwork. Stated plainly: the certificate certifies a hidden-space
  ball that realistic input-space attacks never need to enter. Tight number, wrong
  question — see the follow-up section below.
- **Scale.** Certification is in activation space through the concept probe, not
  back through all of GPT-2 to the tokens. Full-depth certification through
  attention/LayerNorm is the real research frontier (and why the token side is
  done heuristically). This is the same scale wall we discussed.
- **Separability.** The synthetic eval/deploy task is near-linearly separable, so
  probe accuracy saturates and thresholds are clean. Real eval-awareness is
  subtler; swap in NLA-derived labels to make it honest science.
- **Concept fidelity.** A probe direction ≠ a causal feature. The loop-closure
  agreement check mitigates but does not eliminate this.

## Follow-up experiments (2026-07-24)

A 30-agent audit sprint stress-tested the certificates. Two things came out of it,
and they pull in opposite directions.

**1. The tightness is real, but it's tightness for the wrong threat model.**
Measured against its *own* threat model — a hidden-space ε·σ box — the shallow
certificate is genuinely tight: certified radius within **24–31%** of the best PGD
attack radius at all 8 anchors. That's about as tight as a sound bound gets. The
problem is the box. The attacks that actually matter live far outside it: the saved
GCG token-space attack lands **≥16×** outside every certified ball, ordinary benign
prompt-to-prompt variation sits **~13.5×** outside, and the deep frozen-attention
certificate is up to **~430×** looser than the best attack on its own subnetwork.
The one-line reading: *the certificate certifies a hidden-space ball that realistic
input-space attacks never need to enter.* A "guarantee of unreachability" over a
region nobody was trying to reach is a tight bound and a vacuous safety claim at
the same time.

**2. A noise-adapted certificate that is not vacuous — and is directly useful for
activation-noise probing.** The naive port of the box certificate to Gaussian
activation noise is vacuous (median flip bound **1.0**): a per-dimension box says
nothing about a distribution with unbounded tails. Replacing it with a proper
hidden-space **flip-probability** certificate fixes this. It soundly bounds
P(flip into the eval-awareness region) **≤ 1%** up to a median **σ\*≈0.088** of
activation noise. Cross-checked against sampling: at a typical noise-probing
operating point (**σ=0.3**), **0.54%** of draws actually flip — consistent with the
certificate, and far from the vacuous naive bound.

The method takeaway is the point worth keeping: **flip-probability certificates can
give activation-noise experiments an epistemic backbone.** Before interpreting a
claimed "surfacing" from an activation-noise probe, certify the σ at which the
target region is even *reachable* — below σ\* a claimed surfacing can't be the noise
tripping the concept, and above it you at least know reachability is on the table.

*(These experiments live in a private research archive; the numbers above are the
audited results.)*

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
