# Paper draft: Certified Reachability of NLA-Defined Concepts

LaTeX source for an ~8-page draft on certifying when a model can(not) recognize it
is being tested, by combining Anthropic's Natural Language Autoencoder (NLA) concept
definition with Raghunathan/Steinhardt/Liang certified defenses. All numbers are
grounded in this repo's `README.md`, `reports/`, and `artifacts/` (handoff.json,
meta.json, last_run.json); nothing is fabricated.

## Build

The preamble is standard `\documentclass{article}` with only core LaTeX packages
(amsmath, booktabs, hyperref, xcolor, geometry, ...) — no external `.sty` downloads.

```bash
cd paper
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex   # resolve refs + citations
```

If `pdflatex` is not on your PATH but TeX Live is installed (e.g. macOS TeX Live in
`/usr/local/texlive/2024/bin/universal-darwin`), call it by absolute path or add that
directory to PATH.

Output: `paper/main.pdf`.

## File layout

```
paper/
  main.tex            preamble + \todo macro + \input of sections
  refs.bib            bibliography
  sections/
    abstract.tex
    intro.tex         the question, the two-paper mash-up, contributions
    related.tex       NLA/interp; certified robustness; eval-awareness; reachability != attribution
    method.tex        A* via NLA->classifier->ReLU surrogate; CROWN(+IBP) bound;
                      auto_LiRPA incompatibility -> native bound; differentiable regularizer;
                      GCG inversion; loop-closure
    results.tex       bottom-half + top-half numbers, tables, the four \todo slots
    limitations.tex   PROMINENT, honest
    conclusion.tex
  figs/               (empty; reserved — paper uses text/ASCII pipeline figure, no external images)
```

## The four `\todo` items (in-flight results from parallel agents)

Each appears as a yellow boxed `\todo{...}` in `sections/results.tex`, with the
current placeholder value next to it so the section reads sensibly now:

1. **Harder, non-saturated task with error bars** — replace the saturated AUROC/acc
   `1.00` with the achieved AUROC ± standard error on a genuinely overlapping
   eval-awareness dataset (target AUROC ~0.75–0.92, per `cnr/data_hard.py`).
2. **Deeper-layer certificate + CROWN/IBP ablations** — add a layer sweep and an
   ablation separating native CROWN, IBP, and their min (placeholder: mean certified
   radius 0.29 bottom-half, ~0.13 top-half at layer 6).
3. **Full off-manifold gaming sweep** — replace the 4-point single-anchor sweep +
   8/8 disagreement count with a dense budget grid across all anchors (placeholder:
   crossing at ~20–24σ, P(eval)≈0.04 at the crossing).
4. **Real open-model NLA** — swap the retrieval/eval-axis stand-in verbalizer for a
   trained activation-verbalizer on an open model (placeholder readouts tagged
   `[retrieval-AV stand-in]`).

## Honesty notes

- Bottom-half (direct probe) and top-half (NLA→classifier→surrogate) numbers are
  reported separately and exactly as in the run artifacts.
- The certificate is sound (runtime sampling-witness check); the verbalizer is a
  stand-in; the synthetic task is near-separable, so metrics saturate — all stated
  prominently in the Limitations section.
