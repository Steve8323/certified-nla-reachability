"""Certified NLA reachability.

A prototype mashing together two ideas:

  * Natural Language Autoencoders (Anthropic, transformer-circuits.pub/2026/nla)
    -- used here to *define a target activation set* A* for a concept
    (eval-awareness: "this is a constructed scenario / I'm being tested").

  * Certified defenses against adversarial examples (Raghunathan, Steinhardt,
    Liang, ICLR/NeurIPS 2018) -- their convex-relaxation machinery, repurposed
    to *certify whether A* is reachable* within an epsilon-ball.

The concept target A* is operationalized as the decision region of a small
one-hidden-layer ReLU probe -- deliberately the same architecture class
Raghunathan's 2018 certificate was designed for.

See README.md for the honest "what is real vs approximated" breakdown.
"""

__version__ = "0.1.0"
