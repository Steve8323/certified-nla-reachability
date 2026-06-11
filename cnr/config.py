"""Central configuration. One dataclass, derived paths in __post_init__."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- target model (cached locally; see README) ---
    model_name: str = "gpt2"
    layer: int = 6              # residual-stream layer to read the concept from
    pooling: str = "mean"      # mean-pool over non-pad tokens
    max_len: int = 48
    device: str = "cpu"        # cpu is the most reproducible; "mps"/"cuda" also work

    # --- probe (defines A*) ---
    hidden: int = 64            # ReLU hidden units (Raghunathan one-hidden-layer setting)
    probe_epochs: int = 200
    probe_lr: float = 1e-3

    # --- certification ---
    eps: float = 0.25           # L-inf radius of the activation-space ball, in
                                # units of per-dim activation std (see probe.py)
    norm: str = "inf"          # "inf" or "2"

    # --- data ---
    n_per_class: int = 256
    seed: int = 0

    # --- paths (derived) ---
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    artifacts: Path = field(init=False)
    reports: Path = field(init=False)

    def __post_init__(self) -> None:
        self.artifacts = self.root / "artifacts"
        self.reports = self.root / "reports"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)

    # convenient artifact paths
    @property
    def probe_path(self) -> Path:
        return self.artifacts / "probe.pt"

    @property
    def bank_path(self) -> Path:
        return self.artifacts / "bank.npz"

    @property
    def bank_text_path(self) -> Path:
        return self.artifacts / "bank_texts.json"

    @property
    def meta_path(self) -> Path:
        return self.artifacts / "meta.json"
