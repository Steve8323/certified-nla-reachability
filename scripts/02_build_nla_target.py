"""Top-half driver: build the classifier-on-NLA-output target, distil the
certifiable surrogate, certify it, run the loop-closure diagnostic, emit the
handoff for the inversion half, and write a timestamped HTML report.

    python scripts/02_build_nla_target.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnr.config import Config
from cnr.nla_target import build
from cnr.report_nla import build_report_nla


def main():
    cfg = Config()
    print("[top-half] building NLA-classifier target on", cfg.model_name, "layer", cfg.layer)
    results = build(cfg)

    nc = results["nla_classify"]
    sg = results["surrogate"]
    ct = results["certificate"]
    print(f"  classifier-on-NLA  : acc {nc['acc_vs_truth']:.3f}  AUROC {nc['auroc']:.3f}")
    print(f"  surrogate fidelity : corr {sg['fidelity_corr_val']:.3f} (val)  "
          f"sign-agree {sg['sign_agree_val']:.3f}")
    print(f"  certificate        : {ct['method']}  mean radius {ct['mean_radius']:.3f}")
    n_cert = sum(a["certified_unreachable"] for a in ct["anchors"])
    print(f"                       {n_cert}/{len(ct['anchors'])} anchors certified unreachable")

    out = build_report_nla(results, cfg.reports)
    print("REPORT:", out)
    return out


if __name__ == "__main__":
    main()
