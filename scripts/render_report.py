"""Regenerate the HTML report (including the plain-language 'does it work?'
verdict) from the last saved run -- no pipeline recompute. Prints the path.

    python scripts/render_report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnr.config import Config
from cnr.report import build_report


def main():
    cfg = Config()
    f = cfg.artifacts / "last_run.json"
    if not f.exists():
        raise SystemExit("no artifacts/last_run.json yet -- run scripts/run_pipeline.py once first")
    results = json.loads(f.read_text())
    out = build_report(results, cfg.reports)
    print("REPORT:", out)
    return out


if __name__ == "__main__":
    main()
