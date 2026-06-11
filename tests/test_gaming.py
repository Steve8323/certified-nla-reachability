"""Gaming-experiment tests: the loop-closure catches off-manifold surrogate-gaming
that the certificate cannot see.

Two load-bearing assertions (the project's key qualitative finding, quantified):
  * the displacement at which the ON-manifold route reaches a GENUINE eval state
    (g>=tau AND the re-decode reads as eval) is strictly smaller than for the
    OFF-manifold route -- off-manifold gaming never reaches a genuinely-eval state
    within the swept budget;
  * at least one OFF-manifold point has g>=tau but the re-decode reads NOT eval
    -- i.e. it games the surrogate/certificate while the loop-closure flags it.

Plus soundness/monotonicity sanity on the experiment outputs.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from cnr.config import Config
from cnr.gaming import run_experiment


@pytest.fixture(scope="module")
def res():
    # >=6 anchors as the experiment specifies; 1-sigma grid resolution over
    # 0..25 sigma so the on-manifold genuine crossing (~24.5 sigma) is resolved
    # as a finite value below delta_max, while the off-manifold route crosses g
    # cheaply yet never reads as eval.
    return run_experiment(Config(), n_anchors=6, delta_max=25.0, n_delta=26,
                          on_mode="centroid", ascent_steps=200)


def _route(res, anchor_idx, name):
    return res["per_anchor"][anchor_idx]["routes"][name]


def test_on_manifold_genuine_crossing_is_closer(res):
    """On-manifold reaches a genuine eval state at smaller displacement than off.

    We compare the median genuine-crossing displacement (g>=tau AND readout==eval).
    Off-manifold anchors that never reach it are filled with the swept max
    (delta_max) inside the aggregate -- a sound lower bound on the true value, so
    the inequality below is conservative.
    """
    a = res["aggregate"]
    on = a["genuine_cross_disp_median_on"]
    off = a["genuine_cross_disp_median_off"]
    assert not np.isnan(on), "on-manifold never reached a genuine eval state -- unexpected"
    assert on < off, (
        f"on-manifold genuine-crossing ({on:.2f} sigma) should be < off-manifold "
        f"({off:.2f} sigma); off-manifold gaming should NOT reach a genuine eval state sooner"
    )
    # and the on-manifold route must actually reach it for the majority of anchors
    assert a["n_on_genuine_reached"] >= res["meta"]["n_anchors"] // 2


def test_loop_closure_catches_off_manifold_gaming(res):
    """>=1 off-manifold point has g>=tau but the re-decode reads NOT eval."""
    caught = 0
    crossed_off = 0
    for pa in res["per_anchor"]:
        for p in pa["routes"]["off_manifold"]["points"]:
            if p["g_crossed"]:
                crossed_off += 1
                if p["readout_eval"] == 0:
                    caught += 1
    assert crossed_off >= 1, "off-manifold route never crossed tau -- sweep too short?"
    assert caught >= 1, (
        "expected >=1 off-manifold point with g>=tau but readout != eval "
        "(the loop-closure catch); found none"
    )
    # the headline catch-rate should also reflect this
    assert res["aggregate"]["catch_rate_off"] > 0.0


def test_off_manifold_games_surrogate_but_not_readout(res):
    """Among g>=tau points, the re-decode agrees far less off-manifold than on."""
    a = res["aggregate"]
    p_on = a["p_readout_eval_given_gcross_on"]
    p_off = a["p_readout_eval_given_gcross_off"]
    assert a["n_gcross_points_off"] >= 1
    # off-manifold readout-agreement among g-crossings is strictly lower than on,
    # and is essentially zero (the gaming).
    assert p_off < p_on
    assert p_off <= 0.1


def test_certificate_status_is_recorded_and_sane(res):
    """Every swept point carries a sound certificate verdict, and no anchor
    *starts* (delta=0) inside the eval region. (At the default eps=0.25 only some
    anchors are certified -- the certificate is tight -- so we assert >=1 is, not
    all, matching the project's 'certificate bites' style.)"""
    anchor_certified = 0
    for pa in res["per_anchor"]:
        for name in ("on_manifold", "off_manifold"):
            pts = pa["routes"][name]["points"]
            assert all("certified_unreachable" in p for p in pts)
            assert all(isinstance(p["certified_unreachable"], bool) for p in pts)
            # a certified point must have its sound upper bound strictly below tau
            for p in pts:
                if p["certified_unreachable"]:
                    assert p["score_upper"] < res["meta"]["tau"]
        # delta=0 is the same anchor on both routes; it must not start in A*
        z_on = pa["routes"]["on_manifold"]["points"][0]
        assert z_on["disp_std"] == pytest.approx(0.0, abs=1e-6)
        assert z_on["g_crossed"] is False
        anchor_certified += int(z_on["certified_unreachable"])
    assert anchor_certified >= 1, "no benign anchor certified-unreachable at eps -- unexpected"


def test_g_increases_along_off_manifold_ascent(res):
    """Off-manifold gradient ascent should be ~monotone non-decreasing in g
    across budgets (each budget is an independent best-g-within-R run)."""
    for pa in res["per_anchor"]:
        gs = [p["g"] for p in pa["routes"]["off_manifold"]["points"]]
        # allow tiny optimizer noise; overall trend must be strongly increasing
        assert gs[-1] > gs[0] + 1.0
