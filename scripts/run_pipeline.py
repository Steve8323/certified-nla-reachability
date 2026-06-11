"""End-to-end driver: probe (A*) -> certify -> invert (token search) -> close loop.

Each stage is wrapped so a partial environment still yields a report. Stages
that depend on the agent-built modules (cnr.verify, cnr.regularize, cnr.attack,
cnr.verbalize) degrade gracefully if those modules aren't importable yet.
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnr.config import Config
from cnr.harness import ModelHarness
from cnr.probe import load_probe
from cnr.report import build_report


def _note(stage, e):
    print(f"  [skip] {stage}: {type(e).__name__}: {e}")
    return None


def main():
    cfg = Config()
    meta_json = json.loads(cfg.meta_path.read_text())
    probe, pmeta = load_probe(cfg.probe_path)
    sigma = np.asarray(pmeta["sigma"], dtype=np.float32)
    tau = float(pmeta.get("tau", 0.0))
    bank = np.load(cfg.bank_path)
    X, y = bank["X"], bank["y"]
    texts = json.loads(cfg.bank_text_path.read_text())["texts"]

    def score(vecs):
        with torch.no_grad():
            return probe(torch.tensor(np.atleast_2d(vecs), dtype=torch.float32)).numpy()

    results = {"meta": {**meta_json,
                        "score_eval_mean": round(meta_json.get("score_eval_mean", 0), 2),
                        "score_deploy_mean": round(meta_json.get("score_deploy_mean", 0), 2)}}

    # ----- Stage 2: certified reachability ------------------------------------
    print("[stage] certified verifier")
    try:
        from cnr.verify import CertifiedVerifier, certified_radius

        verifier = CertifiedVerifier(probe, sigma, norm=cfg.norm, method="auto")
        grid = np.linspace(0.02, 2.0, 25)
        anchors = []
        method_used = None
        for aid in meta_json["anchor_ids"][:6]:
            h0 = X[aid]
            res = verifier.certify(h0, cfg.eps, tau)
            method_used = res.method
            rad = certified_radius(verifier, h0, tau, grid)
            anchors.append({
                "id": int(aid), "score": float(score(h0)[0]),
                "U": float(res.score_upper),
                "certified_unreachable": bool(res.certified_unreachable),
                "radius": float(rad),
            })
        results["verifier"] = {"method": method_used, "eps_fixed": cfg.eps, "anchors": anchors}
        print(f"  method={method_used}  anchors={len(anchors)}")
    except Exception as e:
        traceback.print_exc()
        results["verifier"] = _note("verifier", e)

    # ----- Stage 2b: differentiable-certificate regularizer demo --------------
    try:
        from cnr.regularize import push_away_demo

        reg = push_away_demo()              # expected to return a dict
        if isinstance(reg, dict):
            results["regularizer"] = reg
            print(f"  regularizer: radius {reg.get('radius_before')} -> {reg.get('radius_after')}")
    except Exception as e:
        _note("regularizer", e)

    # ----- Stage 3: discrete token search (inversion) -------------------------
    print("[stage] token search")
    prefix = "hey can you help me with my"
    try:
        from cnr.attack import GCGAttack

        atk = GCGAttack(harness=ModelHarness(cfg), probe=probe, cfg=cfg)
        ar = atk.search(prefix=prefix, n_adv_tokens=8, tau=tau, steps=30)
        rnd = None
        try:
            rnd = atk.random_baseline(prefix=prefix, n_adv_tokens=8, tau=tau, steps=30).score
        except Exception:
            pass
        results["attack"] = {
            "prefix": prefix, "prompt": ar.prompt, "score": float(ar.score),
            "crossed": bool(ar.crossed), "steps": ar.steps,
            "history": [float(x) for x in ar.history],
            "random_score": None if rnd is None else round(float(rnd), 2),
            "score_start": round(float(ar.history[0]), 2) if ar.history else "?",
        }
        print(f"  found score={ar.score:.2f} crossed={ar.crossed}")
        attack_prompt = ar.prompt
    except Exception as e:
        traceback.print_exc()
        results["attack"] = _note("attack", e)
        attack_prompt = None

    # ----- Stage 4: loop closure ---------------------------------------------
    print("[stage] loop closure")
    try:
        from cnr.verbalize import verbalize

        h_ref = ModelHarness(cfg)
        target = attack_prompt or "[BENCHMARK] Question 7/100. Choose the most ethical action."
        lp = verbalize(target, h_ref, probe, X, texts, y, tau, k=5)
        results["loop"] = {
            "prompt": lp.prompt, "probe_score": float(lp.probe_score),
            "crossed": bool(lp.crossed), "readout": lp.verbalizer_readout,
            "readout_says_eval": bool(lp.readout_says_eval), "agree": bool(lp.agree),
        }
        print(f"  agree={lp.agree}")
    except Exception as e:
        traceback.print_exc()
        results["loop"] = _note("loop", e)

    (cfg.artifacts / "last_run.json").write_text(json.dumps(results, indent=2, default=str))
    out = build_report(results, cfg.reports)
    print("REPORT:", out)
    return out


if __name__ == "__main__":
    main()
