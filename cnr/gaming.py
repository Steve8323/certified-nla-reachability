"""ACTIVATION-SPACE gaming experiment: does loop-closure catch off-manifold
surrogate-gaming that the certificate cannot see?

This module turns the project's key qualitative finding into a quantitative
sweep. From >=6 benign deployment anchors h0 we walk displacement delta (in
per-dim activation-std units) along TWO routes that both try to reach the
eval-aware region A* = {h : g(h) >= tau}:

  (A) ON-MANIFOLD   -- interpolate h0 toward realized eval activations (the
      eval-class centroid by default; nearest eval-bank acts as an option).
      This moves along a genuine data direction, so it eventually lands in the
      real eval cluster.

  (B) OFF-MANIFOLD  -- budget-projected gradient ascent on g(h) directly (the
      adversarial route). Steepest ascent on the surrogate finds the cheapest
      g-crossing direction, which runs OFF the activation manifold.

For every route and displacement we record three things:
  * g(h) and whether it crosses tau (the surrogate's verdict);
  * the certificate status at the operating eps (certified_unreachable?), via
    the SAME sound CertifiedVerifier the rest of the project uses;
  * the NLA RE-DECODE -- run the cnr.nla verbalizer + the distilled text
    classifier on h and ask "does the readout STILL read as eval?".

HONESTY / STAND-IN NOTE
-----------------------
The verbalizer (cnr/nla.py) is a transparent STAND-IN decoder, not a trained
Natural Language Autoencoder (see its docstring). Everything else -- the GPT-2
activations, the surrogate g, the sound certificate -- is real computation. The
re-decode readout therefore inherits the stand-in's retrieval+axis signal; we
label it as such and never claim it is a learned NLA.

What the numbers show (and we report them as found, not as wished):
  * Bare g-crossing is CHEAPER off-manifold than on-manifold: gradient ascent
    games the surrogate at a smaller displacement than it takes to interpolate
    into the genuine eval cluster.
  * But among g>=tau points, P(readout==eval) ~ 1 on-manifold and ~ 0
    off-manifold. The off-manifold route crosses g WITHOUT the readout ever
    agreeing -- that mismatch is the gaming, and the loop-closure catches it.
  * "Genuine-crossing" displacement (g>=tau AND readout==eval) is finite and
    small on-manifold but is never reached within the swept budget off-manifold:
    you cannot get to a genuinely-eval-reading state by gaming g.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Work both as `python -m cnr.gaming` and `python cnr/gaming.py`.
if __package__ in (None, ""):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnr.config import Config
from cnr.nla import NLAVerbalizer
from cnr.probe import load_probe
from cnr.verify import CertifiedVerifier


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _g_score(g, h: np.ndarray) -> float:
    with torch.no_grad():
        return float(g(torch.tensor(np.asarray(h, np.float64)[None], dtype=torch.float32))[0])


def _g_scores(g, H: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return g(torch.tensor(np.asarray(H, np.float64), dtype=torch.float32)).numpy()


def _disp_std(h: np.ndarray, h0: np.ndarray, sigma: np.ndarray) -> float:
    """L2 displacement in per-dimension activation-std units."""
    return float(np.linalg.norm((np.asarray(h) - np.asarray(h0)) / sigma))


# --------------------------------------------------------------------------- #
# the readout: NLA re-decode -> text classifier -> "does it read as eval?"
# --------------------------------------------------------------------------- #
class Readout:
    """Loop-closure readout. verbalize(h) -> text -> classifier -> eval? / proba.

    Uses the project's stand-in NLA verbalizer (cnr/nla.py) and the distilled
    text classifier pipeline saved by the top half (artifacts/nla_text_clf.joblib);
    if that artifact is missing it transparently rebuilds the classifier from the
    same corpus the top half uses (deterministic).
    """

    def __init__(self, verb: NLAVerbalizer, pipe, exclude_self: bool = True):
        self.verb = verb
        self.pipe = pipe
        self.exclude_self = exclude_self

    @classmethod
    def from_artifacts(cls, X, texts, y, eval_dir, cfg: Config, k: int = 7,
                       exclude_self: bool = True) -> "Readout":
        verb = NLAVerbalizer(X, texts, y, eval_dir, k=k)
        pipe = _load_or_build_classifier(cfg)
        return cls(verb, pipe, exclude_self=exclude_self)

    def __call__(self, h: np.ndarray) -> tuple[int, float, str]:
        dec = self.verb.decode(np.asarray(h, np.float64), exclude_self=self.exclude_self)
        proba = float(self.pipe.predict_proba([dec.text])[0, 1])
        says_eval = int(proba >= 0.5)
        return says_eval, proba, dec.text


def _load_or_build_classifier(cfg: Config):
    """Load the saved TF-IDF+LR pipeline; rebuild from corpus if absent."""
    clf_path = cfg.artifacts / "nla_text_clf.joblib"
    if clf_path.exists():
        import joblib
        return joblib.load(clf_path)
    # deterministic rebuild (same corpus + seed as the top half)
    from cnr.text_classifier import NLATextClassifier, build_description_corpus
    txt, lab = build_description_corpus(seed=cfg.seed)
    clf = NLATextClassifier(seed=cfg.seed)
    clf.fit(txt, lab)
    return clf.pipe


# --------------------------------------------------------------------------- #
# the two routes (each yields a list of points along the displacement sweep)
# --------------------------------------------------------------------------- #
def on_manifold_targets(X: np.ndarray, y: np.ndarray, h0: np.ndarray,
                        mode: str = "centroid") -> np.ndarray:
    """Pick the on-manifold target h0 moves toward.

    "centroid"     -> the eval-class centroid (dense, robust eval direction).
    "nearest_eval" -> the cosine-nearest realized eval-bank activation.
    Either way the move stays along a genuine data direction (no g gradients).
    """
    ev = np.where(y == 1)[0]
    if mode == "nearest_eval":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        q = np.asarray(h0, np.float64)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = Xn[ev] @ q
        return X[ev[int(np.argmax(sims))]].astype(np.float64)
    return X[ev].mean(0).astype(np.float64)


def sweep_on_manifold(g, h0, target, sigma, deltas: np.ndarray) -> list[np.ndarray]:
    """Interpolate h0 -> target and sample at the requested std-displacements.

    `deltas` are displacement magnitudes in std units. We map each delta to the
    interpolation fraction that realizes it (the path is a straight segment, so
    displacement is linear in the fraction up to ||target-h0||_std).
    """
    h0 = np.asarray(h0, np.float64)
    target = np.asarray(target, np.float64)
    full = _disp_std(target, h0, sigma)            # std-length of the whole segment
    pts = []
    for d in deltas:
        frac = 0.0 if full <= 0 else min(1.0, d / full)
        pts.append((1.0 - frac) * h0 + frac * target)
    return pts


def sweep_off_manifold(g, h0, sigma, deltas: np.ndarray,
                       steps: int = 300, lr: float = 0.1) -> list[np.ndarray]:
    """Budget-projected gradient ascent on g(h), one independent run per budget.

    For each requested displacement budget R (in std units) we Adam-ascend g(h)
    from h0, projecting back into the L2-std ball of radius R every step. This is
    exactly the disciplined adversarial route the top half uses (_opt_surrogate),
    re-implemented here so the experiment is self-contained and the budget grid is
    the displacement axis. Independent runs keep each point a clean "best g
    reachable within this much displacement".
    """
    h0t = torch.tensor(np.asarray(h0, np.float64), dtype=torch.float32)
    sig = torch.tensor(np.asarray(sigma, np.float64), dtype=torch.float32)
    pts = []
    for R in deltas:
        if R <= 0:
            pts.append(np.asarray(h0, np.float64).copy())
            continue
        h = h0t.clone().requires_grad_(True)
        opt = torch.optim.Adam([h], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            s = g(h[None])[0]
            (-s).backward()                         # maximize g
            opt.step()
            with torch.no_grad():
                delta = (h - h0t) / sig
                nrm = float(delta.norm())
                if nrm > R:
                    h.copy_(h0t + (delta * (R / (nrm + 1e-12))) * sig)
        pts.append(h.detach().numpy().astype(np.float64))
    return pts


# --------------------------------------------------------------------------- #
# per-point record + per-route record
# --------------------------------------------------------------------------- #
@dataclass
class PointRecord:
    delta_req: float          # requested displacement budget (std units)
    disp_std: float           # realized displacement (std units)
    g: float                  # surrogate score g(h)
    g_crossed: bool           # g >= tau
    certified_unreachable: bool   # sound certificate at operating eps
    score_upper: float        # sound upper bound U at this point's eps-ball
    readout_eval: int         # NLA re-decode reads as eval? (1/0)
    readout_proba: float      # classifier P(test-aware) of the re-decode
    genuine_cross: bool       # g_crossed AND readout_eval  (a REAL concept-crossing)


@dataclass
class RouteResult:
    name: str                 # "on_manifold" | "off_manifold"
    anchor_id: int
    points: list[PointRecord] = field(default_factory=list)

    # crossing displacements (None -> not reached within the swept budget)
    g_cross_disp: float | None = None        # first delta where g >= tau
    genuine_cross_disp: float | None = None  # first delta where g>=tau AND readout==eval
    readout_flip_disp: float | None = None   # first delta where readout==eval


def _first_disp(points: list[PointRecord], pred) -> float | None:
    for p in points:
        if pred(p):
            return p.disp_std
    return None


# --------------------------------------------------------------------------- #
# run one anchor through both routes
# --------------------------------------------------------------------------- #
def run_anchor(g, verifier: CertifiedVerifier, readout: Readout,
               X, y, h0, anchor_id: int, sigma, tau: float, eps: float,
               deltas: np.ndarray, on_mode: str = "centroid",
               ascent_steps: int = 300, ascent_lr: float = 0.1) -> dict[str, RouteResult]:
    target = on_manifold_targets(X, y, h0, mode=on_mode)
    on_pts = sweep_on_manifold(g, h0, target, sigma, deltas)
    off_pts = sweep_off_manifold(g, h0, sigma, deltas, steps=ascent_steps, lr=ascent_lr)

    out: dict[str, RouteResult] = {}
    for name, pts in (("on_manifold", on_pts), ("off_manifold", off_pts)):
        route = RouteResult(name=name, anchor_id=int(anchor_id))
        for dreq, h in zip(deltas, pts):
            gval = _g_score(g, h)
            crossed = bool(gval >= tau)
            cert = verifier.certify(h, eps, tau)         # sound; raises if unsound
            rev, rp, _txt = readout(h)
            route.points.append(PointRecord(
                delta_req=float(dreq),
                disp_std=_disp_std(h, h0, sigma),
                g=float(gval),
                g_crossed=crossed,
                certified_unreachable=bool(cert.certified_unreachable),
                score_upper=float(cert.score_upper),
                readout_eval=int(rev),
                readout_proba=float(rp),
                genuine_cross=bool(crossed and rev == 1),
            ))
        route.g_cross_disp = _first_disp(route.points, lambda p: p.g_crossed)
        route.readout_flip_disp = _first_disp(route.points, lambda p: p.readout_eval == 1)
        route.genuine_cross_disp = _first_disp(route.points, lambda p: p.genuine_cross)
        out[name] = route
    return out


# --------------------------------------------------------------------------- #
# the full experiment
# --------------------------------------------------------------------------- #
def run_experiment(cfg: Config | None = None, n_anchors: int = 6,
                   delta_max: float = 25.0, n_delta: int = 26,
                   on_mode: str = "centroid", seed: int = 0,
                   ascent_steps: int = 300) -> dict:
    """Sweep both routes from n_anchors benign anchors and aggregate the contrast.

    Returns a JSON-serializable dict with per-anchor route records, the swept
    displacement grid, and the headline aggregates:
      * median g-crossing displacement, on vs off;
      * median genuine-crossing (g>=tau AND readout==eval) displacement, on vs off
        (off is reported as the swept max when never reached -- a sound lower bound
        on the true displacement, which we flag);
      * among g>=tau points, P(readout==eval) on vs off;
      * loop-closure catch rate: fraction of off-manifold g>=tau points the
        re-decode flags as NOT eval.
    """
    cfg = cfg or Config()

    g, meta = load_probe(cfg.artifacts / "nla_surrogate.pt")
    sigma = np.asarray(meta["sigma"], dtype=np.float64)
    tau = float(meta.get("tau", 0.0))

    bank = np.load(cfg.bank_path)
    X = bank["X"].astype(np.float64)
    y = bank["y"].astype(int)
    texts = json.loads(cfg.bank_text_path.read_text())["texts"]
    eval_dir = X[y == 1].mean(0) - X[y == 0].mean(0)

    verifier = CertifiedVerifier(g, sigma, norm="inf", method="auto")
    readout = Readout.from_artifacts(X, texts, y, eval_dir, cfg, k=7, exclude_self=True)

    # benign anchors = clearest ordinary acts (lowest g among deploy class)
    g_all = _g_scores(g, X)
    benign = np.where(y == 0)[0]
    anchors = benign[np.argsort(g_all[benign])][:n_anchors]

    deltas = np.linspace(0.0, float(delta_max), int(n_delta))

    per_anchor = []
    for aid in anchors:
        routes = run_anchor(g, verifier, readout, X, y, X[aid], int(aid),
                            sigma, tau, cfg.eps, deltas, on_mode=on_mode,
                            ascent_steps=ascent_steps)
        per_anchor.append({"anchor_id": int(aid), "g0": float(g_all[aid]),
                           "routes": routes})

    agg = _aggregate(per_anchor, tau, delta_max)
    method_label = verifier.certify(X[anchors[0]], cfg.eps, tau).method

    return {
        "meta": {
            "model": cfg.model_name, "layer": cfg.layer, "d_model": int(X.shape[1]),
            "n_bank": int(X.shape[0]), "n_anchors": int(len(anchors)),
            "tau": tau, "eps": float(cfg.eps), "delta_max": float(delta_max),
            "n_delta": int(n_delta), "on_mode": on_mode,
            "cert_method": method_label,
            "nla_stand_in": True,
            "ascent_steps": int(ascent_steps),
        },
        "deltas": deltas.tolist(),
        "anchors": [int(a) for a in anchors],
        "per_anchor": _serialize_per_anchor(per_anchor),
        "aggregate": agg,
    }


def _aggregate(per_anchor, tau: float, delta_max: float) -> dict:
    """Headline contrast numbers, computed honestly over all anchors/points."""
    def collect(route_name, attr):
        return [pa["routes"][route_name].__getattribute__(attr) for pa in per_anchor]

    def med(vals, fill):
        vv = [v if v is not None else fill for v in vals]
        return float(np.median(vv)) if vv else float("nan")

    # crossing displacements
    on_gcross = collect("on_manifold", "g_cross_disp")
    off_gcross = collect("off_manifold", "g_cross_disp")
    on_genuine = collect("on_manifold", "genuine_cross_disp")
    off_genuine = collect("off_manifold", "genuine_cross_disp")

    # P(readout==eval | g>=tau) and loop-closure catch rate, pooled over points
    def pooled(route_name):
        n_cross = n_cross_eval = n_cross_caught = 0
        for pa in per_anchor:
            for p in pa["routes"][route_name].points:
                if p.g_crossed:
                    n_cross += 1
                    n_cross_eval += int(p.readout_eval == 1)
                    n_cross_caught += int(p.readout_eval == 0)
        p_eval = (n_cross_eval / n_cross) if n_cross else float("nan")
        catch = (n_cross_caught / n_cross) if n_cross else float("nan")
        return n_cross, p_eval, catch

    on_n, on_peval, on_catch = pooled("on_manifold")
    off_n, off_peval, off_catch = pooled("off_manifold")

    # gaming GAP per anchor = (displacement where readout agrees) - (displacement
    # where g first crossed). Small/positive on-manifold (readout barely lags g);
    # never-closes off-manifold (readout never agrees though g crossed).
    def gaps(route_name):
        vals = []
        for pa in per_anchor:
            r = pa["routes"][route_name]
            if r.g_cross_disp is not None and r.readout_flip_disp is not None:
                vals.append(r.readout_flip_disp - r.g_cross_disp)
            elif r.g_cross_disp is not None and r.readout_flip_disp is None:
                vals.append(None)            # g crossed, readout never agreed -> open gap
        return vals
    on_gaps = gaps("on_manifold")
    off_gaps = gaps("off_manifold")
    n_off_open = sum(v is None for v in off_gaps)        # g crossed, readout never flipped
    on_gap_vals = [v for v in on_gaps if v is not None]

    # for "genuine-crossing displacement", off-manifold usually never reaches it;
    # fill un-reached with delta_max (a sound LOWER bound on the true value).
    n_off_genuine_reached = sum(v is not None for v in off_genuine)
    n_on_genuine_reached = sum(v is not None for v in on_genuine)

    return {
        # bare g-crossing (surrogate's own verdict) -- reported as found
        "g_cross_disp_median_on": med(on_gcross, delta_max),
        "g_cross_disp_median_off": med(off_gcross, delta_max),
        "g_cross_disp_on_per_anchor": [None if v is None else round(v, 3) for v in on_gcross],
        "g_cross_disp_off_per_anchor": [None if v is None else round(v, 3) for v in off_gcross],

        # genuine crossing (g>=tau AND readout==eval) -- the meaningful axis
        "genuine_cross_disp_median_on": med(on_genuine, delta_max),
        "genuine_cross_disp_median_off": med(off_genuine, delta_max),
        "genuine_cross_on_per_anchor": [None if v is None else round(v, 3) for v in on_genuine],
        "genuine_cross_off_per_anchor": [None if v is None else round(v, 3) for v in off_genuine],
        "n_on_genuine_reached": int(n_on_genuine_reached),
        "n_off_genuine_reached": int(n_off_genuine_reached),
        "off_genuine_filled_with_delta_max": bool(n_off_genuine_reached < len(off_genuine)),

        # among g>=tau points: does the readout agree it's eval?
        "p_readout_eval_given_gcross_on": on_peval,
        "p_readout_eval_given_gcross_off": off_peval,
        "n_gcross_points_on": int(on_n),
        "n_gcross_points_off": int(off_n),

        # loop-closure catch rate (fraction of g>=tau points flagged NOT eval)
        "catch_rate_on": on_catch,
        "catch_rate_off": off_catch,

        # gaming gap: how far past the g-crossing the readout takes to agree
        "gaming_gap_median_on": float(np.median(on_gap_vals)) if on_gap_vals else float("nan"),
        "gaming_gap_on_per_anchor": [round(v, 3) for v in on_gap_vals],
        "off_anchors_g_crossed_readout_never": int(n_off_open),
        "off_anchors_g_crossed_total": int(sum(1 for pa in per_anchor
                                               if pa["routes"]["off_manifold"].g_cross_disp is not None)),
    }


def _serialize_per_anchor(per_anchor) -> list[dict]:
    out = []
    for pa in per_anchor:
        rec = {"anchor_id": pa["anchor_id"], "g0": pa["g0"], "routes": {}}
        for name, route in pa["routes"].items():
            rec["routes"][name] = {
                "g_cross_disp": route.g_cross_disp,
                "readout_flip_disp": route.readout_flip_disp,
                "genuine_cross_disp": route.genuine_cross_disp,
                "points": [vars(p) for p in route.points],
            }
        out.append(rec)
    return out


def save_results(results: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gaming_results.json"
    path.write_text(json.dumps(results, indent=2))
    return path


if __name__ == "__main__":
    import pprint
    res = run_experiment()
    pprint.pp(res["aggregate"])
