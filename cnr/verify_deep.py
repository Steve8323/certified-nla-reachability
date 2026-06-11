"""DEEPER certified eval-awareness reachability: certify an eps-ball one GPT-2
transformer block BEFORE the probe, propagated through REAL nonlinearities.

WHY
---
`cnr/verify.py` certifies in *activation space*: it bounds  max_{h in B} probe(h)
for an eps-ball B placed directly on the layer-`L` pooled activation that the
probe reads. That is a guarantee about perturbations of the probe's *input*, but
it says nothing about whether such an activation is reachable by perturbing the
text -- the perturbation lives one nonlinearity away from the tokens.

This module pushes the certificate one transformer block deeper. The probe reads
`hidden_states[L]` (= the residual stream *after* block L-1). We place the
eps-ball at `hidden_states[L-1]` (the residual stream *entering* block L-1) and
propagate it, with SOUND bounds, through the block's real nonlinearities up to
the probe. Concretely the covered computation is

    a   = h + attn_anchor                  (residual after attention; see SCOPE)
    z   = LayerNorm_2(a)                    (real LayerNorm, sound interval/linear)
    f   = z @ W_fc^T + b_fc                 (MLP up-projection, 768 -> 3072)
    g   = gelu_new(f)                       (real GELU, sound linear/interval envelope)
    p   = g @ W_proj^T + b_proj             (MLP down-projection, 3072 -> 768)
    h6  = a + p                             (residual after MLP) = hidden_states[L]
    s   = probe( mean_pool_t h6[t] )        (the eval-awareness logit)

so the bound covers FOUR genuine nonlinearities in series -- LayerNorm, two
Linears, GELU, and the probe's own ReLU -- versus one (the probe ReLU) in the
shallow certificate. That is the point of the exercise: deeper == closer to the
tokens, and (as the numbers show) strictly harder to certify.

SCOPE / HONESTY  (read this -- it is the whole contract)
--------------------------------------------------------
Full sound CROWN through self-attention + LayerNorm is intractable here
(attention couples all token positions through a softmax). We therefore certify a
clearly-delimited SOUND SUBNETWORK and label it as such everywhere:

    "frozen-attention partial certificate"
        The attention sublayer output is FROZEN to its value at the anchor
        prompt:  attn_anchor = attn(LN_1(h0)).  Inside the ball we then treat
        a = h + attn_anchor, i.e. attention does NOT react to the perturbation.

    This is a SOUND certificate *for the covered subnetwork* -- every bound below
    is a true upper bound on the score of the frozen-attention network over the
    ball, and a sampling witness re-runs exactly that frozen-attention network and
    checks U >= max(samples). It is NOT a certificate for the full model: a real
    text perturbation also moves the attention output, which we do not bound. We
    never claim otherwise. The residual + MLP path that we DO bound is the
    block's nonlinear core (LayerNorm + 2 Linears + GELU), so this is a genuine
    multi-nonlinearity sound certificate, just over a named subnetwork.

The eps-ball is per-dimension L-inf with radius eps * sigma_i (sigma from the
probe meta), matching `cnr/verify.py` exactly. By default we perturb every real
token position; `token_index` restricts the perturbation to a single position
(the rest of `h` frozen at the anchor), which models "what the next/this token
alone can do" and is where a positive deep radius actually exists.

SOUND PRIMITIVES
----------------
* Linear  -> reuse `ibp_bounds_linear` from cnr.verify (GPT-2 Conv1D weight is
  (in,out); we transpose to the (out,in) convention those helpers expect).
* GELU (gelu_new, the tanh approximation GPT-2 uses) -> `gelu_new` is C-infinity
  with a SINGLE global minimum at x* ~ -0.75245 (decreasing for x<x*, increasing
  after), so its exact min/max over any interval is closed-form (interval bound),
  and a sound *linear* envelope is the endpoint secant lifted/dropped by the
  grid-sampled gap PLUS a provable Lipschitz slack ((Lip + |slope|) * grid
  spacing, Lip(gelu_new) <= 1.13) so the discrete grid provably dominates the
  continuum. Verified sound on dense grids in tests.
* LayerNorm -> centering x - mean is an exact linear map; the normalizer
  r = 1/sqrt(var+eps) is bounded by a sound interval [r_lo, r_hi] from a
  straddle-aware lower bound on var. Interval mode multiplies the two sound
  intervals; linear (CROWN-backward) mode keeps the linear centering term with
  the mid normalizer r0 and pushes the |r-r0| uncertainty into the bias as a
  sound symmetric slack.

Two bound methods, both sound; we return min where both apply (neither dominates):
    "ibp"        forward interval propagation through the whole chain.
    "crown"      a backward CROWN pass (linear GELU + linear LayerNorm + the
                 probe-ReLU relaxation from verify.py), which keeps cross-neuron
                 correlations and is ~2x tighter here; reported as min(CROWN,IBP).

As in verify.py, every certify() draws >=2000 samples inside the ball, runs the
EXACT frozen-attention forward on them, and asserts U >= max(samples) - 1e-4. A
violation raises -- we never return a value we cannot defend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

if __package__ in (None, ""):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cnr.verify import ibp_bounds_linear
else:
    from .verify import ibp_bounds_linear


# --------------------------------------------------------------------------- #
# GELU (gelu_new == GPT-2's tanh-approx GELU) and its sound relaxations
# --------------------------------------------------------------------------- #
_GELU_C = math.sqrt(2.0 / math.pi)
# x* and g(x*): the single global minimum of gelu_new (found numerically once;
# decreasing for x<x*, increasing for x>x*). Used for the exact interval min.
_GELU_XSTAR = -0.7524499805
# Sound global Lipschitz bound for gelu_new (sup|g'| ~ 1.1290 -> round up).
_GELU_LIP = 1.13


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GPT-2's `gelu_new` activation (tanh approximation of GELU)."""
    return 0.5 * x * (1.0 + torch.tanh(_GELU_C * (x + 0.044715 * x * x * x)))


def _gelu_star_val(dtype) -> torch.Tensor:
    return gelu_new(torch.tensor(_GELU_XSTAR, dtype=dtype))


def gelu_interval(zl: torch.Tensor, zu: torch.Tensor):
    """Sound EXACT min/max of gelu_new over each interval [zl, zu].

    gelu_new decreases on (-inf, x*) and increases on (x*, inf), so:
        max = max(g(zl), g(zu))                       (extremes at endpoints)
        min = g(x*) if x* in [zl,zu] else min(g(zl), g(zu))
    This is the tightest possible interval (it is the true range), hence sound.
    """
    gl = gelu_new(zl)
    gu = gelu_new(zu)
    gmax = torch.maximum(gl, gu)
    contains_star = (zl <= _GELU_XSTAR) & (zu >= _GELU_XSTAR)
    gstar = _gelu_star_val(zl.dtype).to(zl)
    gmin = torch.where(
        contains_star,
        torch.minimum(torch.minimum(gl, gu), gstar.expand_as(gl)),
        torch.minimum(gl, gu),
    )
    return gmin, gmax


def gelu_linear_relax(l: torch.Tensor, u: torch.Tensor, ngrid: int = 96):
    """Sound linear envelope of gelu_new on [l,u]: a_l x + b_l <= g(x) <= a_u x + b_u.

    Both lines use the endpoint secant slope a_sec. The secant is lifted (for the
    upper line) / dropped (for the lower line) by the maximum gap to gelu_new
    measured on an `ngrid` grid PLUS a provable Lipschitz slack so the discrete
    grid provably bounds the continuum:  gap(x) = g(x) - secant(x) is Lipschitz
    with constant (Lip(g) + |a_sec|), so between adjacent grid points (spacing
    (u-l)/(ngrid-1)) it cannot exceed the grid max by more than that constant
    times the spacing. Returns (a_l, b_l, a_u, b_u), same shape as l/u.
    """
    width = (u - l).clamp(min=0.0)
    gl = gelu_new(l)
    gu = gelu_new(u)
    safe = width > 1e-12
    a_sec = torch.where(safe, (gu - gl) / width.clamp(min=1e-12), torch.zeros_like(l))
    b_sec = gl - a_sec * l

    ts = torch.linspace(0.0, 1.0, ngrid, dtype=l.dtype).view(-1, *([1] * l.dim()))
    xs = l.unsqueeze(0) + ts * width.unsqueeze(0)
    gs = gelu_new(xs)
    sec = a_sec.unsqueeze(0) * xs + b_sec.unsqueeze(0)
    diff = gs - sec
    spacing = width / (ngrid - 1)
    slack = (_GELU_LIP + a_sec.abs()) * spacing

    gap_up = diff.amax(0).clamp(min=0.0) + slack          # provable max of (g - sec)
    gap_lo = (-diff).amax(0).clamp(min=0.0) + slack       # provable max of (sec - g)
    a_u = a_sec
    b_u = b_sec + gap_up
    a_l = a_sec
    b_l = b_sec - gap_lo
    return a_l, b_l, a_u, b_u


# --------------------------------------------------------------------------- #
# LayerNorm sound bounds
# --------------------------------------------------------------------------- #
def _layernorm_var_interval(l: torch.Tensor, u: torch.Tensor, eps: float):
    """Sound bounds for a LayerNorm over the last dim given x in [l, u] (per row).

    Returns (n_l, n_u, r_lo, r_hi, nmax) where:
        n_l, n_u : sound interval of the centered value (x_i - mean), shape (...,D)
        r_lo,r_hi: sound interval of the normalizer r = 1/sqrt(var+eps), shape (...,1)
        nmax     : max(|n_l|, |n_u|), shape (...,D)
    var lower bound uses a straddle-aware per-coordinate square lower bound
    ((x_j-mean)^2 >= 0 if the centered interval straddles 0, else the nearer
    endpoint squared) averaged over D; that is a sound lower bound on var, and
    1/sqrt is decreasing so it gives the sound r upper bound (and vice versa).
    """
    D = l.shape[-1]
    mean_l = l.mean(-1, keepdim=True)
    mean_u = u.mean(-1, keepdim=True)
    n_l = l - mean_u                         # sound lower of (x_i - mean)
    n_u = u - mean_l                         # sound upper of (x_i - mean)
    straddle = (n_l <= 0) & (n_u >= 0)
    sq_lb = torch.where(straddle, torch.zeros_like(n_l), torch.minimum(n_l * n_l, n_u * n_u))
    sq_ub = torch.maximum(n_l * n_l, n_u * n_u)
    var_lb = sq_lb.mean(-1, keepdim=True)
    var_ub = sq_ub.mean(-1, keepdim=True)
    r_lo = 1.0 / torch.sqrt(var_ub + eps)
    r_hi = 1.0 / torch.sqrt(var_lb + eps)
    nmax = torch.maximum(n_l.abs(), n_u.abs())
    return n_l, n_u, r_lo, r_hi, nmax


def layernorm_interval(l, u, gamma, beta, eps):
    """Sound interval [y_lo, y_hi] of LayerNorm(x)*gamma+beta over x in [l,u].

    y_i = gamma_i * r * (x_i - mean) + beta_i, with r in [r_lo,r_hi] (>0) and
    (x_i-mean) in [n_l_i, n_u_i]. We interval-multiply the two sound intervals
    (4-corner min/max) then apply gamma (flipping for gamma_i<0) and add beta.
    """
    n_l, n_u, r_lo, r_hi, _ = _layernorm_var_interval(l, u, eps)
    c1 = n_l * r_lo
    c2 = n_l * r_hi
    c3 = n_u * r_lo
    c4 = n_u * r_hi
    prod_lo = torch.minimum(torch.minimum(c1, c2), torch.minimum(c3, c4))
    prod_hi = torch.maximum(torch.maximum(c1, c2), torch.maximum(c3, c4))
    gpos = gamma >= 0
    y_lo = torch.where(gpos, prod_lo * gamma + beta, prod_hi * gamma + beta)
    y_hi = torch.where(gpos, prod_hi * gamma + beta, prod_lo * gamma + beta)
    return y_lo, y_hi


def layernorm_backward(lam, l, u, gamma, beta, eps):
    """Backward CROWN step through LayerNorm for an UPPER bound.

    Given a backward coefficient `lam` (..., D) onto the LayerNorm output y, return
        A_x   : (..., D) linear coefficient onto the LayerNorm INPUT x
        const : (...,)   constant contribution
    such that   sum_i lam_i y_i  <=  A_x . x + const  for all x in [l, u].

    Decomposition  y_i = gamma_i * r * (x_i - mean) + beta_i  with r = r0 +/- dr:
        linear part  gamma_i * r0 * (x_i - mean)  -> exact linear in x
            (centering: d(x_i-mean)/dx_j = e_ij - 1/D, so summing over i gives
             A_x[j] = coef[j] - mean(coef), coef_i = lam_i * gamma_i * r0)
        bias part    lam_i * beta_i
        uncertainty  lam_i * gamma_i * (r-r0) * (x_i-mean), |.| <= |lam_i gamma_i| dr nmax_i
            -> added to const as the worst-case POSITIVE slack (sound for an upper
               bound; the true term is <= this slack everywhere on the box).
    """
    D = l.shape[-1]
    n_l, n_u, r_lo, r_hi, nmax = _layernorm_var_interval(l, u, eps)
    r0 = 0.5 * (r_lo + r_hi)
    dr = 0.5 * (r_hi - r_lo)
    coef = lam * (gamma * r0)                       # (..., D)
    A_x = coef - coef.sum(-1, keepdim=True) / D     # centering Jacobian applied
    const = (lam * beta).sum(-1)                    # bias term
    slack = (lam.abs() * gamma.abs() * dr * nmax).sum(-1)   # r-uncertainty, >= true
    return A_x, const + slack


# --------------------------------------------------------------------------- #
# Block parameter container (frozen attention anchor + MLP/LN weights)
# --------------------------------------------------------------------------- #
@dataclass
class DeepCertResult:
    """Result of a deep (frozen-attention) certificate. SOUND for the covered
    subnetwork only -- see module docstring SCOPE."""
    certified_unreachable: bool
    score_upper: float
    score_at_anchor: float
    tau: float
    eps: float
    layer: int
    method: str                 # "crown" | "ibp"
    scope: str                  # human-readable scope label (frozen attention etc.)
    sampled_lb: float           # max over witness samples of the EXACT frozen net
    n_witness: int
    token_index: int            # -1 == all real tokens perturbed
    details: dict = field(default_factory=dict)


class DeepFrozenAttnCertifier:
    """Sound deep certifier through block (layer-1) -> probe, frozen attention.

    Construction extracts the static pieces (MLP/LN weights, probe weights) once.
    `certify(...)` needs the per-anchor tensors (h_{L-1} for one prompt, the
    attention anchor, the pooling weights); use `prepare_anchor` to compute them
    from a harness + prompt, or pass them directly for testing.
    """

    def __init__(self, block, probe, sigma, layer: int, n_witness: int = 2000):
        self.layer = int(layer)
        self.n_witness = int(n_witness)
        self.sigma = np.asarray(sigma, dtype=np.float64)

        # LayerNorm 2 (pre-MLP)
        self.gamma2 = block.ln_2.weight.detach().to(torch.float64)
        self.beta2 = block.ln_2.bias.detach().to(torch.float64)
        self.ln_eps = float(block.ln_2.eps)
        # MLP: GPT-2 Conv1D weight is (in, out) -> transpose to (out, in)
        self.Wfc = block.mlp.c_fc.weight.detach().to(torch.float64).T.contiguous()
        self.bfc = block.mlp.c_fc.bias.detach().to(torch.float64)
        self.Wpr = block.mlp.c_proj.weight.detach().to(torch.float64).T.contiguous()
        self.bpr = block.mlp.c_proj.bias.detach().to(torch.float64)
        # probe Linear->ReLU->Linear
        self.W1 = probe.net[0].weight.detach().to(torch.float64)
        self.b1 = probe.net[0].bias.detach().to(torch.float64)
        self.W2 = probe.net[2].weight.detach().to(torch.float64)
        self.b2 = probe.net[2].bias.detach().to(torch.float64)
        self.D = self.W1.shape[1]
        self._probe = probe                          # for the witness forward

    # ---- exact frozen-attention forward (probe score) for the witness ----
    def _frozen_forward_batch(self, h_pert, attn_anchor, w):
        """Exact frozen-attention subnetwork scores for a BATCH of perturbed inputs.

        h_pert: (N, T, D); attn_anchor: (T, D); w: (T,). Returns (N,) probe scores.
        Runs the REAL ln_2 / c_fc / gelu_new / c_proj (float64) then the probe, fully
        vectorised over the N samples (so the >=2000-sample witness is cheap).
        """
        a = h_pert + attn_anchor                                   # (N,T,D)
        mean = a.mean(-1, keepdim=True)
        var = ((a - mean) ** 2).mean(-1, keepdim=True)
        z = (a - mean) / torch.sqrt(var + self.ln_eps) * self.gamma2 + self.beta2
        f = z @ self.Wfc.T + self.bfc                              # (N,T,3072)
        g = gelu_new(f)
        p = g @ self.Wpr.T + self.bpr                              # (N,T,D)
        h6 = a + p
        pooled = (h6 * w[:, None]).sum(1)                          # (N,D)
        zp = torch.relu(pooled @ self.W1.T + self.b1)
        return (zp @ self.W2.T + self.b2).squeeze(-1)             # (N,)

    def _frozen_forward(self, h_pert, attn_anchor, w):
        """Scalar score of the frozen-attention subnetwork on one h_{L-1} (T,D)."""
        return float(self._frozen_forward_batch(
            h_pert.unsqueeze(0), attn_anchor, w).item())

    def _input_box(self, h, eps, token_index):
        """Per-dim L-inf box on h_{L-1}; only `token_index` perturbed if >=0."""
        eps_vec = torch.tensor(eps * self.sigma, dtype=torch.float64)
        xl = h.clone()
        xu = h.clone()
        if token_index is None or token_index < 0:
            T = h.shape[0]
            xl = h - eps_vec
            xu = h + eps_vec
            # but only real tokens matter; caller passes h already trimmed if needed
            _ = T
        else:
            xl[token_index] = h[token_index] - eps_vec
            xu[token_index] = h[token_index] + eps_vec
        return xl, xu, eps_vec

    # ---- forward interval bound through the whole chain ----
    def _ibp_upper(self, h, attn_anchor, w, eps, token_index):
        xl, xu, _ = self._input_box(h, eps, token_index)
        a_l = xl + attn_anchor
        a_u = xu + attn_anchor
        z_lo, z_hi = layernorm_interval(a_l, a_u, self.gamma2, self.beta2, self.ln_eps)
        f_lo, f_hi = ibp_bounds_linear(self.Wfc, self.bfc, z_lo, z_hi)
        g_lo, g_hi = gelu_interval(f_lo, f_hi)
        p_lo, p_hi = ibp_bounds_linear(self.Wpr, self.bpr, g_lo, g_hi)
        h6_lo = a_l + p_lo
        h6_hi = a_u + p_hi
        pooled_lo = (h6_lo * w[:, None]).sum(0)
        pooled_hi = (h6_hi * w[:, None]).sum(0)
        # probe IBP upper over the pooled box
        zl, zu = ibp_bounds_linear(self.W1, self.b1, pooled_lo, pooled_hi)
        al, au = torch.relu(zl), torch.relu(zu)
        _, U = ibp_bounds_linear(self.W2, self.b2, al, au)
        return float(U.squeeze(-1).item()), (pooled_lo, pooled_hi, f_lo, f_hi)

    # ---- backward CROWN bound through the whole chain ----
    def _crown_upper(self, h, attn_anchor, w, eps, token_index):
        xl, xu, _ = self._input_box(h, eps, token_index)
        a_l = xl + attn_anchor
        a_u = xu + attn_anchor
        # pre-activation boxes (cheap interval pre-bounds drive the relaxations)
        z_lo, z_hi = layernorm_interval(a_l, a_u, self.gamma2, self.beta2, self.ln_eps)
        f_lo, f_hi = ibp_bounds_linear(self.Wfc, self.bfc, z_lo, z_hi)
        g_lo0, g_hi0 = gelu_interval(f_lo, f_hi)
        p_lo, p_hi = ibp_bounds_linear(self.Wpr, self.bpr, g_lo0, g_hi0)
        pooled_lo = ((a_l + p_lo) * w[:, None]).sum(0)
        pooled_hi = ((a_u + p_hi) * w[:, None]).sum(0)

        # --- probe ReLU CROWN relaxation (same algorithm as verify.crown_upper) ---
        Wp1 = self.W1.clamp(min=0)
        Wn1 = self.W1.clamp(max=0)
        zp_lo = pooled_lo @ Wp1.T + pooled_hi @ Wn1.T + self.b1
        zp_hi = pooled_hi @ Wp1.T + pooled_lo @ Wn1.T + self.b1
        active = zp_lo >= 0
        inactive = zp_hi <= 0
        unstable = ~(active | inactive)
        denom = (zp_hi - zp_lo).clamp(min=1e-12)
        slope = zp_hi / denom
        alpha_u = torch.where(active, torch.ones_like(zp_lo),
                  torch.where(inactive, torch.zeros_like(zp_lo), slope))
        beta_u = torch.where(unstable, -slope * zp_lo, torch.zeros_like(zp_lo))
        a_low = (slope >= 0.5).to(zp_lo.dtype)
        alpha_l = torch.where(active, torch.ones_like(zp_lo),
                  torch.where(inactive, torch.zeros_like(zp_lo), a_low))
        lam = self.W2.squeeze(0)                      # (hidden,)
        pos = lam >= 0
        alpha = torch.where(pos, alpha_u, alpha_l)
        beta_relu = torch.where(pos, beta_u, torch.zeros_like(beta_u))
        lam_z = lam * alpha
        A_pooled = lam_z @ self.W1                    # (D,) coeff onto pooled h6
        const = (lam * beta_relu).sum() + self.b2.sum() + (lam_z * self.b1).sum()

        # pooled = sum_t w_t h6[t]; coeff onto h6[t] = w_t * A_pooled
        A_h6 = w[:, None] * A_pooled[None, :]         # (T, D)
        # h6 = a + p  -> residual branch contributes A_h6 directly onto a
        A_a = A_h6.clone()
        # p = g @ Wpr^T + bpr
        const = const + (A_h6 * self.bpr[None, :]).sum()
        A_g = A_h6 @ self.Wpr                         # (T, 3072) coeff onto gelu out
        # GELU linear relaxation on [f_lo, f_hi]
        al_g, bl_g, au_g, bu_g = gelu_linear_relax(f_lo, f_hi)
        posg = A_g >= 0
        a_sel = torch.where(posg, au_g, al_g)
        b_sel = torch.where(posg, bu_g, bl_g)
        const = const + (A_g * b_sel).sum()
        A_f = A_g * a_sel                             # (T, 3072) coeff onto fc out
        # f = z @ Wfc^T + bfc
        const = const + (A_f * self.bfc[None, :]).sum()
        A_z = A_f @ self.Wfc                          # (T, D) coeff onto ln2 out
        # LayerNorm backward -> coeff onto a
        A_a_ln, const_ln = layernorm_backward(A_z, a_l, a_u, self.gamma2, self.beta2, self.ln_eps)
        const = const + const_ln.sum()
        A_a = A_a + A_a_ln                            # total coeff onto a
        # a = h + attn_anchor (linear); coeff onto h is A_a
        const = const + (A_a * attn_anchor).sum()
        A_x = A_a
        # maximize A_x . h over the box [xl, xu]
        Ap = A_x.clamp(min=0)
        An = A_x.clamp(max=0)
        U = (Ap * xu + An * xl).sum() + const
        return float(U.item())

    # ---- public certify ----
    def certify(self, h, attn_anchor, w, eps, tau, method: str = "crown",
                token_index: int = -1, witness_seed: int = 0) -> DeepCertResult:
        """Sound deep certificate. h, attn_anchor: (T,D) float64; w: (T,) sums to 1.

        method "crown" returns min(CROWN, IBP); "ibp" returns the interval bound.
        token_index >= 0 perturbs only that token; -1 perturbs all rows of h.
        Raises AssertionError if the bound fails the >=n_witness sampling witness.
        """
        h = torch.as_tensor(h, dtype=torch.float64)
        attn_anchor = torch.as_tensor(attn_anchor, dtype=torch.float64)
        w = torch.as_tensor(w, dtype=torch.float64)
        ti = int(token_index)

        U_ibp, _ = self._ibp_upper(h, attn_anchor, w, eps, ti)
        if method == "ibp":
            U = U_ibp
            used = "ibp"
        else:
            U_crown = self._crown_upper(h, attn_anchor, w, eps, ti)
            U = min(U_crown, U_ibp)              # min of two sound bounds, still sound
            used = "crown"

        # anchor score (frozen forward at the unperturbed h)
        s0 = self._frozen_forward(h, attn_anchor, w)

        # ---- sampling soundness witness: EXACT frozen forward, U >= max ----
        # Draws n_witness (>=2000) uniform points in the ball, runs the EXACT
        # frozen-attention subnetwork on all of them at once, and requires U to
        # dominate their max. Chunked so memory stays bounded for all-token balls.
        eps_vec = torch.tensor(eps * self.sigma, dtype=torch.float64)
        T = h.shape[0]
        gen = torch.Generator().manual_seed(int(witness_seed))
        sampled_lb = s0                                    # anchor included
        n = self.n_witness
        chunk = 256
        drawn = 0
        while drawn < n:
            b = min(chunk, n - drawn)
            noise = (2 * torch.rand(b, T, self.D, generator=gen, dtype=torch.float64) - 1) * eps_vec
            h_p = h.unsqueeze(0).expand(b, T, self.D).clone()
            if ti is None or ti < 0:
                h_p = h_p + noise
            else:
                h_p[:, ti, :] = h[ti] + noise[:, ti, :]
            scores = self._frozen_forward_batch(h_p, attn_anchor, w)
            sampled_lb = max(sampled_lb, float(scores.max().item()))
            drawn += b

        if not (U >= sampled_lb - 1e-4):
            raise AssertionError(
                f"DEEP SOUNDNESS VIOLATION [{used}]: U={U:.6f} < sampled frozen max "
                f"{sampled_lb:.6f} (eps={eps}, layer={self.layer}, token={ti}). The "
                f"deep verifier returned a value that is NOT an upper bound -- bug."
            )

        certified = bool(U < tau)
        scope = ("frozen-attention partial certificate "
                 "(sound for residual+LayerNorm+MLP(GELU)+probe subnetwork; "
                 "attention frozen to anchor)")
        details = {
            "margin_to_tau": float(tau - U),
            "U_ibp": float(U_ibp),
            "n_witness": self.n_witness,
            "eps_vec_l2": float(np.linalg.norm(eps * self.sigma)),
        }
        if method != "ibp":
            details["U_crown"] = float(U_crown)
        return DeepCertResult(
            certified_unreachable=certified,
            score_upper=float(U),
            score_at_anchor=float(s0),
            tau=float(tau),
            eps=float(eps),
            layer=self.layer,
            method=used,
            scope=scope,
            sampled_lb=float(sampled_lb),
            n_witness=self.n_witness,
            token_index=ti,
            details=details,
        )


# --------------------------------------------------------------------------- #
# Anchor preparation from a harness + prompt
# --------------------------------------------------------------------------- #
def prepare_anchor(harness, prompt: str, layer: int):
    """Compute (h_{L-1}, attn_anchor, w, pooled_h_L, T) for one prompt in float64.

    h_{L-1}     : (T, D) residual stream entering block (layer-1)  [= hidden_states[layer-1]]
    attn_anchor : (T, D) the block's attention sublayer output at the anchor (FROZEN)
    w           : (T,) mean-pool weights over real tokens (sum to 1)
    pooled_h_L  : (D,) the probe's actual input (mean-pooled hidden_states[layer])
    Only real (non-pad) token rows are returned (T = number of real tokens).
    """
    model = harness.model
    tok = harness.tokenizer
    enc = tok([prompt], return_tensors="pt", truncation=True, max_length=harness.cfg.max_len)
    enc = {k: v.to(harness.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    mask = enc["attention_mask"][0].to(torch.float64)
    T = int(mask.sum().item())
    hs_prev = out.hidden_states[layer - 1][0].to(torch.float64)[:T]   # (T, D)
    hs_cur = out.hidden_states[layer][0].to(torch.float64)[:T]        # (T, D)
    w = mask[:T] / mask[:T].sum()
    block = model.transformer.h[layer - 1]
    # attention anchor = attn(ln_1(h_{L-1}))  (block applies ln_1 then attn on the residual)
    with torch.no_grad():
        ln1 = block.ln_1(hs_prev.to(next(model.parameters()).dtype).unsqueeze(0))
        attn_anchor = block.attn(ln1)[0][0].to(torch.float64)[:T]     # (T, D)
    pooled_cur = (hs_cur * w[:, None]).sum(0)
    return hs_prev, attn_anchor, w, pooled_cur, T


def make_certifier_from_harness(harness, probe, sigma, layer: int, n_witness: int = 2000):
    """Build a DeepFrozenAttnCertifier for block (layer-1) of the harness model
    (cast to float64 internally via the extracted weights)."""
    block = harness.model.transformer.h[layer - 1]
    return DeepFrozenAttnCertifier(block, probe, sigma, layer, n_witness=n_witness)


def deep_certified_radius(certifier, h, attn_anchor, w, tau, grid, method="crown",
                          token_index=-1, witness_seed=0):
    """Largest eps in `grid` whose deep certificate still certifies unreachable.

    U is monotone non-decreasing in eps, so we sweep ascending and stop at the
    first failure. Returns the last certified eps (0.0 if none)."""
    grid = np.sort(np.asarray(grid, dtype=float))
    last = 0.0
    for eps in grid:
        res = certifier.certify(h, attn_anchor, w, float(eps), tau,
                                method=method, token_index=token_index,
                                witness_seed=witness_seed)
        if res.certified_unreachable:
            last = float(eps)
        else:
            break
    return last
