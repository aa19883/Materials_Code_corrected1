"""Controlled example with known score distributions (Section 4.10).
Question: when the assumptions behind a layer of the audit are violated by construction, does the layer's
evidential label (A: rate transfer; B: + ROC family; C: + calibration) flag the right failure?
Everything except the coverage check is exact (no sampling).  Run: python3 controlled.py  -> results_controlled.json
Seed 20261021+7 (numpy default_rng / PCG64) for the coverage check only."""
import json, numpy as np
from scipy.stats import norm, beta
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad

SEED = 20261021 + 7
PI_T, PI, K = 0.488, 0.01, 10.0          # evaluation prevalence, deployment prevalence, cost ratio
AUC_PUB = 0.99992                         # the AUC the "source" prints
FPR_OP = 0.013                            # operating point of the "source" (as in Case A)
N_POS, N_NEG = 1463, 1537
logit = lambda p: np.log(p / (1 - p)); expit = lambda z: 1 / (1 + np.exp(-z))
ppv = lambda t, f, pi: t * pi / (t * pi + f * (1 - pi))
cost = lambda t, f, k, pi: 1e4 * (k * pi * (1 - t) + (1 - pi) * f)

# ---------- score models: functions fpr(c), tpr(c) of a decision cut c on the latent score
class Normal:                              # negatives N(delta,1), positives N(d,s_pos)
    def __init__(s, d, s_pos=1.0, delta=0.0): s.d, s.sp, s.delta = d, s_pos, delta
    def fpr(s, c): return norm.sf(c - s.delta)
    def tpr(s, c): return norm.sf((c - s.d) / s.sp)
class Laplace:                             # negatives Laplace(0,1), positives Laplace(mu,1)
    def __init__(s, mu): s.mu = mu
    def fpr(s, c): return np.where(c >= 0, .5 * np.exp(-c), 1 - .5 * np.exp(c))
    def tpr(s, c): return np.where(c >= s.mu, .5 * np.exp(-(c - s.mu)), 1 - .5 * np.exp(c - s.mu))
def auc_of(model):
    cs = np.linspace(-60, 60, 600001); f = model.fpr(cs); t = model.tpr(cs)
    return float(np.trapezoid(t[::-1], f[::-1])) if hasattr(np, 'trapezoid') else float(np.trapz(t[::-1], f[::-1]))
def cut_for_fpr(model, f): return brentq(lambda c: float(model.fpr(c)) - f, -40, 40, xtol=1e-12)
def opt_cut(model, k, pi):
    r = minimize_scalar(lambda c: float(cost(model.tpr(c), model.fpr(c), k, pi)), bounds=(-5, 15), method='bounded', options=dict(xatol=1e-10)); return float(r.x)

# ---------- the audit, applied to what a source prints: one operating point (t0,f0) and one AUC
def audit_layer_B(t0, f0, auc, k=K, pi=PI):
    zt, zf = norm.ppf(t0), norm.ppf(f0); fam = {}
    d_auc = np.sqrt(2) * norm.ppf(auc); fam['EV-AUC'] = lambda f, d=d_auc: norm.cdf(d + norm.ppf(f))
    d_pt = zt - zf; fam['EV-point'] = lambda f, d=d_pt: norm.cdf(d + norm.ppf(f))
    g = lambda b: (zt - b * zf) / np.sqrt(1 + b * b) - norm.ppf(auc)
    for i, (lo, hi) in enumerate([(0.05, 0.9), (0.9, 3)]):
        try:
            b = brentq(g, lo, hi); a = zt - b * zf; fam[f'UV{i+1}'] = lambda f, a=a, b=b: norm.cdf(a + b * norm.ppf(f))
        except ValueError: pass
    def auc_log(mu): return quad(lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2 * (1 - 1 / (1 + np.exp(-(x - mu)))), -60, 60, limit=200)[0]
    mu = brentq(lambda m: auc_log(m) - auc, 3, 30); fam["Logistic"] = lambda f, mu=mu: 1 - 1 / (1 + np.exp(-(np.log((1 - f) / f) - mu)))
    for a_ in [0.9998, 0.99997]:
        d = np.sqrt(2) * norm.ppf(a_); fam[f'EV-AUC={a_}'] = lambda f, d=d: norm.cdf(d + norm.ppf(f))
    out = {}
    c_obs = cost(t0, f0, k, pi)
    for nm, roc in fam.items():
        r = minimize_scalar(lambda lf: float(cost(roc(np.exp(lf)), np.exp(lf), k, pi)), bounds=(np.log(1e-7), np.log(.2)), method='bounded', options=dict(xatol=1e-9))
        out[nm] = dict(fpr=float(np.exp(r.x)), saving=float(1 - r.fun / c_obs))
    return out

def tau_eval(k, pi, pit=PI_T):               # Eq. (17)
    o = (1 / k) * (pit / (1 - pit)) / (pi / (1 - pi)); return o / (1 + o)

def run_world(name, ev, dep, T=1.0, d_ev=None):
    """ev: evaluation-world model (what the source measured); dep: deployment-world model; T: score over-confidence (T=1 calibrated)."""
    c_op = cut_for_fpr(ev, FPR_OP); t0, f0 = float(ev.tpr(c_op)), float(ev.fpr(c_op)); auc = auc_of(ev)
    res = dict(world=name, t0=t0, f0=f0, auc_evaluation=auc)
    # Layer A
    res['A_audit_ppv'] = float(ppv(t0, f0, PI)); res['A_true_ppv'] = float(ppv(dep.tpr(c_op), dep.fpr(c_op), PI))
    # Layer B (audit uses the printed operating point and AUC only; the true optimum is computed in the deployment world)
    B = audit_layer_B(t0, f0, AUC_PUB if abs(auc - AUC_PUB) > 0.5e-5 else AUC_PUB)
    res['B_audit'] = B; sav = [v['saving'] for v in B.values()]; fp = [v['fpr'] for v in B.values()]
    res['B_audit_saving_range'] = [min(sav), max(sav)]; res['B_audit_fpr_range'] = [min(fp), max(fp)]
    c_star = opt_cut(dep, K, PI)
    res['B_true_opt_fpr'] = float(dep.fpr(c_star)); res['B_true_saving'] = float(1 - cost(dep.tpr(c_star), dep.fpr(c_star), K, PI) / cost(dep.tpr(c_op), dep.fpr(c_op), K, PI))
    res['B_true_inside_audit_range'] = bool(min(sav) - 1e-9 <= res['B_true_saving'] <= max(sav) + 1e-9)
    res['B_true_fpr_inside_audit_range'] = bool(min(fp) - 1e-9 <= res['B_true_opt_fpr'] <= max(fp) + 1e-9)
    return res, (c_op, c_star)

d_pub = np.sqrt(2) * norm.ppf(AUC_PUB)
W = {}
# W1: every assumption holds (equal-variance binormal; pure prior shift; calibrated scores)
ev1 = Normal(d_pub); W['W1 all assumptions hold'], _ = run_world('W1', ev1, Normal(d_pub))
# W2: ROC family outside the audit's list (Laplace location shift matched to the published AUC); pure prior shift
mu_lap = brentq(lambda m: auc_of(Laplace(m)) - AUC_PUB, 2, 40, xtol=1e-9); ev2 = Laplace(mu_lap)
W['W2 ROC shape outside the families'], _ = run_world('W2', ev2, Laplace(mu_lap))
# W3: rate shift: deployment negatives drift by delta = 0.4 sd (positives unchanged), prior also shifts
W['W3 rate shift (negatives drift)'], _ = run_world('W3', ev1, Normal(d_pub, delta=0.4))
# W4: same world as W1 but the network's scores are over-confident (temperature T = 3), a monotone map of calibrated scores
W['W4 miscalibrated scores (rank-preserving)'], _ = run_world('W4', ev1, Normal(d_pub))

# ---------- Layer C: threshold policy from Eq. (17) applied to the (possibly distorted) scores
def layer_C(ev, dep, T, d):
    """Equal-variance normal evaluation world: calibrated posterior at evaluation prevalence is p(c)=expit(logit(PI_T)+d c-d^2/2).
    Audit: flag when the reported score >= tau_eval(K,PI) with reported score = expit(T*logit p).  Truth: cost at that cut in the deployment world."""
    tau = tau_eval(K, PI)
    c_audit = (logit(tau) / T - logit(PI_T) + d * d / 2) / d          # cut implied by the audit's threshold
    c_star = opt_cut(dep, K, PI)
    ca, cs = cost(dep.tpr(c_audit), dep.fpr(c_audit), K, PI), cost(dep.tpr(c_star), dep.fpr(c_star), K, PI)
    return dict(tau_eval=float(tau), cut_audit=float(c_audit), cut_true_opt=float(c_star), fpr_audit=float(dep.fpr(c_audit)), fpr_true_opt=float(dep.fpr(c_star)),
                cost_audit=float(ca), cost_true_opt=float(cs), excess_cost_pct=float(100 * (ca / cs - 1)))
C = {}
C['W1'] = layer_C(ev1, Normal(d_pub), 1.0, d_pub)
C['W3'] = layer_C(ev1, Normal(d_pub, delta=0.4), 1.0, d_pub)
C['W4'] = layer_C(ev1, Normal(d_pub), 3.0, d_pub)
C['W2'] = None   # not defined: the Laplace world has no Gaussian posterior map; Layer C is not evaluated (the audit would also stop here, no calibration evidence)

# ---------- sampling coverage of the Layer-A joint limits (95.06% nominal) under W1 and W3
rng = np.random.default_rng(SEED); REPS = 20000
def coverage(ev, dep):
    c_op = cut_for_fpr(ev, FPR_OP); t_ev, f_ev = float(ev.tpr(c_op)), float(ev.fpr(c_op)); truth = ppv(float(dep.tpr(c_op)), float(dep.fpr(c_op)), PI)
    tp = rng.binomial(N_POS, t_ev, REPS); fp = rng.binomial(N_NEG, f_ev, REPS)
    t_lo = np.where(tp == 0, 0, beta.ppf(.025, np.maximum(tp, 1e-9), N_POS - tp + 1)); t_hi = np.where(tp == N_POS, 1, beta.ppf(.975, tp + 1, np.maximum(N_POS - tp, 1e-9)))
    f_lo = np.where(fp == 0, 0, beta.ppf(.025, np.maximum(fp, 1e-9), N_NEG - fp + 1)); f_hi = np.where(fp == N_NEG, 1, beta.ppf(.975, fp + 1, np.maximum(N_NEG - fp, 1e-9)))
    lo = ppv(t_lo, f_hi, PI); hi = ppv(t_hi, np.maximum(f_lo, 1e-12), PI)
    return float(np.mean((lo <= truth) & (truth <= hi))), float(truth)
cov = {}
cov['W1'] = coverage(ev1, Normal(d_pub)); cov['W3'] = coverage(ev1, Normal(d_pub, delta=0.4))
cov['mc_se_pp_max'] = float(100 * np.sqrt(.25 / REPS))

json.dump(dict(worlds=W, layer_C=C, coverage_joint_limits=cov, settings=dict(pi_t=PI_T, pi=PI, k=K, auc_published=AUC_PUB, fpr_operating=FPR_OP, laplace_mu=mu_lap, d_equal_variance=float(d_pub),
          drift_delta=0.4, temperature=3.0, seed=SEED, reps=REPS, rng='numpy.random.default_rng (PCG64)')), open('results_controlled.json', 'w'), indent=1, default=float)
for k_, v in W.items():
    print(k_, 'op TPR %.5f FPR %.4f' % (v['t0'], v['f0']), '| A audit %.3f true %.3f' % (v['A_audit_ppv'], v['A_true_ppv']),
          '| B saving range %.3f-%.3f true %.3f inside %s | FPR range %.4f-%.4f true %.4f' % (*v['B_audit_saving_range'], v['B_true_saving'], v['B_true_inside_audit_range'], *v['B_audit_fpr_range'], v['B_true_opt_fpr']))
print(json.dumps(C, indent=1)); print(cov); print('laplace mu', mu_lap)
