"""Revision analyses (round 2): limiting-case p-values, joint precision bounds,
uncertainty layers, Table 9 parameter listing, clustered-simulation checks,
Monte Carlo error of the simulations.  Reads results.json, writes results_rev.json.
Run after part2.py..part4.py:  python3 part5.py
RNG: numpy.random.default_rng (PCG64); seeds are listed in the output."""
import json, numpy as np
from scipy.stats import norm, beta, binom, binomtest, chi2
exec(open('core.py').read().split("mp=metrics")[0])      # helpers + inputs (I, wilson, cp, jeff, metrics)
R0 = json.load(open('results.json'))
OUT = {}
SEED = I['seed']

# ---------------------------------------------------------------- 1. limiting-case p-values (Section 4.3)
p1, n1 = 2979 / 3000, 3000
lim = {}
for name, pb in [('Decision tree', .99), ('LR+SVM+DT', .9812), ('LSTM', .9737), ('KNN', .94)]:
    se_unpooled = np.sqrt(p1 * (1 - p1) / n1)                 # baseline treated as known exactly (n_b = infinity)
    z = (p1 - pb) / se_unpooled
    se0 = np.sqrt(pb * (1 - pb) / n1)                         # null-variance (score) version
    z0 = (p1 - pb) / se0
    ex = binomtest(2979, 3000, pb).pvalue                     # exact two-sided binomial test of H0: p = pb
    lim[name] = dict(z_eq9=float(z), p_eq9=float(2 * norm.sf(abs(z))), z_score=float(z0),
                     p_score=float(2 * norm.sf(abs(z0))), p_exact_binomial=float(ex))
OUT['limiting_case'] = lim

# ---------------------------------------------------------------- 2. precision lower limits: conditional vs joint (Section 4.9, Table 12)
def ppv(t, f, pi): return t * pi / (t * pi + f * (1 - pi))
rngJ = np.random.default_rng(SEED + 5)
DRAWS = 200000
cases = R0['cases']
joint = {}
for key in ['A', 'B', 'C']:
    c = cases[key]; TP_, FN_, FP_, TN_ = c['TP'], c['FN'], c['FP'], c['TN']
    tpr, fpr = TP_ / (TP_ + FN_), FP_ / (FP_ + TN_)
    f_hi = cp(FP_, FP_ + TN_)[1]                              # upper limit of two-sided 95% CP = one-sided 97.5%
    t_lo = cp(TP_, TP_ + FN_)[0]                              # lower limit of two-sided 95% CP = one-sided 97.5%
    # Jeffreys posterior: TPR and FPR are independent Beta variables (Dirichlet property)
    t_d = rngJ.beta(TP_ + .5, FN_ + .5, DRAWS); f_d = rngJ.beta(FP_ + .5, TN_ + .5, DRAWS)
    row = dict(tpr=tpr, fpr=fpr, fpr_cp_upper=f_hi, tpr_cp_lower=t_lo)
    for pi in [0.001, 0.01, 0.05]:
        pp = ppv(t_d, f_d, pi)
        row[str(pi)] = dict(point=ppv(tpr, fpr, pi) if fpr > 0 else 1.0,
                            fpr_only_cp=ppv(tpr, f_hi, pi),               # conditional on estimated TPR (previous Table 12)
                            joint_cp=ppv(t_lo, f_hi, pi),                  # both limits moved (coverage >= 0.975^2)
                            joint_cp_cover_lower_bound=0.975 ** 2,
                            joint_post_lower_975=float(np.percentile(pp, 2.5)),
                            joint_post_95=[float(np.percentile(pp, 2.5)), float(np.percentile(pp, 97.5))])
    joint[key] = row
OUT['joint_precision'] = joint
OUT['joint_precision_meta'] = dict(draws=DRAWS, rng='numpy.random.default_rng (PCG64)', seed=SEED + 5)

# Monte Carlo error of the 200,000-draw credible intervals for Case A precision at 1% (10 seeds)
ends = []
for s in range(10):
    r = np.random.default_rng(SEED + 100 + s)
    t = r.beta(TP + .5, FN + .5, DRAWS); f = r.beta(FP + .5, TN + .5, DRAWS); pp = ppv(t, f, .01)
    ends.append([np.percentile(pp, 2.5), np.percentile(pp, 97.5)])
ends = np.array(ends)
OUT['mc_error_credible_A_1pct'] = dict(sd_lower_pp=float(100 * ends[:, 0].std(ddof=1)), sd_upper_pp=float(100 * ends[:, 1].std(ddof=1)),
                                       mean=[float(100 * ends[:, 0].mean()), float(100 * ends[:, 1].mean())])

# ---------------------------------------------------------------- 3. three kinds of uncertainty for Case A precision at 1% prevalence
t0, f0 = TP / (TP + FN), FP / (FP + TN)
unc = dict(
    point=ppv(t0, f0, .01),
    source_rates_credible=R0['A_ppv']['0.010000']['ci'],                                  # Dirichlet, prevalence fixed at 1%
    source_rates_joint_cp=[joint['A']['0.01']['joint_cp'], None],
    prevalence_0p5_to_2=[ppv(t0, f0, .005), ppv(t0, f0, .02)],                            # rates fixed at observed
    rate_transfer_fpr_x2_to_x0p5=[ppv(t0, 2 * f0, .01), ppv(t0, .5 * f0, .01)],         # TPR, prevalence fixed
)
# combined scenario box: prevalence 0.5-2% x FPR multiplier 0.5-2 at observed TPR; and with the joint lower CP limit at worst corner
corners = {}
for pi in [.005, .02]:
    for m in [.5, 2]:
        corners[f'pi={pi}|fprx{m}'] = ppv(t0, m * f0, pi)
unc['scenario_box_corners'] = corners
unc['worst_corner_with_joint_cp_limits'] = ppv(cp(TP, TP + FN)[0], 2 * cp(FP, FP + TN)[1], .005)
OUT['uncertainty_layers_A_1pct'] = unc

# ---------------------------------------------------------------- 4. Table 9: parameter listing and the two kinds of saving
S = R0['roc_sens']
t9 = {}
for name, v in S.items():
    row = dict(label=v['label'], param=v['param'])
    for k in ['10', '100']:
        for pi in ['0.01', '0.05']:
            o = v['cost'][k]['opt_pi' + pi]
            row[f'k{k}_pi{pi}'] = dict(opt_fpr_pct=100 * o['fpr'], opt_tpr_pct=100 * o['tpr'], saving_vs_observed=o['saving_vs_obs'],
                                       saving_vs_model_at_observed_fpr=o['saving_vs_model'])
    t9[name] = row
OUT['table9_detail'] = t9
OUT['roc_numerics'] = dict(
    ev_auc_dprime=R0['dprime']['auc'], ev_point_dprime=R0['dprime']['point'],
    model_tpr_at_observed_fpr_ev_auc=R0['dprime']['tpr_model_at_obs'],
    unequal_variance=dict(UV1=S['UV-1']['param'], UV2=S['UV-2']['param']),
    logistic_mu=S['Logistic']['param']['mu'],
    solver=dict(unequal_variance='scipy.optimize.brentq on g(b)=(z_TPR - b z_FPR)/sqrt(1+b^2) - Phi^-1(AUC), brackets [0.05,0.9] and [0.9,3], default xtol 2e-12',
                logistic='scipy.optimize.brentq on AUC(mu)=AUC_stated, AUC(mu)=integral f_neg(x)(1-F_pos(x))dx by scipy.integrate.quad over [-60,60], brackets mu in [3,30]',
                optimum='scipy.optimize.minimize_scalar(method=bounded) on log(FPR) in [log 1e-7, log 0.2], xatol 1e-9'))

# ---------------------------------------------------------------- 5. clustered simulation: truncation versus residual cluster, MC error
def intervals(k, m, kind):
    k = np.asarray(k, float); p = k / m; z = 1.959964
    if kind == 'wilson':
        c = (p + z * z / (2 * m)) / (1 + z * z / m); h = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / (1 + z * z / m); return c - h, c + h
    if kind == 'cp':
        lo = np.where(k == 0, 0.0, beta.ppf(0.025, np.maximum(k, 1e-9), m - k + 1)); hi = np.where(k == m, 1.0, beta.ppf(0.975, k + 1, np.maximum(m - k, 1e-9))); return lo, hi
    if kind == 'jeff':
        lo = np.where(k == 0, 0.0, beta.ppf(0.025, k + .5, m - k + .5)); hi = np.where(k == m, 1.0, beta.ppf(0.975, k + .5, m - k + .5)); return lo, hi
REPS = 200000
rngC = np.random.default_rng(SEED + 6)
p_true, m_full = 20 / 1537, 1537
def sim(csize, rho, residual, reps=REPS):
    s = 1 / rho - 1; a, b = p_true * s, (1 - p_true) * s            # beta(a,b): mean p, intra-cluster correlation 1/(a+b+1)=rho
    ncl = m_full // csize; rem = m_full - ncl * csize
    th = rngC.beta(a, b, (reps, ncl)); k = rngC.binomial(csize, th).sum(1); m = ncl * csize
    if residual and rem > 0:
        th2 = rngC.beta(a, b, reps); k = k + rngC.binomial(rem, th2); m = m_full
    return k, m
clus = {}
for csize in [10, 50]:
    for rho in [0.02, 0.05, 0.10]:
        row = dict(deff=1 + (csize - 1) * rho, n_trunc=(m_full // csize) * csize, residual_size=m_full - (m_full // csize) * csize)
        for residual in [False, True]:
            k, m = sim(csize, rho, residual)
            for kind in ['wilson', 'cp', 'jeff']:
                lo, hi = intervals(k, m, kind); cov = float(np.mean((lo <= p_true) & (p_true <= hi)))
                row[f"{kind}_{'with_residual' if residual else 'truncated'}"] = cov
        row['mc_se_max_pp'] = float(100 * np.sqrt(.25 / REPS))
        clus[f'{csize}|{rho}'] = row
OUT['cluster_check'] = dict(reps=REPS, seed=SEED + 6, results=clus)

# analytic power with n_eff substituted into the normal-approximation power formula (not simulated)
def power_neff(n_eff, pa=.993, pb=.990):
    se = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / n_eff); d = pa - pb
    return float(norm.sf(1.959964 - d / se) + norm.cdf(-1.959964 - d / se))
OUT['cluster_power_analytic'] = {str(d): power_neff(3000 / d) for d in [1, 1.45, 2, 3.45]}

# Monte Carlo spread of the sample sizes needed for 80% power (paired designs), 5 independent seeds
pa, pb = .993, .990
cells = {'0.5%': (0.989, 0.004, 0.001, 0.006), '1%': (0.9865, 0.0065, 0.0035, 0.0035)}
grid = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000]
def n80(rows, key):
    xs = [r[0] for r in rows]; ys = [r[key] for r in rows]
    for i in range(1, len(xs)):
        if ys[i] >= .8 > ys[i - 1]: return xs[i - 1] + (.8 - ys[i - 1]) * (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1])
sp = {}
for lab, cp_ in cells.items():
    vals_ex, vals_cc = [], []
    for s in range(5):
        r = np.random.default_rng(SEED + 200 + s); rows = []
        for n_ in grid:
            C = r.multinomial(n_, cp_, 40000); b, c = C[:, 1], C[:, 2]; m = b + c
            with np.errstate(all='ignore'):
                pex = np.where(m > 0, np.minimum(1.0, 2 * binom.cdf(np.minimum(b, c), m, .5)), 1.0)
                cc = np.where(m > 0, (np.abs(b - c) - 1) ** 2 / m, 0.0); pcc = chi2.sf(cc, 1)
            rows.append((n_, float(np.mean(pex < .05)), float(np.mean(pcc < .05))))
        vals_ex.append(n80(rows, 1)); vals_cc.append(n80(rows, 2))
    sp[lab] = dict(exact=[float(np.mean(vals_ex)), float(np.min(vals_ex)), float(np.max(vals_ex))],
                   cc=[float(np.mean(vals_cc)), float(np.min(vals_cc)), float(np.max(vals_cc))])
OUT['paired_n80_mc_spread'] = sp
OUT['mc_se_rule'] = 'SE of a simulated proportion = sqrt(p(1-p)/reps) <= 0.0025 for reps = 40,000 (0.0011 for 200,000)'

json.dump(OUT, open('results_rev.json', 'w'), indent=1, default=float)
print(json.dumps(OUT['limiting_case'], indent=1)); print(json.dumps(OUT['joint_precision'], indent=1)); print(OUT['mc_error_credible_A_1pct'])
print(json.dumps(OUT['uncertainty_layers_A_1pct'], indent=1)); print(json.dumps(OUT['cluster_check'], indent=1)); print(OUT['cluster_power_analytic']); print(sp)
