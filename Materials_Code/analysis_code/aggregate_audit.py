"""Executable aggregate audit (minimum reporting protocol, Section 5.4).
Usage:  python3 aggregate_audit.py cases/caseA.json [--prev 0.001 0.01 0.05] [--k 10 100] [--out record_A.json]
Input : one JSON per published report (counts or derived counts, printed metrics, provenance fields; see cases/).
Output: an audit record: for every step, the inputs it needs, the inputs missing, a status (done / partial / stopped),
        the evidential label of the result, and the stop reason when an assumption is unsupported.
RNG   : numpy.random.default_rng (PCG64), seed 20261021 + 5, 200,000 draws for credible intervals."""
import json, sys, argparse, numpy as np
from scipy.stats import norm, beta
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad

ap = argparse.ArgumentParser(); ap.add_argument('case'); ap.add_argument('--prev', type=float, nargs='*', default=[0.001, 0.01, 0.05])
ap.add_argument('--k', type=float, nargs='*', default=[10, 100]); ap.add_argument('--out', default=None); a = ap.parse_args()
C = json.load(open(a.case)); rec = dict(case=C['id'], reference=C['reference'], steps=[])
Z = 1.959964; SEED = 20261021 + 5; DRAWS = 200000
def wilson(k, m):
    p = k / m; c = (p + Z * Z / (2 * m)) / (1 + Z * Z / m); h = Z * np.sqrt(p * (1 - p) / m + Z * Z / (4 * m * m)) / (1 + Z * Z / m); return [float(c - h), float(c + h)]
def cp(k, m):
    return [0.0 if k == 0 else float(beta.ppf(.025, k, m - k + 1)), 1.0 if k == m else float(beta.ppf(.975, k + 1, m - k))]
def ppv(t, f, pi): return t * pi / (t * pi + f * (1 - pi))
def step(name, needs, missing, status, label, result=None, note=None, stop=None):
    rec['steps'].append(dict(step=name, inputs_needed=needs, inputs_missing=missing, status=status, evidential_label=label, result=result, note=note, stop_reason=stop))

cn = C.get('counts'); unresolved = []
if not cn:
    step('S0 counts', ['four confusion-matrix cells or derivable rates'], ['counts'], 'stopped', 'none', stop='no counts: only printed-rate checks are possible'); cn = None
if cn:
    TP, FN, FP, TN = cn['TP'], cn['FN'], cn['FP'], cn['TN']; n = TP + FN + FP + TN
    tpr, fpr = TP / (TP + FN), FP / (FP + TN); pit = (TP + FN) / n; acc = (TP + TN) / n
    # ---- S1 reconstruction and class convention
    def met(TP, FN, FP, TN):
        n = TP + FN + FP + TN; pe = ((TP + FN) * (TP + FP) + (FP + TN) * (FN + TN)) / n ** 2; po = (TP + TN) / n
        return dict(accuracy=po, precision=TP / (TP + FP), recall=TP / (TP + FN), f1=2 * TP / (2 * TP + FP + FN), jaccard=TP / (TP + FP + FN), kappa=(po - pe) / (1 - pe))
    mp, ml = met(TP, FN, FP, TN), met(TN, FP, FN, TP); d = C['printed']['decimals']; res = {}
    for key, val in C['printed'].items():
        if key == 'decimals': continue
        def how(x):
            r = round(100 * x, d); t = np.floor(100 * x * 10 ** d + 1e-9) / 10 ** d
            return [w for w, v in (('rounding', r), ('truncation', t)) if abs(v - val) < 10 ** (-d) / 2 + 1e-12]
        res[key] = dict(printed=val, phishing_positive=round(100 * mp[key], d + 2), legitimate_positive=round(100 * ml[key], d + 2),
                        phishing_matches=how(mp[key]), legitimate_matches=how(ml[key]), invariant=bool(abs(mp[key] - ml[key]) < 1e-12))
    ident = [k for k, v in res.items() if not v['invariant'] and bool(v['phishing_matches']) != bool(v['legitimate_matches'])]
    conv = 'not identifiable'
    if ident:
        conv = 'legitimate class positive' if all(res[k]['legitimate_matches'] for k in ident) else 'phishing class positive' if all(res[k]['phishing_matches'] for k in ident) else 'mixed'
    step('S1 metric reconstruction and class convention', ['counts', 'printed metrics'], [], 'done', 'arithmetic (V; C if counts derived)',
         dict(convention_evidence=conv, convention_identified_by=ident, per_metric=res), 'invariant measures (accuracy, kappa, MCC, Youden) carry no evidence on the convention; printed values are compared at the printed number of decimals with the unrounded recomputation')
    # ---- S2 internal consistency
    inc = []
    if abs(100 * acc - C['printed']['accuracy']) > 10 ** (-d) / 2 + 1e-9: inc.append(f"printed accuracy {C['printed']['accuracy']} vs from counts {100*acc:.3f}")
    if C.get('stated_n') and C['stated_n'] != n: inc.append(f"stated n {C['stated_n']} vs sum of cells {n}")
    step('S2 internal consistency', ['counts', 'printed accuracy', 'stated n'], [], 'done', 'arithmetic', dict(inconsistencies=inc or 'none found'),
         'when inconsistent, all later steps are conditional on which set of numbers is correct (clarification by the authors needed)')
    # ---- S3 sampling intervals
    iv = {'accuracy': (TP + TN, n), 'TPR': (TP, TP + FN), 'FPR': (FP, FP + TN), 'PPV (phishing positive)': (TP, TP + FP)}
    step('S3 sampling intervals', ['counts', 'independence of instances'], [], 'done', 'conditional on the partition being an independent, representative sample',
         {k: dict(estimate=v[0] / v[1], wilson=wilson(*v), clopper_pearson=cp(*v)) for k, v in iv.items()}, 'sampling (frequentist) intervals; they say nothing about deployment')
    # ---- S4 provenance record
    P = C['provenance']; miss = [k for k, v in P.items() if str(v).startswith('not') or v in ('none', 'symbolic only')]
    step('S4 provenance record', list(P.keys()), miss, 'partial' if miss else 'done', 'checklist (no inference)', P,
         'unresolved items are recorded, not assumed; every interval and deployment figure below inherits them')
    # ---- S5 Layer A
    rng = np.random.default_rng(SEED); td = rng.beta(TP + .5, FN + .5, DRAWS); fd = rng.beta(FP + .5, TN + .5, DRAWS)
    t_lo, f_hi = cp(TP, TP + FN)[0], cp(FP, FP + TN)[1]; A = {}
    for pi in a.prev:
        pp = ppv(td, fd, pi)
        A[str(pi)] = dict(point=float(ppv(tpr, fpr, pi)) if fpr > 0 else 1.0,
                          conditional_fpr_only_lower=float(ppv(tpr, f_hi, pi)),
                          joint_lower_clopper_pearson=float(ppv(t_lo, f_hi, pi)), joint_lower_coverage_at_least=0.975 ** 2,
                          credible_95_jeffreys=[float(np.percentile(pp, 2.5)), float(np.percentile(pp, 97.5))],
                          fpr_max_for_90pct=float(tpr * pi * 0.1 / (0.9 * (1 - pi))))
    step('S5 Layer A: precision at stated prevalence', ['TPR, FPR', 'stated prevalence (H)'], [], 'done', 'conditional on rate transfer and on the stated prevalence (H)', A,
         'conditional_fpr_only_lower holds TPR at its estimate (not a joint bound); joint_lower moves both rates (independent class-conditional binomials, coverage >= 0.975^2 = 95.06%); neither covers prevalence or rate-transfer uncertainty')
    # ---- S6 accuracy bound
    if acc >= max(pit, 1 - pit):
        B = {str(pi): dict(min_precision=float(pi / (pi + (1 - pi) * (1 - acc) / (1 - pit))), accuracy_needed_for_90pct=float(1 - (1 - pit) * pi * 0.1 / (0.9 * (1 - pi)))) for pi in a.prev}
        step('S6 accuracy-to-precision bound', ['accuracy', 'evaluation prevalence'], [], 'done', 'analytic bound, conditional on rate transfer', B)
    else:
        step('S6 accuracy-to-precision bound', ['accuracy', 'evaluation prevalence'], [], 'stopped', 'none', stop='accuracy below max(pi_t, 1 - pi_t): Eq. (19) not applicable')
    # ---- S7 Layer B
    auc = C.get('auc'); miss = [] if auc else ['AUC']
    if fpr == 0 or tpr in (0, 1):
        step('S7 Layer B: ROC-model scenarios', ['TPR, FPR > 0', 'AUC'], miss, 'stopped', 'none',
             stop='observed FPR (or miss rate) is zero: a binormal curve through the observed point is undefined; no AUC to anchor a model' if not auc else 'observed point on the boundary of ROC space')
    else:
        zt, zf = norm.ppf(tpr), norm.ppf(fpr); fam = {'EV-point': lambda f, d=zt - zf: norm.cdf(d + norm.ppf(f))}
        if auc:
            fam['EV-AUC'] = lambda f, d=np.sqrt(2) * norm.ppf(auc): norm.cdf(d + norm.ppf(f))
            g = lambda b: (zt - b * zf) / np.sqrt(1 + b * b) - norm.ppf(auc)
            for i, (lo, hi) in enumerate([(0.05, 0.9), (0.9, 3)]):
                try: b = brentq(g, lo, hi); aa = zt - b * zf; fam[f'UV{i+1} (b={b:.2f})'] = lambda f, aa=aa, b=b: norm.cdf(aa + b * norm.ppf(f))
                except ValueError: pass
            lg = lambda m: quad(lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2 * (1 - 1 / (1 + np.exp(-(x - m)))), -60, 60, limit=200)[0] - auc
            try: mu = brentq(lg, 3, 30); fam['Logistic'] = lambda f, mu=mu: 1 - 1 / (1 + np.exp(-(np.log((1 - f) / f) - mu)))
            except ValueError: pass
        Bm = {}
        for pi in a.prev:
            for k in a.k:
                cost = lambda t, f: 1e4 * (k * pi * (1 - t) + (1 - pi) * f); c_obs = cost(tpr, fpr)
                for nm, roc in fam.items():
                    r = minimize_scalar(lambda lf: float(cost(roc(np.exp(lf)), np.exp(lf))), bounds=(np.log(1e-7), np.log(.2)), method='bounded', options=dict(xatol=1e-9))
                    Bm[f'pi={pi}|k={k:g}|{nm}'] = dict(optimum_fpr=float(np.exp(r.x)), saving_vs_observed=float(1 - r.fun / c_obs), saving_vs_model_at_observed_fpr=float(1 - r.fun / cost(roc(fpr), fpr)))
        step('S7 Layer B: ROC-model scenarios', ['TPR, FPR > 0', 'AUC (optional)'], miss, 'partial' if miss else 'done', 'model-dependent scenario (not an achieved result)', Bm,
             'saving_vs_observed can be negative when the model curve lies below the observed point; that is a model discrepancy, not an optimal-policy result. Candidate AUCs are sensitivity choices, not confidence limits')
    # ---- S8 Layer C
    P = C['provenance']; cal = P.get('calibration_evidence', 'none'); thr = P.get('operating_threshold', 'not stated')
    if cal == 'none' or str(thr).startswith('not'):
        tau = {str(pi): {f'k={k:g}': float(((1 / k) * (pit / (1 - pit)) / (pi / (1 - pi))) / (1 + (1 / k) * (pit / (1 - pit)) / (pi / (1 - pi)))) for k in a.k} for pi in a.prev}
        step('S8 Layer C: thresholds', ['score release or calibration curve', 'operating threshold'], ['calibration evidence', 'operating threshold'], 'stopped', 'conditional on calibration (unsupported)',
             dict(eq17_mapping_for_reference_only=tau), stop='no calibration evidence or threshold: no statement about what threshold to use; Eq. (17) values are shown only as a conditional mapping')
    # ---- S9 baseline scenarios
    if C.get('baselines'):
        out = {}
        for b in C['baselines']:
            pb = b['accuracy'] / 100; row = {}
            for nb in [1000, 3000, 10000, None]:
                se = np.sqrt(acc * (1 - acc) / n + (pb * (1 - pb) / nb if nb else 0)); row[str(nb)] = float(2 * norm.sf(abs((acc - pb) / se)))
            out[b['name']] = row
        step('S9 baseline scenarios', ['baseline accuracies', 'baseline sample sizes'], ['baseline sample sizes'] if 'not' in str(C.get('baseline_sample_sizes', '')) else [], 'partial', 'pedagogical sampling scenario; not comparative evidence',
             out, 'different datasets: neither a small p-value nor an interval crossing another study\'s point estimate identifies a method effect; a common-population comparison needs the same data, split and (for paired tests) per-instance predictions')
    # ---- S10 clustering
    if str(P.get('cluster_structure', 'not stated')).startswith('not'):
        step('S10 dependence', ['cluster identifiers or sizes'], ['cluster structure'], 'partial', 'scenario',
             {f'deff={dd}': wilson(FP / dd, (FP + TN) / dd) for dd in (1.45, 3.45)}, 'Wilson FPR interval at the effective size n/deff (width illustration only; coverage under clustering requires the cluster model of Section 4.8)')
rec['unresolved_inputs'] = sorted({m for s in rec['steps'] for m in (s['inputs_missing'] or [])})
rec['stopped_steps'] = [s['step'] for s in rec['steps'] if s['status'] == 'stopped']
out = a.out or f"record_{C['id']}.json"; json.dump(rec, open(out, 'w'), indent=1, default=float)
print(f"Audit record for case {C['id']} -> {out}")
for s in rec['steps']: print(f"  {s['step']:<50} {s['status']:<8} [{s['evidential_label']}]" + (f"  STOP: {s['stop_reason']}" if s['stop_reason'] else ''))
print('  unresolved inputs:', ', '.join(rec['unresolved_inputs']) or 'none')
