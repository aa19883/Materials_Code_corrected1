import json, numpy as np, scipy, sys, platform
from scipy.stats import norm, beta, binom, chi2
exec(open('core.py').read().split("mp=metrics")[0])
R=json.load(open('results_core.json'))
rng=np.random.default_rng(I['seed']+1)
REPS=40000
# ---------- independent test sets ----------
pa,pb=0.993,0.990
def zstat(k1,k2,n):
    p1=k1/n;p2=k2/n; se=np.sqrt(p1*(1-p1)/n+p2*(1-p2)/n)
    with np.errstate(all='ignore'): z=np.where(se>0,(p1-p2)/se,0.0)
    return z
ns=[500,1000,2000,3000,5000,10000,14700,20000,50000]
ind=[]
for n_ in ns:
    k1=rng.binomial(n_,pa,REPS); k2=rng.binomial(n_,pb,REPS)
    z=zstat(k1,k2,n_)
    ind.append(dict(n=n_,power=float(np.mean(np.abs(z)>1.959964)),rank_first=float(np.mean(k1>k2)),tie=float(np.mean(k1==k2))))
R['sim_independent']=ind
# paired
cells={'0.5%':(0.989,0.004,0.001,0.006),'1%':(0.9865,0.0065,0.0035,0.0035)}
pairs={}
for lab,cp_ in cells.items():
    assert abs(sum(cp_)-1)<1e-12 and abs((cp_[0]+cp_[1])-pa)<1e-12 and abs((cp_[0]+cp_[2])-pb)<1e-12
    rows=[]
    for n_ in [1000,2000,3000,4000,5000,6000,8000,10000,12000,15000]:
        C=rng.multinomial(n_,cp_,REPS); b=C[:,1]; c=C[:,2]; m=b+c
        with np.errstate(all='ignore'):
            pex=np.minimum(1.0,2*binom.cdf(np.minimum(b,c),m,0.5)); pex=np.where(m>0,pex,1.0)
            cc=np.where(m>0,(np.abs(b-c)-1)**2/m,0.0); pcc=chi2.sf(cc,1)
        rows.append(dict(n=n_,power_exact=float(np.mean(pex<0.05)),power_cc=float(np.mean(pcc<0.05)),rank_first=float(np.mean(b>c)),tie=float(np.mean(b==c))))
    pairs[lab]=dict(cells=cp_,rows=rows)
R['sim_paired']=pairs
def n_for(rows,key,target=0.8):
    xs=[r['n'] for r in rows]; ys=[r[key] for r in rows]
    for i in range(1,len(xs)):
        if ys[i]>=target>ys[i-1]: return xs[i-1]+(target-ys[i-1])*(xs[i]-xs[i-1])/(ys[i]-ys[i-1])
    return None
R['sim_paired_n80']={lab:dict(exact=n_for(v['rows'],'power_exact'),cc=n_for(v['rows'],'power_cc')) for lab,v in pairs.items()}
# equal accuracy gap, exact
n_=3000; pe=0.9915; ks=np.arange(n_+1); pm=binom.pmf(ks,n_,pe)
# distribution of k1-k2: convolution
dist=np.convolve(pm,pm[::-1])  # index offset: diff = idx-n_
diffs=np.arange(-n_,n_+1)
gap9=float(dist[np.abs(diffs)>=9].sum())
R['equal_acc_gap']=dict(p_gap_ge_0_3pp=gap9,n=n_,acc=pe)
k1=rng.binomial(n_,pe,REPS);k2=rng.binomial(n_,pe,REPS); R['equal_acc_gap']['type1_z']=float(np.mean(np.abs(zstat(k1,k2,n_))>1.959964))
# ---------- exact coverage ----------
def intervals(k,m,kind):
    k=np.asarray(k,float); p=k/m; z=1.959964
    if kind=='wald':
        h=z*np.sqrt(p*(1-p)/m); return p-h,p+h
    if kind=='wilson':
        c=(p+z*z/(2*m))/(1+z*z/m); h=z*np.sqrt(p*(1-p)/m+z*z/(4*m*m))/(1+z*z/m); return c-h,c+h
    if kind=='cp':
        lo=np.where(k==0,0.0,beta.ppf(0.025,np.maximum(k,1e-9),m-k+1)); hi=np.where(k==m,1.0,beta.ppf(0.975,k+1,np.maximum(m-k,1e-9))); return lo,hi
    if kind=='jeff':
        lo=np.where(k==0,0.0,beta.ppf(0.025,k+.5,m-k+.5)); hi=np.where(k==m,1.0,beta.ppf(0.975,k+.5,m-k+.5)); return lo,hi
def coverage(p,m,kind):
    ks=np.arange(m+1); lo,hi=intervals(ks,m,kind); w=binom.pmf(ks,m,p)
    ok=(lo<=p)&(p<=hi); zero=(hi-lo)<=1e-15
    return float(w[ok].sum()),float(w[zero].sum())
cov={}
for kind in ['wald','wilson','cp','jeff']:
    c1,z1=coverage(1/1463,1463,kind); c2,z2=coverage(20/1537,1537,kind); cov[kind]=dict(cov1=c1,zero1=z1,cov20=c2)
R['coverage']=cov
# ---------- clustered extension ----------
# beta-binomial cluster model.  Cluster-level error probability theta ~ Beta(a,b) with mean p and intra-cluster correlation rho:
#   a = p(1/rho - 1), b = (1-p)(1/rho - 1)  (rho = 1/(a+b+1)); errors in a cluster of size c ~ Binomial(c, theta).
# n = 1,537 is split into floor(n/c) clusters of size c plus ONE residual cluster of size n mod c (7 for c = 10; 37 for c = 50),
# so the simulated sample always has n = 1,537 instances.  Each replicate yields k errors out of n; the interval is the ordinary
# (cluster-ignoring) Wilson / Clopper-Pearson / Jeffreys interval for k/n, unweighted; coverage = share of replicates whose interval
# contains the true p = 20/1,537.  No effective sample size is substituted into any interval.
REPS_C=200000                       # Monte Carlo SE of a coverage <= 0.11 percentage points
rngC=np.random.default_rng(I['seed']+6)
def sim_cov(p,m,csize,rho,kind,reps=REPS_C):
    if rho==0: k=rngC.binomial(m,p,reps)
    else:
        s=1/rho-1; ncl=m//csize; rem=m-ncl*csize
        k=rngC.binomial(csize,rngC.beta(p*s,(1-p)*s,(reps,ncl))).sum(1)
        if rem>0: k=k+rngC.binomial(rem,rngC.beta(p*s,(1-p)*s,reps))
    lo,hi=intervals(k,m,kind); return float(np.mean((lo<=p)&(p<=hi)))
cl={}
p=20/1537; m=1537
for csize in [1,10,50]:
    for rho in ([0.0] if csize==1 else [0.02,0.05,0.10]):
        cl[f'{csize}|{rho}']=dict(deff=1+(csize-1)*rho,**{kind:sim_cov(p,m,csize,rho,kind) for kind in ['wilson','cp','jeff']})
R['cluster_cov']=cl
R['cluster_meta']=dict(reps=REPS_C,seed=I['seed']+6,n=m,residual_cluster_sizes={'10':m%10,'50':m%50},mc_se_pp_max=100*(0.25/REPS_C)**0.5)
# power under clustering via effective n (design effect)
def power_neff(n_eff): 
    p1,p2=pa,pb; se=np.sqrt((p1*(1-p1)+p2*(1-p2))/n_eff); return float(norm.sf(1.959964-(p1-p2)/se)+norm.cdf(-1.959964-(p1-p2)/se))
R['cluster_power']={f'{d}':power_neff(3000/d) for d in [1,1.5,2,3.45]}
# ---------- additional cases ----------
cases={}
def case_block(name,TP,FN,FP,TN,pt_override=None):
    mm=metrics(TP,FN,FP,TN); pt=(TP+FN)/mm['n']
    d=dict(TP=TP,FN=FN,FP=FP,TN=TN,n=mm['n'],pt=pt,acc=mm['acc'],prec=mm['prec'],rec=mm['rec'],f1=mm['f1'],tpr=mm['tpr'],fpr=mm['fpr'],
           ppv={str(pi):(mm['tpr']*pi/(mm['tpr']*pi+mm['fpr']*(1-pi))) for pi in [0.001,0.01,0.05]},
           fpr_max90={str(pi):mm['tpr']*pi*0.1/(0.9*(1-pi)) for pi in [0.001,0.01,0.05]},
           Astar90={str(pi):1-(1-pt)*pi*0.1/(0.9*(1-pi)) for pi in [0.001,0.01,0.05]},
           fpr_ci=dict(wilson=wilson(FP,FP+TN),cp=cp(FP,FP+TN),jeff=jeff(FP,FP+TN)),
           eq18_min={str(pi):pi/(pi+(1-pi)*(1-mm['acc'])/(1-pt)) for pi in [0.001,0.01,0.05]})
    # PPV at CP-upper FPR (conservative) with point TPR
    up=d['fpr_ci']['cp'][1]
    d['ppv_lower_cp']={str(pi):mm['tpr']*pi/(mm['tpr']*pi+up*(1-pi)) for pi in [0.001,0.01,0.05]}
    up=d['fpr_ci']['jeff'][1]
    d['ppv_lower_jeff']={str(pi):mm['tpr']*pi/(mm['tpr']*pi+up*(1-pi)) for pi in [0.001,0.01,0.05]}
    return d
a=I['caseA']; cases['A']=case_block('A',a['TP'],a['FN'],a['FP'],a['TN'])
b=I['caseB']; cases['B']=case_block('B',b['TP'],b['FN'],b['FP'],b['TN'])
cases['B']['sum_matrix']=b['TP']+b['FN']+b['FP']+b['TN']; cases['B']['stated_n']=b['stated_n']; cases['B']['printed']=b['printed']
c=I['caseC']; n_c=c['n']; V=c['valid_text']; P=n_c-V; FNc=int(round((1-c['rec']/100)*P)); TNc=V; FPc=0; TPc=P-FNc
cases['C']=case_block('C',TPc,FNc,FPc,TNc); cases['C']['derived_check']=dict(acc=(TPc+TNc)/n_c,rec=TPc/(TPc+FNc),printed_acc=c['acc'],printed_rec=c['rec'],FN_from_acc=(1-c['acc']/100)*n_c)
R['cases']=cases
R['env']=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,seed=I['seed'],reps=REPS)
json.dump(R,open('results.json','w'),indent=1,default=float)
print(R['env']); print('indep',[(r['n'],round(r['power'],3),round(r['rank_first'],3)) for r in ind])
print('paired n80',R['sim_paired_n80']); print('gap',R['equal_acc_gap'])
print('cov',cov); print('cluster',cl); print('cpow',R['cluster_power'])
for k,v in cases.items(): print(k,{kk:(vv if not isinstance(vv,dict) else {a:round(float(b),5) if not isinstance(b,(tuple,list)) else b for a,b in vv.items()}) for kk,vv in v.items() if kk in ['n','pt','acc','prec','rec','fpr','ppv','Astar90','ppv_lower_cp','fpr_ci','eq18_min','sum_matrix','derived_check','TP','FN']})
