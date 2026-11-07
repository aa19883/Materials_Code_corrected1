import json, numpy as np
from scipy.stats import norm, beta, binom, chi2
from scipy.optimize import brentq, minimize_scalar
exec(open('core.py').read().split("mp=metrics")[0])  # helpers + inputs
R=json.load(open('results_core.json'))
TP,FN,FP,TN=A['TP'],A['FN'],A['FP'],A['TN']; n=TP+FN+FP+TN
t0=TP/(TP+FN); f0=FP/(FP+TN); pit=(TP+FN)/n; AUC=A['auc']['LR3']
def cost(t,f,k,pi): return 1e4*(k*pi*(1-t)+(1-pi)*f)
def ppv(t,f,pi): return t*pi/(t*pi+f*(1-pi))
# ---------- ROC families ----------
zt,zf=norm.ppf(t0),norm.ppf(f0)
fam={}
d_auc=np.sqrt(2)*norm.ppf(AUC)
fam['EV-AUC']=dict(label='Equal-variance binormal, d\u2032 from stated AUC',roc=lambda f,d=d_auc: norm.cdf(d+norm.ppf(f)),param=dict(dprime=d_auc))
d_pt=zt-zf
fam['EV-point']=dict(label='Equal-variance binormal through observed point',roc=lambda f,d=d_pt: norm.cdf(d+norm.ppf(f)),param=dict(dprime=d_pt))
# unequal variance through point and AUC: a+b*zf=zt ; a/sqrt(1+b^2)=Phi^-1(AUC)
za=norm.ppf(AUC)
g=lambda b: (zt-b*zf)/np.sqrt(1+b*b)-za
bs=[]
for lo,hi in [(0.05,0.9),(0.9,3)]:
    try: bs.append(brentq(g,lo,hi))
    except Exception as e: pass
for i,b in enumerate(bs):
    a=zt-b*zf
    fam[f'UV-{i+1}']=dict(label=f'Unequal-variance binormal (slope b={b:.2f}) through point and AUC',roc=lambda f,a=a,b=b: norm.cdf(a+b*norm.ppf(f)),param=dict(a=a,b=b))
# logistic location-shift matched to AUC
from scipy.integrate import quad
def auc_logistic(mu):
    # neg ~ Logistic(0,1), pos ~ Logistic(mu,1); AUC = P(pos>neg)
    f=lambda x: (1/(1+np.exp(-(x-mu))))*0+ np.exp(-(x))/ (1+np.exp(-x))**2 * (1-1/(1+np.exp(-(x-mu))))
    # P(neg>=x)... easier: AUC = int f_neg(x) F_pos(x)?? P(pos>neg)=int f_neg(x) (1-F_pos(x)) dx
    v=quad(lambda x: np.exp(-x)/(1+np.exp(-x))**2*(1-1/(1+np.exp(-(x-mu)))),-60,60,limit=200)[0]
    return v
mu=brentq(lambda m: auc_logistic(m)-AUC,3,30)
def roc_log(f,mu=mu):
    c=np.log((1-f)/f)  # threshold such that FPR=f for neg: 1-F=1/(1+e^{c})=f => c=ln((1-f)/f)
    return 1-1/(1+np.exp(-(c-mu)))
fam['Logistic']=dict(label='Logistic location-shift (heavier tails) matched to AUC',roc=roc_log,param=dict(mu=mu))
# AUC uncertainty for EV-AUC family
for a_ in [0.9998,0.99997]:
    d=np.sqrt(2)*norm.ppf(a_); fam[f'EV-AUC={a_}']=dict(label=f'Equal-variance binormal, AUC={a_}',roc=lambda f,d=d: norm.cdf(d+norm.ppf(f)),param=dict(dprime=d))
def opt(roc,k,pi):
    r=minimize_scalar(lambda lf: cost(roc(np.exp(lf)),np.exp(lf),k,pi),bounds=(np.log(1e-7),np.log(0.2)),method='bounded',options=dict(xatol=1e-9))
    f=float(np.exp(r.x)); return f,float(roc(f)),float(r.fun)
out={}
for name,F in fam.items():
    roc=F['roc']; row=dict(label=F['label'],param={k:float(v) for k,v in F['param'].items()})
    row['tpr_at_obs_fpr']=float(roc(f0))
    row['tpr_at']={str(f):float(roc(f)) for f in [0.005,0.002,0.001,0.0005]}
    row['cost']={}
    for k in [10,100]:
        row['cost'][str(k)]={}
        for f in [f0,0.005,0.002,0.001,0.0005]:
            row['cost'][str(k)][f'{f:.6f}']=cost(roc(f),f,k,0.01)
        for pi in [0.001,0.01,0.05]:
            fo,to,co=opt(roc,k,pi)
            c_obs_model=cost(roc(f0),f0,k,pi); c_obs=cost(t0,f0,k,pi)
            row['cost'][str(k)][f'opt_pi{pi}']=dict(fpr=fo,tpr=to,cost=co,cost_model_at_obs=c_obs_model,cost_obs=c_obs,saving_vs_model=1-co/c_obs_model,saving_vs_obs=1-co/c_obs, ppv=ppv(to,fo,pi))
    out[name]=row
R['roc_sens']=out
# source-point checks
R['dprime']=dict(auc=d_auc,point=d_pt,tpr_model_at_obs=float(fam['EV-AUC']['roc'](f0)),expected_misses_model=float((1-fam['EV-AUC']['roc'](f0))*(TP+FN)))
# ---------- prior shift ----------
def tau_test(pi,k,pit=pit):
    o=(1/k)*(pit/(1-pit))/(pi/(1-pi)); return o/(1+o)
R['prior_shift']={f'{pi}|{k}':tau_test(pi,k) for pi in [0.001,0.01,0.05,pit] for k in [10,100]}
tau_src=A['threshold_curve']['argmax']; lo,hi=A['threshold_curve']['plateau1pct']
def implied_k_at(tau_t,pi):  # scores calibrated at test prevalence; deployment prevalence pi
    o_t=tau_t/(1-tau_t); o_dep=o_t*(pi/(1-pi))/(pit/(1-pit)); return 1/o_dep
R['implied_k']={f'{pi}':dict(src=implied_k_at(tau_src,pi),lo=implied_k_at(lo,pi),hi=implied_k_at(hi,pi)) for pi in [pit,0.05,0.01,0.001]}
# ---------- Eq 18/19 domain ----------
def seg_bounds(Acc,pt,pi,npts=200001):
    E=1-Acc
    flo=max(0,(E-pt)/(1-pt)); fhi=min(1,E/(1-pt))
    f=np.linspace(flo,fhi,npts); t=1-(E-(1-pt)*f)/pt
    ok=(t>=-1e-12)&(t<=1+1e-12)
    f=f[ok]; t=np.clip(t[ok],0,1)
    with np.errstate(all='ignore'):
        p=np.where(t*pi+f*(1-pi)>0, t*pi/(t*pi+f*(1-pi)), np.nan)
    return np.nanmin(p),np.nanmax(p),flo,fhi
Acc=(TP+TN)/n
mn,mx,flo,fhi=seg_bounds(Acc,pit,0.01)
eq18=lambda Acc,pt,pi: pi/(pi+(1-pi)*(1-Acc)/(1-pt))
R['eq18']=dict(A=Acc,domain_min=max(pit,1-pit),min_num=mn,min_eq=eq18(Acc,pit,0.01),max_num=mx,flo=flo,fhi=fhi)
R['eq18_by_pi']={str(pi):dict(minppv=eq18(Acc,pit,pi),obs=ppv(t0,f0,pi)) for pi in [0.001,0.01,0.05]}
Astar=lambda pt,pi,target: 1-(1-pt)*pi*(1-target)/(target*(1-pi))
R['Astar']={str(pi):Astar(pit,pi,0.9) for pi in [0.001,0.01,0.05]}
# boundary demos (pt=0.4877): outside-domain example
demo=[]
for pt_,Acc_ in [(0.4877,0.993),(0.05,0.97),(0.05,0.90),(0.9,0.5)]:
    mn_,mx_,flo_,fhi_=seg_bounds(Acc_,pt_,0.01,20001); demo.append(dict(pt=pt_,A=Acc_,in_domain=Acc_>=max(pt_,1-pt_),min=float(mn_),max=float(mx_),eq18=float(eq18(Acc_,pt_,0.01))))
R['eq18_demo']=demo
# ---------- baseline scenarios ----------
bl={k:v for k,v in I['baselines'].items() if k!='status'}
def ztest(p1,n1,p2,n2):
    se=np.sqrt(p1*(1-p1)/n1+(p2*(1-p2)/n2 if n2 else 0)); z=(p1-p2)/se
    return float(z),float(2*norm.sf(abs(z)))
p1=Acc
sc={}
for name,pb in bl.items():
    pb/=100; sc[name]={}
    for nb in [1000,3000,10000,None]:
        z,p=ztest(p1,3000,pb,nb); sc[name][str(nb)]=dict(z=z,p=p)
    # intervals with fractional counts at n=3000
    sc[name]['wilson3000']=wilson(pb*3000,3000)
    sc[name]['diff']=p1-pb
    d=p1-pb; za_,zb_=1.959964,0.841621
    sc[name]['n80']=float((za_+zb_)**2*(p1*(1-p1)+pb*(1-pb))/d**2)
    sc[name]['n3000_counts']=pb*3000
R['baseline_scen']=sc; R['cnn_wilson']=wilson(TP+TN,n)
json.dump(R,open('results_core.json','w'),indent=1,default=float)
print('dprime',R['dprime']); print('UV b', bs, 'logistic mu',mu)
for k,v in out.items():
    print(k, 'TPR@obsFPR',round(100*v['tpr_at_obs_fpr'],3),'TPR@0.1%',round(100*v['tpr_at']['0.001'],3))
    for kk in ['10','100']:
        o=v['cost'][kk]['opt_pi0.01']; print('  k',kk,'opt fpr %.4f%% tpr %.3f%% cost %.2f sav_model %.1f%% sav_obs %.1f%%'%(100*o['fpr'],100*o['tpr'],o['cost'],100*o['saving_vs_model'],100*o['saving_vs_obs']))
print(R['prior_shift']); print(R['implied_k']); print(R['eq18'],R['eq18_demo'],R['Astar'])
for k,v in sc.items(): print(k,{a:(round(b['p'],3) if isinstance(b,dict) else b) for a,b in v.items() if a in ['1000','3000','10000','None']},round(v['n80']))
