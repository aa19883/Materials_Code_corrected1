import json, numpy as np, scipy, sys
from scipy.stats import norm, beta, binom, chi2
I=json.load(open('inputs.json'))
R={}
A=I['caseA']; TP,FN,FP,TN=A['TP'],A['FN'],A['FP'],A['TN']; n=TP+FN+FP+TN
def wilson(k,m,z=1.959964):
    p=k/m; c=(p+z*z/(2*m))/(1+z*z/m); h=z*np.sqrt(p*(1-p)/m+z*z/(4*m*m))/(1+z*z/m); return c-h,c+h
def cp(k,m,a=0.05):
    lo=0. if k==0 else beta.ppf(a/2,k,m-k+1); hi=1. if k==m else beta.ppf(1-a/2,k+1,m-k); return float(lo),float(hi)
def jeff(k,m,a=0.05):
    lo=0. if k==0 else beta.ppf(a/2,k+.5,m-k+.5); hi=1. if k==m else beta.ppf(1-a/2,k+.5,m-k+.5); return float(lo),float(hi)
def metrics(TP,FN,FP,TN):
    n=TP+FN+FP+TN
    acc=(TP+TN)/n; prec=TP/(TP+FP); rec=TP/(TP+FN); f1=2*TP/(2*TP+FP+FN); jac=TP/(TP+FP+FN)
    po=acc; pe=((TP+FN)*(TP+FP)+(FP+TN)*(FN+TN))/n**2; kap=(po-pe)/(1-pe)
    mcc=(TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)); tpr=rec; tnr=TN/(TN+FP); fpr=FP/(FP+TN)
    return dict(n=n,acc=acc,prec=prec,rec=rec,f1=f1,jac=jac,kappa=kap,mcc=mcc,tpr=tpr,tnr=tnr,fpr=fpr,fnr=FN/(FN+TP),youden=tpr+tnr-1,
                lrp=tpr/fpr if fpr>0 else np.inf, balacc=(tpr+tnr)/2)
def swap(TP,FN,FP,TN): return TN,FP,FN,TP
def rnd(x,d=2): return float(np.round(x*100,d))
def trunc(x,d=2): return float(np.floor(x*100*10**d+1e-9)/10**d)
mp=metrics(TP,FN,FP,TN); ml=metrics(*swap(TP,FN,FP,TN))
R['A_phish']=mp; R['A_legit']=ml
pr=A['printed']
aud={}
for key,val,name in [('prec',pr['prec'],'prec'),('rec',pr['rec'],'rec'),('f1',pr['f1'],'f1'),('jac',pr['jac'],'jac'),('kappa',pr['kappa'],'kappa')]:
    x=ml[name]; xp=mp[name]
    aud[key]=dict(printed=val,legit_exact=100*x,legit_round=rnd(x),legit_trunc=trunc(x),phish_exact=100*xp,phish_round=rnd(xp),phish_trunc=trunc(xp))
aud['FPV']=dict(printed=pr['FPV'],exact=FN/(FN+TP),round6=round(FN/(FN+TP),6),trunc6=np.floor(FN/(FN+TP)*1e6)/1e6)
aud['FNR']=dict(printed=pr['FNR'],exact=FP/(FP+TN),round6=round(FP/(FP+TN),6),trunc6=np.floor(FP/(FP+TN)*1e6)/1e6)
R['audit']=aud
# intervals
iv={}
cells={'acc':(TP+TN,n),'rec':(TP,TP+FN),'prec':(TP,TP+FP),'tnr':(TN,TN+FP),'fpr':(FP,FP+TN),'fnr':(FN,FN+TP)}
for k,(a,m) in cells.items(): iv[k]=dict(k=a,m=m,est=a/m,wilson=wilson(a,m),cp=cp(a,m),jeff=jeff(a,m))
R['A_intervals']=iv
# Dirichlet draws
rng=np.random.default_rng(I['seed'])
D=rng.dirichlet([TP+.5,FN+.5,FP+.5,TN+.5],size=200000)
tp,fn,fp,tn=D.T
def q(x): return [float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]
f1=2*tp/(2*tp+fp+fn); mcc=(tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
R['A_composite_ci']=dict(f1=q(f1),mcc=q(mcc),youden=q(tp/(tp+fn)+tn/(tn+fp)-1))
tpr_d=tp/(tp+fn); fpr_d=fp/(fp+tn)
def ppv(t,f,pi): return t*pi/(t*pi+f*(1-pi))
R['A_ppv']={}
for pi in [0.001,0.005,0.01,0.02,0.05,0.10,0.25,TP_prev:=(TP+FN)/n]:
    t0=TP/(TP+FN); f0=FP/(FP+TN)
    R['A_ppv'][f'{pi:.6f}']=dict(pi=pi,ppv=ppv(t0,f0,pi),ci=q(ppv(tpr_d,fpr_d,pi)),alerts=1e4*(t0*pi+f0*(1-pi)),true=1e4*t0*pi,fa=1e4*f0*(1-pi),miss=1e4*(1-t0)*pi)
t0=TP/(TP+FN); f0=FP/(FP+TN); pit=(TP+FN)/n
R['pit']=pit
def fpr_max(t,pi,target): return t*pi*(1-target)/(target*(1-pi))
R['fpr_max']={f'{pi}':fpr_max(t0,pi,0.9) for pi in [0.001,0.01,0.05]}
R['pi_star_half']=f0/(t0*(1-0.5)/0.5+0.5*f0) if False else None
# break-even prevalence for ppv target
def pi_star(t,f,target): return target*f/(t*(1-target)+target*f)
R['pi_star']={str(tg):pi_star(t0,f0,tg) for tg in [0.5,0.9,0.99]}
R['kstar']={str(pi):(1-pi)/pi*t0/t0*f0/t0 if False else (1-pi)/pi*(t0/f0)**-1*1 for pi in []}
# break-even k: cost of filtering = k*pi*(1-t)+(1-pi)f < k*pi  => k > (1-pi)f/(pi t)
R['kstar']={str(pi):(1-pi)*f0/(pi*t0) for pi in [0.001,0.01,0.05]}
# scenario grid: FPR multiplier x TPR
R['scen']=[]
for fm in [0.5,1,2,4]:
    row=[]
    for tp_ in [t0,0.99,0.98,0.95]:
        row.append(ppv(tp_,f0*fm,0.01))
    R['scen'].append(dict(fm=fm,fpr=f0*fm,ppv=row))
R['scen_tpr']=[t0,0.99,0.98,0.95]
json.dump(R,open('results_core.json','w'),indent=1,default=float)
print(json.dumps(R['audit'],indent=1,default=float)[:3500])
print(R['A_intervals']['rec']); print(R['A_composite_ci']); print(R['fpr_max'],R['pi_star'],R['kstar'])
for k,v in R['A_ppv'].items(): print(k,round(v['ppv']*100,1),[round(x*100,1) for x in v['ci']],round(v['alerts']),round(v['true'],1),round(v['fa']),round(v['miss'],2))
for s in R['scen']: print(s['fm'],round(s['fpr']*100,3),[round(x*100,1) for x in s['ppv']])
