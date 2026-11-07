import json, os, numpy as np, matplotlib
os.makedirs('figs_wide',exist_ok=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8,'axes.linewidth':0.7,'lines.linewidth':1.4,'figure.dpi':100,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,'mathtext.fontset':'stix'})
R=json.load(open('results.json')); I=json.load(open('inputs.json'))
C=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#444444']
W=6.7; Wc=3.23
A=I['caseA']; TP,FN,FP,TN=A['TP'],A['FN'],A['FP'],A['TN']; n=3000
t0=TP/(TP+FN); f0=FP/(FP+TN); pit=(TP+FN)/n
def tag(ax,s): ax.text(-0.14 if ax.get_figure().get_figwidth()<4 else -0.10,1.04,s,transform=ax.transAxes,fontweight='bold',fontsize=10)
def save(fig,name): fig.savefig(f'figs_wide/{name}.png',bbox_inches='tight',pad_inches=0.04); plt.close(fig)
# ---- Fig1: convention audit
fig,ax=plt.subplots(1,2,figsize=(W,2.45),gridspec_kw=dict(width_ratios=[1.45,1]))
a=R['audit']; mets=[('prec','Precision'),('rec','Recall'),('f1','F1'),('jac','Jaccard'),('kappa','Cohen\u2019s \u03ba')]
x=np.arange(len(mets)); w=0.36
dl=[abs(a[k]['legit_exact']-a[k]['printed']) for k,_ in mets]; dp=[abs(a[k]['phish_exact']-a[k]['printed']) for k,_ in mets]
ax[0].bar(x-w/2,[max(v,1e-4) for v in dl],w,color=C[0],label='Legitimate class positive')
ax[0].bar(x+w/2,[max(v,1e-4) for v in dp],w,color=C[1],label='Phishing class positive')
ax[0].set_yscale('log'); ax[0].set_xticks(x); ax[0].set_xticklabels([m for _,m in mets]); ax[0].set_ylabel('|recomputed \u2212 printed| (pp)')
ax[0].axhline(0.005,color='k',lw=0.6,ls=':'); ax[0].text(4.45,0.0058,'rounding to 2 d.p.',ha='right',va='bottom',fontsize=7.5)
ax[0].legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.02)); ax[0].set_ylim(1e-4,30); tag(ax[0],'(a)')
cm=np.array([[TP,FN],[FP,TN]]); ax[1].imshow(np.log10(cm+1),cmap='Blues',vmin=0,vmax=4.2)
for i in range(2):
    for j in range(2): ax[1].text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=10,color='white' if cm[i,j]>500 else 'black')
ax[1].set_xticks([0,1]); ax[1].set_xticklabels(['Pred. phishing','Pred. legitimate']); ax[1].set_yticks([0,1]); ax[1].set_yticklabels(['Phishing','Legitimate'],rotation=90,va='center')
for s in ax[1].spines.values(): s.set_visible(False)
ax[1].tick_params(length=0); tag(ax[1],'(b)'); save(fig,'fig1')
# ---- Fig2: intervals on error scale
iv=R['A_intervals']; items=[('fpr','False-positive rate'),('fnr','Miss rate'),('prec','1 \u2212 precision (phishing)'),('acc','1 \u2212 accuracy')]
fig,ax=plt.subplots(figsize=(Wc,2.55)); meth=[('wilson','Wilson',C[0],-0.22),('cp','Clopper\u2013Pearson',C[1],0),('jeff','Jeffreys',C[2],0.22)]
for i,(k,lab) in enumerate(items):
    d=iv[k]
    for m,ml,col,off in meth:
        lo,hi=d[m]; est=d['est']
        if k in ['prec','acc']: lo,hi,e=1-hi,1-lo,1-est
        elif k=='fnr': lo,hi,e=lo,hi,est
        elif k=='fpr': e=est
        e_=e
        ax.plot([max(lo,1e-5)*100,hi*100],[i+off,i+off],color=col,lw=2.2,solid_capstyle='butt',label=ml if i==0 else None)
    e=(1-iv[k]['est']) if k in ['prec','acc'] else iv[k]['est']; ax.plot(e*100,i,'k|',ms=9,mew=1.2)
ax.set_xscale('log'); ax.set_yticks(range(len(items))); ax.set_yticklabels([l for _,l in items]); ax.invert_yaxis(); ax.set_xlabel('Rate (%), log scale; black tick = point estimate')
ax.legend(frameon=False,loc='lower right'); save(fig,'fig2')
# ---- Fig3: baseline scenarios
sc=R['baseline_scen']; names=list(sc.keys())
fig,ax=plt.subplots(1,2,figsize=(W,2.55),gridspec_kw=dict(width_ratios=[1.2,1]))
from scipy.stats import norm
lo,hi=R['cnn_wilson']; ax[0].plot([lo*100,hi*100],[0,0],color=C[0],lw=3,solid_capstyle='butt'); ax[0].plot(99.3,0,'k|',ms=9,mew=1.3)
for i,nm in enumerate(names):
    l,h=sc[nm]['wilson3000']; ax[0].plot([l*100,h*100],[i+1,i+1],color=C[1],lw=3,solid_capstyle='butt'); ax[0].plot(100*(I['baselines'][nm]/100),i+1,'k|',ms=9,mew=1.3)
ax[0].set_yticks(range(len(names)+1)); ax[0].set_yticklabels(['CNN (n = 3,000)']+[nm.split(' (')[0]+'\n('+nm.split('(')[1] if '(' in nm else nm for nm in names]); ax[0].invert_yaxis()
ax[0].set_xlabel('Accuracy (%), Wilson 95% interval'); tag(ax[0],'(a)')
xs=[1000,3000,10000,1e5]
for i,nm in enumerate(names):
    ps=[max(sc[nm][k]['p'],1e-30) for k in ['1000','3000','10000','None']]
    ax[1].plot(xs,ps,'o-',color=C[i],ms=3.5,label=nm.split(' (')[0])
ax[1].axhline(0.05,color='k',ls=':',lw=0.8); ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_ylim(1e-30,3); ax[1].set_yticks([1e-30,1e-24,1e-18,1e-12,1e-6,1])
ax[1].set_xticks(xs); ax[1].set_xticklabels(['1,000','3,000','10,000','\u221e']); ax[1].set_xlabel('Assumed baseline sample size'); ax[1].set_ylabel('Two-sided p-value'); ax[1].legend(frameon=False,fontsize=7.2,loc='lower left'); tag(ax[1],'(b)')
save(fig,'fig3')
# ---- Fig4: PPV vs prevalence
pis=np.logspace(-3.3,np.log10(0.5),300)
ppv=lambda t,f,p: t*p/(t*p+f*(1-p))
fig,ax=plt.subplots(1,2,figsize=(W,2.6))
for fm,ls,col in [(0.5,'--',C[2]),(1,'-',C[0]),(2,'-.',C[1]),(4,':',C[3])]:
    ax[0].plot(pis*100,100*ppv(t0,f0*fm,pis),ls=ls,color=col,label=f'FPR \u00d7 {fm:g}' if fm!=1 else 'Observed FPR (1.30%)')
ks=sorted(R['A_ppv'].values(),key=lambda d:d['pi']); ks=[d for d in ks if d['pi']<0.3]
ax[0].fill_between([d['pi']*100 for d in ks],[100*d['ci'][0] for d in ks],[100*d['ci'][1] for d in ks],color=C[0],alpha=0.18,lw=0)
ax[0].axhline(90,color='k',lw=0.6,ls=':'); ax[0].set_xscale('log'); ax[0].set_xlabel('Deployment prevalence of phishing (%)'); ax[0].set_ylabel('Precision (%)'); ax[0].legend(frameon=False,loc='upper left'); ax[0].set_ylim(0,102); tag(ax[0],'(a)')
for t_,lab,col in [(1,'0.1%',C[2]),(1,'1%',C[0]),(1,'5%',C[1])]: pass
fm=np.logspace(-4.5,-1.3,300)
for pi_,col in [(0.001,C[2]),(0.01,C[0]),(0.05,C[1])]:
    ax[1].plot(fm*100,100*ppv(t0,fm,pi_),color=col,label=f'prevalence {pi_*100:g}%')
ax[1].axvline(f0*100,color='k',ls=':',lw=0.8); ax[1].text(f0*100*1.07,6,'observed\nFPR',fontsize=7.5); ax[1].axhline(90,color='k',lw=0.6,ls=':')
ax[1].set_xscale('log'); ax[1].set_xlabel('False-positive rate (%)'); ax[1].set_ylabel('Precision (%)'); ax[1].legend(frameon=False,loc='lower left'); ax[1].set_ylim(0,102); tag(ax[1],'(b)')
save(fig,'fig4')
# ---- Fig5: ROC sensitivity
S=R['roc_sens']; use=[('EV-AUC','Equal-var. (AUC)',C[0],'-'),('EV-point','Equal-var. (point)',C[2],'--'),('UV-1','Unequal-var., b=0.37',C[1],'-.'),('UV-2','Unequal-var., b=1.16',C[3],':'),('Logistic','Logistic',C[4],'-')]
d_auc=R['dprime']['auc']; d_pt=R['dprime']['point']
def mk(name):
    P=S[name]['param']
    if name=='EV-AUC': return lambda f: norm.cdf(P['dprime']+norm.ppf(f))
    if name=='EV-point': return lambda f: norm.cdf(P['dprime']+norm.ppf(f))
    if name.startswith('UV'): return lambda f,P=P: norm.cdf(P['a']+P['b']*norm.ppf(f))
    if name=='Logistic':
        mu=P['mu']; return lambda f: 1-1/(1+np.exp(-(np.log((1-f)/f)-mu)))
def cost(t,f,k,pi): return 1e4*(k*pi*(1-t)+(1-pi)*f)
fg=np.logspace(-4.3,-1.5,400)
fig,ax=plt.subplots(1,3,figsize=(W,2.6))
for nm,lab,col,ls in use:
    r=mk(nm); ax[0].plot(fg*100,100*(1-r(fg)),color=col,ls=ls,label=lab)
    for j,k in enumerate([10,100]): ax[j+1].plot(fg*100,[cost(r(f),f,k,0.01) for f in fg],color=col,ls=ls)
ax[0].plot(f0*100,100*(1-t0),'ko',ms=4); ax[0].errorbar([f0*100],[100*(1-t0)],xerr=[[ (f0-R['A_intervals']['fpr']['cp'][0])*100],[(R['A_intervals']['fpr']['cp'][1]-f0)*100]],fmt='none',ecolor='k',lw=0.8)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('False-positive rate (%)'); ax[0].set_ylabel('Miss rate (%)'); ax[0].set_ylim(1e-4,50); ax[0].legend(frameon=False,fontsize=6.6,loc='lower left'); tag(ax[0],'(a)')
for j,(k,lab) in enumerate([(10,'(b)'),(100,'(c)')]):
    a_=ax[j+1]; a_.set_xscale('log'); a_.set_yscale('log'); a_.set_xlabel('False-positive rate (%)'); a_.set_ylabel(f'Cost per 10,000 ({"k = "+str(k)}, 1%)'); tag(a_,lab)
    a_.axvline(f0*100,color='k',ls=':',lw=0.8)
fig.tight_layout(w_pad=1.2); save(fig,'fig5')
# ---- Fig7: prior-shift thresholds
fig,ax=plt.subplots(figsize=(Wc,2.6))
pp=np.logspace(-3.3,np.log10(0.45),300)
for k,col in [(10,C[0]),(100,C[1])]:
    o=(1/k)*(pit/(1-pit))/(pp/(1-pp)); ax.plot(pp*100,o/(1+o),color=col,label=f'k = {k}')
lo_,hi_=A['threshold_curve']['plateau1pct']; ax.axhspan(lo_,hi_,color='0.85',lw=0); ax.axhline(A['threshold_curve']['argmax'],color='k',ls=':',lw=0.8)
ax.text(0.14,0.30,'digitised plateau\n(within 1% of max P\u00d7R)',fontsize=7.5); ax.text(0.14,0.025,'argmax of P\u00d7R, 0.065',fontsize=7.5)
ax.set_xscale('log'); ax.set_xlabel('Deployment prevalence (%)'); ax.set_ylabel('Threshold on evaluation-prevalence scale'); ax.set_ylim(0,1.02); ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0.55,0.62)); save(fig,'fig7')
# ---- Fig8: accuracy vs precision bounds
fig,ax=plt.subplots(1,2,figsize=(W,2.6))
Ag=np.linspace(0.9,1,500)
for pt_,col,ls in [(0.4877,C[0],'-'),(0.2,C[1],'--'),(0.05,C[2],'-.')]:
    pi=0.01; AA=Ag[Ag>=max(pt_,1-pt_)]
    mn=pi/(pi+(1-pi)*(1-AA)/(1-pt_)); ax[0].plot(AA*100,100*mn,color=col,ls=ls,label=f'evaluation prevalence {pt_*100:.4g}%')
cs=R['cases']
for k,(m,col) in {'A':('o',C[3]),'B':('s',C[4]),'C':('^',C[5])}.items():
    ax[0].plot(100*cs[k]['acc'],100*cs[k]['eq18_min']['0.01'],m,color='k',mfc=col,ms=6,label=f'Case {k}')
ax[0].set_xlabel('Accuracy (%)'); ax[0].set_ylabel('Guaranteed minimum precision (%)\nat 1% prevalence'); ax[0].legend(frameon=False,fontsize=7,loc='upper left'); ax[0].set_xlim(90,100.1); tag(ax[0],'(a)')
pg=np.logspace(-3.3,np.log10(0.3),200)
for pt_,col,ls in [(0.4877,C[0],'-'),(0.2,C[1],'--'),(0.05,C[2],'-.')]:
    As=1-(1-pt_)*pg*0.1/(0.9*(1-pg)); ax[1].plot(pg*100,100*As,color=col,ls=ls,label=f'{pt_*100:.4g}%')
ax[1].set_xscale('log'); ax[1].set_xlabel('Deployment prevalence (%)'); ax[1].set_ylabel('Accuracy needed to guarantee\n90% precision (%)'); ax[1].legend(frameon=False,title='Evaluation prevalence',title_fontsize=7.5,fontsize=7.5,loc='lower right'); tag(ax[1],'(b)')
fig.tight_layout(w_pad=1.4); save(fig,'fig8')
# ---- Fig9: simulations
fig,ax=plt.subplots(1,2,figsize=(W,2.6))
ind=R['sim_independent']; ax[0].plot([r['n'] for r in ind],[100*r['power'] for r in ind],'o-',color=C[0],ms=3,label='Independent test sets (z-test)')
for lab,col,ls in [('0.5%',C[2],'--'),('1%',C[1],'-.')]:
    rows=R['sim_paired'][lab]['rows']; ax[0].plot([r['n'] for r in rows],[100*r['power_exact'] for r in rows],ls=ls,color=col,marker='s',ms=3,label=f'Paired, exact McNemar, discordance {lab}')
ax[0].axhline(80,color='k',ls=':',lw=0.7); ax[0].set_xscale('log'); ax[0].set_xlabel('Test-set size per detector'); ax[0].set_ylabel('Power (%) at 99.3% vs 99.0%'); ax[0].legend(frameon=False,fontsize=7,loc='lower right'); ax[0].set_ylim(0,102); tag(ax[0],'(a)')
cl=R['cluster_cov']
for csize,ls,mk_ in [(10,'-','o'),(50,'--','s')]:
    rhos=[0.0]+[0.02,0.05,0.10]; 
    for meth,col in [('wilson',C[0]),('cp',C[1]),('jeff',C[2])]:
        ys=[100*cl['1|0.0'][meth]]+[100*cl[f'{csize}|{r}'][meth] for r in rhos[1:]]
        ax[1].plot(rhos,ys,ls=ls,color=col,marker=mk_,ms=3,label=f'{ {"wilson":"Wilson","cp":"Clopper\u2013Pearson","jeff":"Jeffreys"}[meth]}, cluster {csize}' if True else None)
ax[1].axhline(95,color='k',ls=':',lw=0.7); ax[1].set_xlabel('Intra-cluster correlation \u03c1'); ax[1].set_ylabel('Actual coverage of nominal 95% (%)'); ax[1].legend(frameon=False,fontsize=6.3,ncol=1,loc='lower left'); ax[1].set_ylim(50,100); tag(ax[1],'(b)')
save(fig,'fig9')
# ---- Fig10: case series
fig,ax=plt.subplots(figsize=(Wc,2.6))
for k,col,ls in [('A',C[0],'-'),('B',C[1],'--'),('C',C[2],'-.')]:
    c=cs[k]; ax.plot(pis*100,100*ppv(c['tpr'],c['fpr'],pis),color=col,ls=ls,label=f'Case {k}'+(' (FP = 0; point est.)' if k=='C' else ''))
c=cs['C']; up=c['fpr_ci']['cp'][1]; ax.plot(pis*100,100*ppv(c['tpr'],up,pis),color=C[2],ls=':',label='Case C, 95% upper FPR')
ax.axhline(90,color='k',lw=0.6,ls=':'); ax.set_xscale('log'); ax.set_xlabel('Deployment prevalence (%)'); ax.set_ylabel('Precision (%)'); ax.set_ylim(0,102); ax.legend(frameon=False,fontsize=7.2,loc='lower right'); save(fig,'fig10')
print('ok')
