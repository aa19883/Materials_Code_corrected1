# Single-column (3.23 in) versions of the multi-panel figures: panels stacked vertically, >= 8 pt text at final size.
import json, os, shutil, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams.update({'font.family':'Liberation Serif','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,'axes.linewidth':0.7,'lines.linewidth':1.3,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,'mathtext.fontset':'stix'})
R=json.load(open('results.json')); I=json.load(open('inputs.json'))
C=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#444444']; Wc=3.23
A=I['caseA']; TP,FN,FP,TN=A['TP'],A['FN'],A['FP'],A['TN']; t0=TP/(TP+FN); f0=FP/(FP+TN); pit=(TP+FN)/3000
os.makedirs('figs',exist_ok=True)
for f in ('fig2','fig7','fig10'): shutil.copy(f'figs_wide/{f}.png',f'figs/{f}.png')
def save(fig,name): fig.savefig(f'figs/{name}.png',bbox_inches='tight',pad_inches=0.04); plt.close(fig)
def tag(ax,s): ax.text(-0.17,1.02,s,transform=ax.transAxes,fontweight='bold',fontsize=9.5)
ppv=lambda t,f,p: t*p/(t*p+f*(1-p))
cs=R['cases']
# Fig1
fig,ax=plt.subplots(2,1,figsize=(Wc,4.6),gridspec_kw=dict(height_ratios=[1.15,1]))
a=R['audit']; mets=[('prec','Precision'),('rec','Recall'),('f1','F1'),('jac','Jaccard'),('kappa','Kappa')]
x=np.arange(5); w=0.38
ax[0].bar(x-w/2,[max(abs(a[k]['legit_exact']-a[k]['printed']),1e-4) for k,_ in mets],w,color=C[0],label='Legitimate = positive')
ax[0].bar(x+w/2,[max(abs(a[k]['phish_exact']-a[k]['printed']),1e-4) for k,_ in mets],w,color=C[1],label='Phishing = positive')
ax[0].set_yscale('log'); ax[0].set_xticks(x); ax[0].set_xticklabels([m for _,m in mets]); ax[0].set_ylabel('|recomputed \u2212 printed| (pp)'); ax[0].set_ylim(1e-4,300); ax[0].legend(frameon=False,loc='upper left'); tag(ax[0],'(a)')
cm=np.array([[TP,FN],[FP,TN]]); ax[1].imshow(np.log10(cm+1),cmap='Blues',vmin=0,vmax=4.2)
for i in range(2):
    for j in range(2): ax[1].text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=10,color='white' if cm[i,j]>500 else 'black')
ax[1].set_xticks([0,1]); ax[1].set_xticklabels(['Pred. phishing','Pred. legitimate']); ax[1].set_yticks([0,1]); ax[1].set_yticklabels(['Phishing','Legitimate'],rotation=90,va='center')
for s in ax[1].spines.values(): s.set_visible(False)
ax[1].tick_params(length=0); tag(ax[1],'(b)'); fig.tight_layout(h_pad=1.0); save(fig,'fig1')
# Fig3
sc=R['baseline_scen']; names=list(sc.keys()); short=['Decision tree','LR+SVM+DT','LSTM','KNN']
fig,ax=plt.subplots(2,1,figsize=(Wc,4.9))
lo,hi=R['cnn_wilson']; ax[0].plot([lo*100,hi*100],[0,0],color=C[0],lw=3,solid_capstyle='butt'); ax[0].plot(99.3,0,'k|',ms=8,mew=1.2)
for i,nm in enumerate(names):
    l,h=sc[nm]['wilson3000']; ax[0].plot([l*100,h*100],[i+1,i+1],color=C[1],lw=3,solid_capstyle='butt'); ax[0].plot(I['baselines'][nm],i+1,'k|',ms=8,mew=1.2)
ax[0].set_yticks(range(5)); ax[0].set_yticklabels(['CNN']+short); ax[0].invert_yaxis(); ax[0].set_xlabel('Accuracy (%), Wilson 95% interval'); tag(ax[0],'(a)')
xs=[1000,3000,10000,1e5]
for i,nm in enumerate(names):
    ps=[max(sc[nm][k]['p'],1e-30) for k in ['1000','3000','10000','None']]; ax[1].plot(xs,ps,'o-',color=C[i],ms=3,label=short[i])
ax[1].axhline(0.05,color='k',ls=':',lw=0.8); ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_ylim(1e-30,3); ax[1].set_yticks([1e-30,1e-24,1e-18,1e-12,1e-6,1])
ax[1].set_xticks(xs); ax[1].set_xticklabels(['1,000','3,000','10,000','\u221e']); ax[1].set_xlabel('Assumed baseline sample size'); ax[1].set_ylabel('Two-sided p-value'); ax[1].legend(frameon=False,loc='lower left',fontsize=7.5); tag(ax[1],'(b)')
fig.tight_layout(h_pad=1.0); save(fig,'fig3')
# Fig4
pis=np.logspace(-3.3,np.log10(0.5),300)
fig,ax=plt.subplots(2,1,figsize=(Wc,4.8))
for fm_,ls,col in [(0.5,'--',C[2]),(1,'-',C[0]),(2,'-.',C[1]),(4,':',C[3])]:
    ax[0].plot(pis*100,100*ppv(t0,f0*fm_,pis),ls=ls,color=col,label=f'FPR \u00d7 {fm_:g}' if fm_!=1 else 'Observed FPR')
ks=sorted([d for d in R['A_ppv'].values() if d['pi']<0.3],key=lambda d:d['pi'])
ax[0].fill_between([d['pi']*100 for d in ks],[100*d['ci'][0] for d in ks],[100*d['ci'][1] for d in ks],color=C[0],alpha=0.18,lw=0)
ax[0].axhline(90,color='k',lw=0.6,ls=':'); ax[0].set_xscale('log'); ax[0].set_xlabel('Phishing prevalence (%)'); ax[0].set_ylabel('Precision (%)'); ax[0].legend(frameon=False,loc='upper left'); ax[0].set_ylim(0,102); tag(ax[0],'(a)')
fm=np.logspace(-4.5,-1.3,300)
for pi_,col in [(0.001,C[2]),(0.01,C[0]),(0.05,C[1])]: ax[1].plot(fm*100,100*ppv(t0,fm,pi_),color=col,label=f'prevalence {pi_*100:g}%')
ax[1].axvline(f0*100,color='k',ls=':',lw=0.8); ax[1].text(f0*100*1.1,55,'observed\nFPR',fontsize=7.5); ax[1].axhline(90,color='k',lw=0.6,ls=':')
ax[1].set_xscale('log'); ax[1].set_xlabel('False-positive rate (%)'); ax[1].set_ylabel('Precision (%)'); ax[1].legend(frameon=False,loc='lower left'); ax[1].set_ylim(0,102); tag(ax[1],'(b)')
fig.tight_layout(h_pad=1.0); save(fig,'fig4')
# Fig5
S=R['roc_sens']; use=[('EV-AUC','Equal-var. (AUC)',C[0],'-'),('EV-point','Equal-var. (point)',C[2],'--'),('UV-1','Unequal-var., b = 0.37',C[1],'-.'),('UV-2','Unequal-var., b = 1.16',C[3],':'),('Logistic','Logistic',C[4],'-')]
def mk(name):
    P=S[name]['param']
    if name in('EV-AUC','EV-point'): return lambda f: norm.cdf(P['dprime']+norm.ppf(f))
    if name.startswith('UV'): return lambda f,P=P: norm.cdf(P['a']+P['b']*norm.ppf(f))
    mu=P['mu']; return lambda f: 1-1/(1+np.exp(-(np.log((1-f)/f)-mu)))
cost=lambda t,f,k,pi: 1e4*(k*pi*(1-t)+(1-pi)*f)
fg=np.logspace(-4.3,-1.5,400)
fig,ax=plt.subplots(3,1,figsize=(Wc,6.6))
for nm,lab,col,ls in use:
    r=mk(nm); ax[0].plot(fg*100,100*(1-r(fg)),color=col,ls=ls,label=lab)
    for j,k in enumerate([10,100]): ax[j+1].plot(fg*100,[cost(r(f),f,k,0.01) for f in fg],color=col,ls=ls)
ax[0].plot(f0*100,100*(1-t0),'ko',ms=4); cp_=R['A_intervals']['fpr']['cp']
ax[0].errorbar([f0*100],[100*(1-t0)],xerr=[[(f0-cp_[0])*100],[(cp_[1]-f0)*100]],fmt='none',ecolor='k',lw=0.8)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('False-positive rate (%)'); ax[0].set_ylabel('Miss rate (%)'); ax[0].set_ylim(1e-4,50); ax[0].legend(frameon=False,fontsize=7.2,loc='lower left'); tag(ax[0],'(a)')
for j,(k,lab) in enumerate([(10,'(b)'),(100,'(c)')]):
    a_=ax[j+1]; a_.set_xscale('log'); a_.set_yscale('log'); a_.set_xlabel('False-positive rate (%)'); a_.set_ylabel(f'Cost per 10,000 (k = {k})'); tag(a_,lab); a_.axvline(f0*100,color='k',ls=':',lw=0.8)
fig.tight_layout(h_pad=1.0); save(fig,'fig5')
# Fig8
fig,ax=plt.subplots(2,1,figsize=(Wc,4.9))
Ag=np.linspace(0.9,1,500)
for pt_,col,ls in [(0.4877,C[0],'-'),(0.2,C[1],'--'),(0.05,C[2],'-.')]:
    AA=Ag[Ag>=max(pt_,1-pt_)]; mn=0.01/(0.01+0.99*(1-AA)/(1-pt_)); ax[0].plot(AA*100,100*mn,color=col,ls=ls,label=f'evaluation prevalence {pt_*100:.4g}%')
for k,(m,col) in {'A':('o',C[3]),'B':('s',C[4]),'C':('^',C[5])}.items(): ax[0].plot(100*cs[k]['acc'],100*cs[k]['eq18_min']['0.01'],m,color='k',mfc=col,ms=6,label=f'Case {k}')
ax[0].set_xlabel('Accuracy (%)'); ax[0].set_ylabel('Guaranteed minimum precision\n(%) at 1% prevalence'); ax[0].legend(frameon=False,fontsize=7,loc='upper left'); ax[0].set_xlim(90,100.1); tag(ax[0],'(a)')
pg=np.logspace(-3.3,np.log10(0.3),200)
for pt_,col,ls in [(0.4877,C[0],'-'),(0.2,C[1],'--'),(0.05,C[2],'-.')]:
    As=1-(1-pt_)*pg*0.1/(0.9*(1-pg)); ax[1].plot(pg*100,100*As,color=col,ls=ls,label=f'{pt_*100:.4g}%')
ax[1].set_xscale('log'); ax[1].set_xlabel('Deployment prevalence (%)'); ax[1].set_ylabel('Accuracy needed for\n90% precision (%)'); ax[1].legend(frameon=False,title='Evaluation prevalence',title_fontsize=7.5,fontsize=7.5,loc='lower left'); tag(ax[1],'(b)')
fig.tight_layout(h_pad=1.0); save(fig,'fig8')
# Fig9
fig,ax=plt.subplots(2,1,figsize=(Wc,5.0))
ind=R['sim_independent']; ax[0].plot([r['n'] for r in ind],[100*r['power'] for r in ind],'o-',color=C[0],ms=3,label='Independent sets (z test)')
for lab,col,ls in [('0.5%',C[2],'--'),('1%',C[1],'-.')]:
    rows=R['sim_paired'][lab]['rows']; ax[0].plot([r['n'] for r in rows],[100*r['power_exact'] for r in rows],ls=ls,color=col,marker='s',ms=3,label=f'Paired, exact McNemar, q = {lab}')
ax[0].axhline(80,color='k',ls=':',lw=0.7); ax[0].set_xscale('log'); ax[0].set_xlabel('Test-set size per detector'); ax[0].set_ylabel('Power (%)'); ax[0].legend(frameon=False,fontsize=7.2,loc='lower right'); ax[0].set_ylim(0,102); tag(ax[0],'(a)')
cl=R['cluster_cov']; names_={'wilson':'Wilson','cp':'Clopper\u2013Pearson','jeff':'Jeffreys'}
for csize,ls,mk_ in [(10,'-','o'),(50,'--','s')]:
    for meth,col in [('wilson',C[0]),('cp',C[1]),('jeff',C[2])]:
        ys=[100*cl['1|0.0'][meth]]+[100*cl[f'{csize}|{r}'][meth] for r in [0.02,0.05,0.1]]
        ax[1].plot([0,0.02,0.05,0.10],ys,ls=ls,color=col,marker=mk_,ms=3,label=f'{names_[meth]}, m = {csize}')
ax[1].axhline(95,color='k',ls=':',lw=0.7); ax[1].set_xlabel('Intra-cluster correlation \u03c1'); ax[1].set_ylabel('Coverage of nominal 95% (%)'); ax[1].legend(frameon=False,fontsize=6.6,loc='lower left',ncol=1); ax[1].set_ylim(50,100); tag(ax[1],'(b)')
fig.tight_layout(h_pad=1.0); save(fig,'fig9')
print('col ok')
