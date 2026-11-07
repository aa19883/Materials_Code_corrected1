import json, numpy as np
I=json.load(open('inputs.json')); R=json.load(open('results.json'))
A=I['caseA']; TP,FN,FP,TN=A['TP'],A['FN'],A['FP'],A['TN']
rng=np.random.default_rng(I['seed'])
D=rng.dirichlet([TP+.5,FN+.5,FP+.5,TN+.5],size=200000); tp,fn,fp,tn=D.T
q=lambda x:[float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]
R['A_crI']=dict(acc=q(tp+tn),rec=q(tp/(tp+fn)),prec=q(tp/(tp+fp)),tnr=q(tn/(tn+fp)),fpr=q(fp/(fp+tn)),fnr=q(fn/(fn+tp)),
                f1=q(2*tp/(2*tp+fp+fn)),mcc=q((tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
json.dump(R,open('results.json','w'),indent=1,default=float)
print({k:[round(100*x,2) for x in v] for k,v in R['A_crI'].items()})
