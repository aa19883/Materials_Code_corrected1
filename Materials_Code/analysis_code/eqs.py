import subprocess, zipfile, re, json
EQ={
 'e15a': r"\mathrm{TPR}=\Phi\left(a+b\,\Phi^{-1}(\mathrm{FPR})\right),\quad \mathrm{AUC}=\Phi\left(\frac{a}{\sqrt{1+b^{2}}}\right),",
 'e15b': r"b=1\;\Rightarrow\; a=d'=\sqrt{2}\,\Phi^{-1}(\mathrm{AUC})",
 'e16': r"\frac{d\,\mathrm{TPR}}{d\,\mathrm{FPR}}=\frac{1-\pi}{k\,\pi}",
 'e17a': r"\hat{p}_{\mathrm{dep}}\ge\frac{1}{1+k}\;\Longleftrightarrow\;\hat{p}_{\mathrm{eval}}\ge\tau_{\mathrm{eval}}=\frac{o}{1+o},",
 'e17b': r"o=\frac{1}{k}\cdot\frac{\pi_t/(1-\pi_t)}{\pi/(1-\pi)}",
 'e18': r"A=\pi_t\,\mathrm{TPR}+(1-\pi_t)\,(1-\mathrm{FPR})",
 'e19a': r"\frac{\pi}{\pi+(1-\pi)\,\frac{1-A}{1-\pi_t}}\le \mathrm{PPV}(\pi)\le 1,",
 'e19b': r"\text{valid for } A\ge\max(\pi_t,\,1-\pi_t)",
 'e20': r"A^{*}=1-(1-\pi_t)\,\frac{\pi}{1-\pi}\,\frac{1-\mathrm{PPV}^{*}}{\mathrm{PPV}^{*}}",
 'e21': r"\mathrm{deff}=1+(m-1)\,\rho,\qquad n_{\mathrm{eff}}=\frac{n}{\mathrm{deff}}",
}
md="\n\n".join(f"$${v}$$" if False else f"${v}$" for v in EQ.values())
open('eq.md','w').write(md)
subprocess.run(['pandoc','eq.md','-o','eq.docx'],check=True)
x=zipfile.ZipFile('eq.docx').read('word/document.xml').decode()
maths=re.findall(r'<m:oMath>.*?</m:oMath>',x,flags=re.S)
print(len(maths),len(EQ))
out=dict(zip(EQ.keys(),maths))
json.dump(out,open('eq_omml.json','w'))
print(maths[0][:600])
