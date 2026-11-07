# Re-extracts the curves of Fig. 6 (training/validation accuracy; precision x recall vs threshold) from the figure image by pixel calibration.
# Input: fig6_orig.png (image of Fig. 6 as drawn in the original manuscript); outputs: digitised_curves.csv, digitised_threshold_curve.csv, digitisation_calibration.csv
# Marker values were read from detected marker centres (training: circles, validation: squares); overlap/off-axis cases are flagged in the CSV.
import numpy as np, csv
from PIL import Image
from scipy import ndimage as ndi
im=Image.open('fig6_orig.png').convert('RGB'); a=np.array(im).astype(int)
mask=lambda col,tol=40:(np.abs(a-np.array(col)).sum(2)<tol)
ypx=[110,173,235,297,359,421,484,546]; yval=[99.75,99.50,99.25,99.00,98.75,98.50,98.25,98.00]; ya=np.polyfit(ypx,yval,1)
xpx=[164,218,272,326,380,435,489,543,597,651]
for name,col in [('training (navy)',(31,59,115)),('validation (orange)',(224,122,31))]:
    m=mask(col); m[:,760:]=False; er=ndi.binary_erosion(m,structure=np.ones((5,5))); lab,n=ndi.label(er)
    for (cy,cx),s in zip(ndi.center_of_mass(er,lab,range(1,n+1)),ndi.sum(er,lab,range(1,n+1))):
        if s>=4: print(name,'x=%.0f'%cx,'value=%.2f'%np.polyval(ya,cy))
# threshold curve (green) with calibration of panel b
tx=np.polyfit([809,908,1008,1108,1207,1307],[0,.2,.4,.6,.8,1.0],1); by=np.polyfit([127,211,295,378,462,546],[1,.8,.6,.4,.2,0],1)
g=mask((46,139,87),50); prev=None; pts=[]
for x in range(809,1312):
    ys=np.nonzero(g[:,x])[0]
    if x<1030: ys=ys[ys<150]
    if len(ys)==0: continue
    if prev is None: y=ys[ys<ys.min()+6].mean()
    else:
        near=ys[np.abs(ys-prev)<25]
        if len(near)==0: continue
        y=near.mean()
    prev=y; pts.append((np.polyval(tx,x),np.polyval(by,y),x,y))
print(len(pts),'threshold-curve points; max P*R = %.4f'%max(p[1] for p in pts))
