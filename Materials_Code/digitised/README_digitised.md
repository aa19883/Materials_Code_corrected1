# Digitised curves (qualitative use only; main text Section 4.6)

Files
- digitised_curves.csv — training and validation accuracy (%) per epoch, with pixel coordinates.
- digitised_threshold_curve.csv — precision × recall versus decision threshold (482 points), with pixel coordinates.
- digitisation_calibration.csv — pixel positions of the axis ticks used for the calibration (panel a: accuracy and epoch; panel b: precision × recall and threshold).
- digitise_fig6.py, fig6_orig.png — the extraction script and the figure image it reads.

How the values were obtained
- Source: Fig. 6 as drawn in the original manuscript, which reproduces the curves of the source study. The values were re-extracted from the figure image by colour segmentation and linear calibration against the axis ticks (marker centres for the epoch curves; curve pixels for the threshold curve).
- Reading error: about one pixel, i.e. ≈ ±0.01 percentage points of accuracy and ≈ ±0.005 on precision × recall.
- Training accuracy at epoch 1 (92.1%) lies below the plotted axis; it is taken from the text of the original manuscript. Training accuracy at epoch 8 (99.3%) is inferred, because its marker is hidden under the validation marker. Both are flagged in the CSV.
- Loss curves are not digitised.
- Threshold curve: the re-extracted maximum lies at a threshold of 0.047 and the 1% plateau at 0.037–0.60 (3% plateau up to 0.86), whereas the values used in the manuscript (maximum 0.065; plateau 0.054–0.585; 3% up to 0.85) come from the original digitisation. The curve is almost flat near its maximum (one pixel ≈ 0.4% of the vertical range), so the position of the maximum is poorly determined; the manuscript keeps the original values.
- Checks against the numbers quoted in Section 4.6: validation accuracy after epoch 2 ranges 98.2–99.3 (SD 0.35 from the values rounded to 0.1; 0.36 in the manuscript, which used unrounded digitised values); largest epoch-to-epoch step 0.8 points; mean training–validation gap over epochs 6–10 is 0.42 points; final validation accuracy 98.8%, final training accuracy 99.5%.
