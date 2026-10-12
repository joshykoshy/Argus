"""
Sensitivity check: how much predicted tumor lies outside the brain, and what a
brain mask (FLAIR > 0) would change. Seed-0 predictions, WT only.

    python evaluation/check_outside_brain.py

Output: results/pred_outside_brain.csv (one row per model x condition x patient)
and a per model x condition summary on stdout. Paper section 4.4.
"""
import json, numpy as np, nibabel as nib, pandas as pd
R="archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
ids=json.load(open("data/splits/split_v1.json"))["test"]
rows=[]
for pid in ids:
    brain=np.asarray(nib.load(f"{R}/{pid}/{pid}_flair.nii").dataobj)>0
    gt=np.asarray(nib.load(f"results/predictions/ground_truth/GT_{pid}.nii.gz").dataobj)>0
    for m in ["M0","M1","M4","M5","M6"]:
        for c in ["clean","kspace_snr5_r1.0_t1","kspace_snr3_r0.5_t5"]:
            p=np.asarray(nib.load(f"results/predictions/{m}_{pid}_{c}.nii.gz").dataobj)>0
            pm=p&brain
            d=lambda a: 2*(a&gt).sum()/max(a.sum()+gt.sum(),1)
            rows.append(dict(model=m,cond=c,pid=pid,wt_cm3=p.sum()/1000,outside_cm3=(p&~brain).sum()/1000,
                             gt_cm3=gt.sum()/1000,dice_wt=d(p),dice_wt_masked=d(pm),
                             err_pct=100*abs(p.sum()-gt.sum())/gt.sum(),err_pct_masked=100*abs(pm.sum()-gt.sum())/gt.sum()))
df=pd.DataFrame(rows); df.to_csv("results/pred_outside_brain.csv",index=False)
g=df.groupby(["model","cond"]).agg(outside_mean=("outside_cm3","mean"),outside_frac=("outside_cm3",lambda s: s.sum()),
   wt=("wt_cm3","sum"),dice=("dice_wt","mean"),dice_m=("dice_wt_masked","mean"),err=("err_pct","median"),err_m=("err_pct_masked","median"))
g["outside_frac"]=g.outside_frac/g.wt
print(g.round(3).to_string())
