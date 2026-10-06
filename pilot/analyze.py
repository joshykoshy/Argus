"""
Summarise the degradation pilot (pilot.py output).

Writes
  <out>/pilot_dice_vs_snr.png   mean Dice vs SNR per model, one panel per
                                (noise order x in-plane r x slice thickness)
  <out>/pilot_summary.csv       per condition: M0 Dice, M0 drop from clean, and the
                                paired gaps M1-M0, M4-M0, M4-M1 (mean over patients)
and prints the conditions where the clean-trained baseline loses >= 0.10 Dice,
which is the range where a robustness method has room to show a gain.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.csv)
n_pat = df.patient_id.nunique()
models = [m for m in ["M0", "M1", "M4", "M5", "M6"] if m in df.model.unique()]

# Paired, per patient: each model's Dice minus its own clean Dice, and gaps between models.
wide = df.pivot_table(index=["patient_id", "condition_id"], columns="model", values="dice_mean").reset_index()
meta = df.drop_duplicates("condition_id").set_index("condition_id")[["noise", "snr", "r", "thick"]]
clean = wide[wide.condition_id == "clean"].set_index("patient_id")[models]
rows = []
for cid, g in wide.groupby("condition_id"):
    g = g.set_index("patient_id")
    r = dict(condition_id=cid, **meta.loc[cid].to_dict(), n=len(g))
    for m in models:
        r[f"{m}_dice"] = g[m].mean()
        r[f"{m}_drop"] = (clean.loc[g.index, m] - g[m]).mean()
    for a, b in [("M1", "M0"), ("M4", "M0"), ("M4", "M1")]:
        if a in g and b in g:
            d = g[a] - g[b]
            r[f"{a}-{b}"] = d.mean()
            r[f"{a}-{b}_sd"] = d.std()
    rows.append(r)
S = pd.DataFrame(rows).sort_values("M0_drop", ascending=False)
S.to_csv(out / "pilot_summary.csv", index=False)

pd.set_option("display.width", 200)
cols = ["condition_id", "M0_dice", "M0_drop", "M1-M0", "M4-M0", "M4-M1"]
print(f"{n_pat} validation patients. Conditions sorted by how much the clean-trained M0 loses:")
print(S[[c for c in cols if c in S]].round(3).to_string(index=False))
hard = S[S.M0_drop >= 0.10]
print(f"\nConditions where M0 loses >= 0.10 mean Dice: {len(hard)}")
print(", ".join(hard.condition_id) if len(hard) else "none - the range must go harsher still")

# Figure: Dice vs SNR, one panel per (noise order, r, thick).
snrs = [8, 5, 3, 2]
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
for i, noise in enumerate(["image", "kspace"]):
    for j, (r, t) in enumerate([(1.0, 1), (1.0, 5), (0.5, 1), (0.5, 5)]):
        ax = axes[i, j]
        for m in models:
            ys = [df[(df.model == m) & (df.condition_id == "clean")].dice_mean.mean()]
            for s in snrs:
                ys.append(df[(df.model == m) & (df.noise == noise) & (df.snr == s)
                             & (df.r == r) & (df.thick == t)].dice_mean.mean())
            ax.plot(range(len(ys)), ys, "o-", lw=1.8, label=m)
        ax.set_xticks(range(len(snrs) + 1))
        ax.set_xticklabels(["clean"] + [str(s) for s in snrs])
        ax.set_title(f"noise {noise}, r = {r}, slice {t} mm", fontsize=11)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("Mean Dice (WT, TC, ET)")
        if i == 1:
            ax.set_xlabel("SNR (linear)")
axes[0, 0].legend(title="model")
fig.suptitle(f"Degradation pilot: trained checkpoints on {n_pat} validation patients", fontsize=13)
fig.tight_layout()
fig.savefig(out / "pilot_dice_vs_snr.png", dpi=150)
print(f"\nSaved {out / 'pilot_dice_vs_snr.png'} and {out / 'pilot_summary.csv'}")
