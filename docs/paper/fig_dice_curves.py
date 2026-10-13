"""Figure: mean 3D Dice per model across the 13 DEC-005 conditions, 95 % bootstrap CI bands.

    python docs/paper/fig_dice_curves.py      -> figures/dice_curves.png (from tables/dice_curves.csv)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

c = pd.read_csv("tables/dice_curves.csv")
c = c[c.metric == "dice_mean"].copy()
c["snr_key"] = c.snr.replace(0, 99)                     # clean first, then SNR 8 -> 3
order = (c[["condition", "snr_key", "r", "thick"]].drop_duplicates()
         .sort_values(["snr_key", "r", "thick"], ascending=[False, False, True]))
lab = ["clean" if s == 99 else f"{int(s)} / {r:g} / {int(t)}" for s, r, t in
       zip(order.snr_key, order.r, order.thick)]
pos = {k: i for i, k in enumerate(order.condition)}

plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
for m, col in zip(["M0", "M1", "M4", "M5", "M6"], ["#444444", "#d55e00", "#0072b2", "#009e73", "#cc79a7"]):
    g = c[c.model == m].assign(x=lambda d: d.condition.map(pos)).sort_values("x")
    ax.plot(g.x, g["mean"], "-o", ms=2.5, lw=1, color=col, label=m)
    ax.fill_between(g.x, g.ci95_low, g.ci95_high, color=col, alpha=0.18, lw=0)
ax.set_xticks(range(len(lab)), lab, rotation=60, ha="right", fontsize=6)
ax.set_xlabel("SNR / in-plane r / slice thickness (mm)")
ax.set_ylabel("Mean 3D Dice (WT, TC, ET)")
ax.grid(alpha=0.3, lw=0.4)
ax.legend(ncol=5, fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
fig.savefig("figures/dice_curves.png")
print("figures/dice_curves.png")
