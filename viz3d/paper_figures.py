"""Static 3D paper figures from the viz3d meshes: ground truth vs M0 vs M4 at the worst condition.

One PNG per case (figures/viz3d_case<k>_<id>.png), three side-by-side 3D scenes on the same camera:
  1. ground truth    WT (blue, see-through), TC (orange), ET (red), as in viewer.py
  2. M0 prediction   WT surface coloured by signed distance (mm) to the true WT boundary
  3. M4 prediction   same, from viz3d.surface_error.vertex_errors (validated: dilate 2 mm -> +2.04)
with a faint brain surface for context. Seed-0 predictions written by evaluation/evaluate_v2.py.

    python -m viz3d.paper_figures --data-root <MICCAI_BraTS2020_TrainingData>
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageOps
from plotly.subplots import make_subplots

from viz3d import meshes
from viz3d.surface_error import vertex_errors

CASES = ["008", "167", "236", "292", "281"]          # diagnostic_selection.csv order (cases 1-5)
COND = "kspace_snr3_r0.5_t5"                         # DEC-005 worst condition
MODELS = ["M0", "M4"]
ERR_MM = 5.0                                         # colour scale limits, as in the MATLAB renderer
CAMERA = dict(eye=dict(x=1.5, y=-1.5, z=0.9))
STYLE = {"WT": ("#1f77b4", 0.20), "TC": ("#ff7f0e", 0.35), "ET": ("#d62728", 0.90)}


def load_pred(path):
    """Prediction label map (1 NCR, 2 ED, 3 ET) -> BraTS labels (ET = 4) so meshes.REGIONS applies."""
    p = np.asarray(nib.load(path).dataobj).astype(np.uint8)
    p[p == 3] = 4
    return p


def mesh_trace(mask, spacing, **kw):
    m = meshes.mask_to_mesh(mask, spacing)
    if m is None:
        return None, None
    v, f = m
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], flatshading=False,
                     lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2), showlegend=False, **kw), v


def case_figure(data_root, pred_dir, case, k):
    name = meshes.case_name(case)
    seg, spacing, _ = meshes.load_seg(data_root, name)
    flair = np.asarray(nib.load(Path(data_root) / name / f"{name}_flair.nii").dataobj) > 0
    brain, _ = mesh_trace(flair[::2, ::2, ::2], spacing * 2, color="lightgrey", opacity=0.08, hoverinfo="skip")
    truth_wt = meshes.region_mask(seg, "WT")

    titles = ["Ground truth"]
    scenes = [[]]
    for r in ("WT", "TC", "ET"):
        t, _ = mesh_trace(meshes.region_mask(seg, r), spacing, color=STYLE[r][0], opacity=STYLE[r][1])
        if t is not None:
            scenes[0].append(t)
    stats = []
    for m in MODELS:
        pred = load_pred(Path(pred_dir) / f"{m}_{name}_{COND}.nii.gz")
        wt, v = mesh_trace(meshes.region_mask(pred, "WT"), spacing, colorscale="RdBu_r", cmin=-ERR_MM,
                           cmax=ERR_MM, opacity=1.0, showscale=(m == MODELS[-1]),
                           colorbar=dict(title="mm", len=0.6, thickness=14))
        tr = []
        if wt is not None:
            err = vertex_errors(v, truth_wt, spacing)
            wt.update(intensity=err)
            tr.append(wt)
            stats.append(f"{m}: mean |d| {np.abs(err).mean():.1f} mm")
        scenes.append(tr)
        titles.append(f"{m} (WT surface error)")

    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3], subplot_titles=titles,
                        horizontal_spacing=0.01)
    for c, traces in enumerate(scenes, 1):
        for t in ([brain] if brain is not None else []) + traces:
            fig.add_trace(t, row=1, col=c)
    ax = dict(visible=False)
    for c in range(1, 4):
        fig.update_layout(**{f"scene{'' if c == 1 else c}": dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data",
                                                                 camera=CAMERA)})
    fig.update_layout(title=dict(text=f"Case {k}: {name}, SNR 3, r 0.5, 5 mm (seed 0). {'; '.join(stats)}",
                                 font=dict(size=15)),
                      margin=dict(l=0, r=0, t=60, b=0), template="plotly_white", width=1500, height=520)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=meshes.DATA_ROOT)
    ap.add_argument("--pred-dir", default="results/predictions")
    ap.add_argument("--out-dir", default="figures")
    a = ap.parse_args()
    for k, case in enumerate(CASES, 1):
        out = Path(a.out_dir) / f"viz3d_case{k}_{meshes.case_name(case)}.png"
        case_figure(a.data_root, a.pred_dir, case, k).write_image(out, scale=2)
        im = Image.open(out).convert("RGB")                # crop the white margins plotly leaves
        box = ImageOps.invert(im).point(lambda v: 255 if v > 8 else 0).getbbox()
        im.crop((max(box[0] - 20, 0), max(box[1] - 20, 0), box[2] + 20, box[3] + 20)).save(out)
        print(out, flush=True)


if __name__ == "__main__":
    main()
