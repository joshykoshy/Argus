"""Interactive 3D tumour viewer: one HTML per case. Default: ground truth only (self-contained).
--compare: ground truth vs every model's seed-0 prediction at clean / SNR 5 / worst, switchable from a
dropdown, prediction surfaces coloured by signed distance to the true boundary. Click legend entries to
toggle layers.

    python -m viz3d.viewer                 # truth-only viewers for the 5 report cases
    python -m viz3d.viewer --compare       # comparison viewers (needs results/predictions/)
"""
import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from viz3d import meshes

DATA_ROOT = meshes.DATA_ROOT
OUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "interactive"

DEFAULT_CASES = ["008", "167", "236", "292", "281"]

# key -> (mask region, colour, opacity, visible). WT, TC and ET are shown by default; ED and NCR are legend toggles.
STYLE = {
    "WT":  ("WT", "#1f77b4", 0.25, True),
    "TC":  ("TC", "#ff7f0e", 0.35, True),
    "ET":  ("ET", "#d62728", 0.90, True),
    "ED":  ("ED", "#e6c229", 0.25, "legendonly"),
    "NCR": ("NCR", "#7b3fa0", 0.60, "legendonly"),
}
FULL_NAMES = {"WT": "Whole tumour (1+2+4)", "TC": "Tumour core (1+4)", "ED": "Edema (2)", "NCR": "Necrotic/non-enhancing (1)", "ET": "Enhancing (4)"}


def build_figure(seg, spacing, axcodes, title):
    voxel_ml = float(spacing.prod()) / 1000.0
    fig = go.Figure()
    volumes = []
    for key, (region, colour, opacity, visible) in STYLE.items():
        mask = meshes.region_mask(seg, region)
        mesh = meshes.mask_to_mesh(mask, spacing, sigma_mm=meshes.SIGMA_MM, level=meshes.LEVEL)   # 0.5 / 0.5 for every layer
        if mesh is None:   # e.g. case 281 has no enhancing tumour
            continue
        v, f = mesh
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=colour, opacity=opacity, name=f"{FULL_NAMES[key]} — {mask.sum() * voxel_ml:.1f} mL",
            showlegend=True, visible=visible, flatshading=False,
            lighting=dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6),
            hovertemplate=f"{FULL_NAMES[key]}<br>x %{{x:.1f}} y %{{y:.1f}} z %{{z:.1f}} mm<extra></extra>"))
        if key in ("WT", "TC", "ET"):
            volumes.append(f"{key} {mask.sum() * voxel_ml:.1f} mL")
    axis = lambda a, c: dict(title=f"{a} (→ {c}) mm", backgroundcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=dict(text=f"{title} — ground truth<br><sup>{' · '.join(volumes)}</sup>"),
        scene=dict(aspectmode="data", xaxis=axis("x", axcodes[0]), yaxis=axis("y", axcodes[1]),
                   zaxis=axis("z", axcodes[2])),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=70, b=0), template="plotly_white")
    return fig


def build_viewer(data_root, case, out_dir):
    name = meshes.case_name(case)
    seg, spacing, axcodes = meshes.load_seg(data_root, name)
    fig = build_figure(seg, spacing, axcodes, name)
    out = Path(out_dir) / f"{name}_truth.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True, full_html=True)   # embedded JS: works offline
    return out


# ---- comparison mode: ground truth vs model predictions (seed 0, evaluation/evaluate_v2.py NIfTIs) ----
PRED_DIR = Path(__file__).resolve().parents[1] / "results" / "predictions"
COMPARE_MODELS = ["M0", "M4"]                         # clean-trained baseline vs proposed (paper)
COMPARE_CONDS = {"clean": "clean", "kspace_snr5_r1.0_t1": "SNR 5", "kspace_snr3_r0.5_t5": "SNR 3, r 0.5, 5 mm"}
ERR_MM = 5.0


def _mesh(mask, spacing, **kw):
    m = meshes.mask_to_mesh(mask, spacing)
    if m is None:
        return None, None
    v, f = m
    v32, f32 = v.astype(np.float32), f.astype(np.int32)   # halves the HTML size; sub-micron precision
    return go.Mesh3d(x=v32[:, 0], y=v32[:, 1], z=v32[:, 2], i=f32[:, 0], j=f32[:, 1], k=f32[:, 2], flatshading=False,
                     lighting=dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6), **kw), v


def build_compare_figure(data_root, case, pred_dir=PRED_DIR, models=COMPARE_MODELS, conds=COMPARE_CONDS):
    """One page, buttons to switch between ground truth and each model x condition.

    Prediction views show the predicted WT surface coloured by signed distance (mm) to the true WT
    boundary (red outside, blue inside; surface_error.vertex_errors) plus predicted TC and ET; the truth
    view is the build_figure style. A faint brain surface is always shown for context.
    """
    import nibabel as nib
    from viz3d.surface_error import vertex_errors

    name = meshes.case_name(case)
    seg, spacing, axcodes = meshes.load_seg(data_root, name)
    flair = np.asarray(nib.load(Path(data_root) / name / f"{name}_flair.nii").dataobj) > 0
    truth_wt = meshes.region_mask(seg, "WT")
    fig = go.Figure()
    brain, _ = _mesh(flair[::2, ::2, ::2], spacing * 2, color="lightgrey", opacity=0.08, name="Brain",
                     hoverinfo="skip", showlegend=False)
    fig.add_trace(brain)
    views = []                                            # (label, trace indices)

    idx = []
    for key in ("WT", "TC", "ET"):
        _, colour, opacity, _ = STYLE[key]
        t, _ = _mesh(meshes.region_mask(seg, key), spacing, color=colour, opacity=opacity, name=FULL_NAMES[key])
        if t is not None:
            fig.add_trace(t); idx.append(len(fig.data) - 1)
    views.append(("Ground truth", idx))

    for m in models:
        for cid, clabel in conds.items():
            f = Path(pred_dir) / f"{m}_{name}_{cid}.nii.gz"
            if not f.exists():
                continue
            p = np.asarray(nib.load(f).dataobj).astype(np.uint8)
            p[p == 3] = 4                                 # prediction ET label 3 -> BraTS 4
            idx = []
            wt, v = _mesh(meshes.region_mask(p, "WT"), spacing, colorscale="RdBu_r", cmin=-ERR_MM, cmax=ERR_MM,
                          opacity=0.55, name=f"{m} WT (surface error)",
                          colorbar=dict(title="mm", len=0.5, thickness=14))
            if wt is not None:
                err = vertex_errors(v, truth_wt, spacing)
                wt.update(intensity=err, hovertemplate="%{intensity:.1f} mm<extra></extra>")
                fig.add_trace(wt); idx.append(len(fig.data) - 1)
            for key in ("TC", "ET"):
                _, colour, opacity, _ = STYLE[key]
                t, _ = _mesh(meshes.region_mask(p, key), spacing, color=colour, opacity=opacity,
                             name=f"{m} {FULL_NAMES[key]}")
                if t is not None:
                    fig.add_trace(t); idx.append(len(fig.data) - 1)
            label = f"{m} · {clabel}"
            if wt is not None:
                label += f" (mean |d| {np.abs(err).mean():.1f} mm)"
            views.append((label, idx))

    n = len(fig.data)
    for i in range(1, n):
        fig.data[i].visible = i in views[0][1]
    buttons = [dict(label=lab, method="update",
                    args=[{"visible": [True] + [i in ids for i in range(1, n)]}, {"title.text": f"{name}: {lab}"}])
               for lab, ids in views]
    axis = lambda a, c: dict(title=f"{a} (→ {c}) mm", backgroundcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=dict(text=f"{name}: Ground truth"),
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, xanchor="left", y=1.0, yanchor="top")],
        scene=dict(aspectmode="data", xaxis=axis("x", axcodes[0]), yaxis=axis("y", axcodes[1]),
                   zaxis=axis("z", axcodes[2])),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=60, b=0), template="plotly_white")
    return fig


def build_compare_viewer(data_root, case, out_dir, pred_dir=PRED_DIR):
    name = meshes.case_name(case)
    out = Path(out_dir) / f"{name}_compare.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_compare_figure(data_root, case, pred_dir).write_html(out, include_plotlyjs="cdn", full_html=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="*", default=DEFAULT_CASES)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--compare", action="store_true",
                    help="truth vs M0/M4 predictions (needs results/predictions/)")
    ap.add_argument("--pred-dir", default=PRED_DIR)
    args = ap.parse_args()
    for c in args.cases:
        if args.compare:
            out = build_compare_viewer(DATA_ROOT, c, args.out_dir, args.pred_dir)
        else:
            out = build_viewer(DATA_ROOT, c, args.out_dir)
        print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
