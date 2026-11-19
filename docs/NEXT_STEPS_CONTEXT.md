# Argus project: full context brief for research on the next (novel) method

> **How to use this document (instructions for Claude):**
> You are helping Joshua Koshy (ECTE408 research project, team of 3) choose the **next, novel research direction**
> after a finished conference paper. Read this whole brief first. Then use the Firecrawl tools
> (`firecrawl_search`, `firecrawl_research_search_papers`, `firecrawl_research_related_papers`,
> `firecrawl_research_read_paper`, `firecrawl_scrape`) to research the literature, **verify novelty**, and
> recommend the best move forward. The research brief and required output format are in **Section 9**.
> All GitHub links below are public. Fetch them with `firecrawl_scrape` when you need detail.

---

## 1. One-paragraph summary

We tested whether splitting brain MRI into low- and high-spatial-frequency bands, processed in separate
U-Net streams, makes brain tumour segmentation robust to **simulated low-field MRI degradation**. We ran a
**pre-registered**, capacity-matched comparison on BraTS 2020. **It did not work.** The dual-stream
frequency model (M4) was no better than a plain U-Net trained on the same degraded data (M1): differences of
about 0.004 Dice, below training noise, so all 3 hypotheses were not supported. **The thing that actually
worked was training on degraded scans (degradation-aware augmentation).** It raised worst-condition Dice from
0.55 to 0.77, removed spurious tumour, and kept tumour volume measurements as reliable as on clean scans. The
conference paper reports this honest negative result: *"we tested this method, it didn't work, here is what
does."* **The next project must be a novel method that builds on this finding**, not a repeat of it.

## 2. Resources (all public)

| What | Link |
|---|---|
| Our repo (Joshua's, up to date) | https://github.com/joshykoshy/Argus |
| Team repo (original) | https://github.com/mayankt411/Argus |
| README (results table, layout, reproduce steps) | https://github.com/joshykoshy/Argus/blob/master/README.md |
| Paper, PDF (IEEE conference format, 8 pages) | https://github.com/joshykoshy/Argus/blob/master/docs/paper/main.pdf |
| Paper, LaTeX source | https://github.com/joshykoshy/Argus/blob/master/docs/paper/main.tex |
| Paper bibliography (31 refs, verified DOIs) | https://github.com/joshykoshy/Argus/blob/master/docs/paper/references.bib |
| Long-form paper draft (most detail) | https://github.com/joshykoshy/Argus/blob/master/docs/PAPER_DRAFT.md |
| Test-set statistics and hypothesis verdicts | https://github.com/joshykoshy/Argus/blob/master/docs/RESULTS_STATS.md |
| Dated decision log (pre-registration) | https://github.com/joshykoshy/Argus/blob/master/docs/DECISIONS.md |
| Full lab notebook (incl. failed v1 pipeline, critical findings C1-C11) | https://github.com/joshykoshy/Argus/blob/master/docs/LAB_NOTEBOOK.md |
| Raw per-patient metrics (14,430 rows) | https://github.com/joshykoshy/Argus/blob/master/results/raw_metrics_v2.csv |
| Stats tables | https://github.com/joshykoshy/Argus/tree/master/tables |
| Figures | https://github.com/joshykoshy/Argus/tree/master/figures |
| Degradation physics code | https://github.com/joshykoshy/Argus/blob/master/pilot/pilot.py |
| Training script | https://github.com/joshykoshy/Argus/blob/master/training/train_v2.py |
| Evaluation script | https://github.com/joshykoshy/Argus/blob/master/evaluation/evaluate_v2.py |
| Dataset | BraTS 2020 on Kaggle: `awsaf49/brats20-dataset-training-validation` |

Paper title: *"Degradation-Aware Training, Not Frequency-Band Decomposition, Makes Brain Tumor Segmentation
Robust to Simulated Low-Field MRI"*. Authors: Joshua Koshy, Muhammad Nazir Ahmed Mapkar, Mayank.

## 3. Motivation (why low-field matters)

- Conventional 1.5-3 T MRI is expensive, so it is concentrated in high-income hospitals. One survey counted
  84 MRI units for ~373 million people in West Africa (Ogbole 2018).
- Low-field and portable scanners (0.25-1 T; Hyperfine Swoop ~0.064 T) cost a fraction as much and can be
  used at the bedside, but they have **lower SNR, larger voxels and thicker slices**.
- Tumour-segmentation networks are trained on high-field data (BraTS). We found **no glioma-segmentation
  evaluation on real low-field scans** in the literature (as of Oct 2026, please re-verify). Most low-field
  work restores the image first, e.g. SynthSR-style super-resolution for brain morphometry (Iglesias 2023).

## 4. Study design (what we actually did)

**Data:** BraTS 2020 training set, 369 patients, 4 sequences (T1, T1ce, T2, FLAIR), 240x240x155 at 1 mm,
labels WT/TC/ET. Fixed split of 258 train / 37 val / 74 test (seed 42, stratified HGG/LGG). The test set
was used once.

**Degradation (DEC-005), simulated per sequence on magnitude images:**
1. Slice thickness: a t mm slice is the mean of t consecutive 1 mm slices.
2. In-plane resolution: keep the central fraction r of (pseudo) k-space.
3. Thermal noise: complex Gaussian noise restricted to the acquired k-space band, sigma = mu_brain / SNR,
   then the magnitude (Rician tissue, Rayleigh background).
4. 13 conditions: SNR {8, 5, 3} x r {1.0, 0.5} x t {1, 5 mm}, plus clean. The worst is SNR 3, r 0.5, 5 mm.
   SNR 2 was excluded as near-unusable (costs up to 0.50 Dice).
5. Val/test noise is seeded per (patient, condition) with SHA-256, so every model sees identical volumes.
   A MATLAB re-implementation is parity-verified against the Python one.

**Frequency decomposition:** Gaussian low-pass in the centred Fourier domain,
H_lp = exp(-D^2 / 2 D0^2), D0 = 0.20 (normalised to Nyquist), high = 1 - low. The split is exactly
invertible: it is a fixed, linear, lossless filter.

**Models (DEC-007):** five capacity-matched 2D U-Nets (4.2-4.4 M parameters, 5 levels, InstanceNorm, LeakyReLU,
sigmoid WT/TC/ET).

| Model | What it is | Training data |
|---|---|---|
| M0 | plain U-Net | clean only |
| M1 | plain U-Net | degraded (1 of 13 conditions drawn uniformly per slice) |
| M4 | **proposed** dual-stream (low band + high band in separate encoders) | degraded |
| M5 | low band only | degraded |
| M6 | early fusion (both bands stacked as channels) | degraded |

Training: 20 epochs, all brain slices, 12k slices/epoch, batch 32, AdamW 3e-4, warm-up + cosine, BCE + soft
Dice, flips, mixed precision, seeds 0/1/2. Model selection used mean 3D val Dice over {clean, SNR 5, worst}.
Compute: Colab and Kaggle free T4 GPUs. Joshua's laptop is a Snapdragon X (no CUDA).

**Pre-registered hypotheses (DEC-006), fixed before training:** paired Wilcoxon on 74 test patients, Holm
correction, plus a rule that any difference < 2x seed-to-seed SD counts as training noise.
- H1 (primary): M4 > M1 at the worst condition.
- H2: M4's clean-to-worst drop is smaller than M1's.
- H3: M4 > M6 at the worst condition (two streams vs one).

## 5. Results (74 test patients, 3 seeds, full-volume 3D Dice = mean of WT/TC/ET)

| Model | Clean | SNR 5 | Worst | Drop | WT vol. error (worst) | WT vol. ICC (worst) |
|---|---|---|---|---|---|---|
| M0 clean-trained | 0.786 | 0.674 | 0.549 | 0.237 | 25.8 % | 0.74 |
| M1 U-Net + aug | 0.830 | 0.822 | 0.768 | 0.061 | 7.1 % | 0.94 |
| M4 dual-stream | 0.822 | 0.816 | 0.765 | 0.057 | 7.0 % | 0.94 |
| M5 low band only | 0.819 | 0.811 | 0.763 | 0.055 | 7.8 % | 0.93 |
| M6 early fusion | 0.826 | 0.820 | 0.769 | 0.057 | 7.2 % | 0.94 |

Hypothesis verdicts (Holm, alpha 0.05):
- H1: M4 - M1 = -0.0037 (95 % CI -0.0086 to +0.0004), Holm p = 0.043, wrong direction, within noise.
  **Not supported.**
- H2: +0.0041, p = 0.70. **Not supported.**
- H3: M4 - M6 = -0.0044, Holm p = 0.0017, wrong direction, within noise. **Not supported.**
- Seed noise floor: 2x SD = 0.014.

Other findings:
- **Augmentation is the effect:** M4 vs M0 at worst = +0.216 Dice (CI 0.195-0.237).
- **False tumour on tumour-free slices:** M0 produced 19.8 cm^3 at worst; every augmented model about 0.4 cm^3.
- **HD95 (WT median):** M0 goes from 5.7 to 105 mm. The augmented models stay at 3.3-4.9 mm.
- **Brain-masked sensitivity analysis (post hoc):** much of M0's false tumour lies outside the brain. With
  masking, the augmentation gain at worst is +0.126 (CI 0.112-0.142). At SNR 5 the masked gain is within noise.
- **ET is the least robust region** for every model (worst ET Dice ~0.65 for the augmented models vs WT ~0.86).
  Every model predicted ET in patients without ET, so ET presence is unreliable in LGG.
- **M0 is unstable across seeds** (clean seed SD 0.048, about 10x that of the augmented models).

Morphometric and agreement lessons (validated on synthetically damaged masks):
- A 1 mm boundary error changes ET volume by a median 48 % but WT by only 17 % (ET is a thin shell).
- A 2 mm shift keeps volume ICC = 1.000 while ET Dice drops to 0.71. Three spurious blobs keep Dice at 0.997,
  and only the connected-component count catches them. Report overlap, volume and component metrics together.
- A systematic +17 % volume bias still gives ICC 0.957. Near-zero mean bias can hide +/-73 cm^3 per-patient
  limits of agreement (M0). Report bias, limits of agreement and ICC together.
- Expert inter-rater glioma volume ICC is 0.92-0.99 and novice 0.73 (Visser 2019). Augmented models at worst
  sit at 0.93-0.94 (low expert range); M0 sits at 0.74 (novice).

## 6. Why the frequency method failed (our interpretation)

- The band split is a **fixed, linear, invertible filter**, so it adds no information a CNN cannot learn in
  its first layers.
- Once the network sees degraded scans in training, it **learns on its own to discount unreliable fine
  detail**, which is exactly what the split was meant to do.
- M5 (low band only) matched M4, so the high band carried ~nothing useful under this degradation (it is mostly
  noise from SNR 5 down). The premise (noise lives in high frequencies) was right; the network didn't need
  the split to exploit it.
- Implication: **a new method must add something augmentation cannot**. Examples: information the network
  does not have (acquisition parameters, physics, k-space, priors, extra data), a different output (uncertainty,
  calibrated volumes, failure detection), or robustness *outside* the training distribution.

## 7. Limitations of the paper (the open space for the next study)

1. **Simulated degradation on high-field magnitude images.** It omits low-field **contrast changes** (T1/T2
   relaxation differ at low field), B0/B1 inhomogeneity, and other artefacts. No real labelled low-field
   tumour scans were used.
2. **In-distribution testing:** augmented models were tested on the same 13 conditions they trained on, which
   is a best case. Unseen or harsher degradation (SNR 2, different r, real scanner) was not tested.
3. 2D networks, ~4 M parameters, 20 epochs, one dataset (BraTS 2020), three seeds.
4. One decomposition cutoff (D0 = 0.20).
5. No post-processing, no uncertainty estimates, no calibration.
6. Morphometrics only on seed 0 at three conditions.
7. Pre-registration was an internal dated log, not a public registry.

## 8. Practical constraints for the next study

- **Team:** Joshua (training, MATLAB morphometrics, 3D rendering, physics port, paper writing), Nazir
  (evaluation, statistics, literature, interactive 3D viewer), Mayank (original pipeline, architectures,
  physics).
- **Compute:** free Colab/Kaggle T4 GPUs (session and quota limits). One 2D U-Net run takes a few GPU-hours.
  Laptop has no CUDA. 3D nnU-Net-scale training is expensive but maybe feasible on Kaggle.
- **Data on hand:** BraTS 2020 (~40 GB locally). Other public data must be downloadable.
- **Reusable infrastructure (big asset):** a validated degradation simulator (Python plus parity-tested MATLAB),
  a training/eval pipeline with deterministic paired noise, a stats pipeline (Wilcoxon, Holm, bootstrap CIs,
  seed-noise rule), a morphometrics and agreement toolkit (volume, sphericity, components, Bland-Altman, ICC),
  3D renderers, and 15 trained checkpoints (M0/M1/M4/M5/M6 x 3 seeds) usable as baselines.
- **Method standards to keep:** pre-registered hypotheses with a fair comparator (**M1, the augmented U-Net,
  is now the baseline to beat**), capacity matching, multiple seeds, a noise-floor rule, a single use of the
  test set, and honest reporting.
- Tools: Python/PyTorch, MATLAB R2025a + Image Processing Toolbox (no Statistics Toolbox).

## 9. Research brief for you (Claude + Firecrawl)

**Goal:** find the best **novel** next method for robust brain tumour segmentation (and clinically reliable
tumour volumes) on low-field / degraded MRI. It must **beat or add to degradation-aware augmentation (M1)**,
be feasible for a 3-person student team on free GPUs, and be publishable (journal extension or new paper).

**Seed directions to investigate. Verify novelty for each; do not assume any is novel:**
1. **Acquisition-conditioned networks:** feed estimated SNR/resolution (or a learned degradation embedding)
   into the network via FiLM / hypernetworks, vs blind augmentation.
2. **Uncertainty and calibrated volumetry:** conformal prediction or ensembles that give a **tumour volume
   with a guaranteed confidence interval** that widens as scan quality drops. Also automatic QC that flags when
   a segmentation should not be trusted.
3. **Realistic low-field simulation:** physics-based contrast synthesis (relaxation-time changes at 0.064 T,
   bias fields, real Hyperfine noise characteristics), plus testing on *unseen* degradation (OOD robustness).
4. **Joint or task-driven restoration + segmentation** (super-resolution/denoising trained for the segmentation
   loss) vs augmentation only.
5. **Worst-case / adversarial / curriculum degradation augmentation** instead of uniform sampling.
6. **Missing sequences:** portable scanners often acquire fewer sequences, so test robustness to dropped
   modalities combined with degradation.
7. **Foundation models** (SAM-Med3D, BrainSegFounder, MedSAM-style, BraTS-pretrained nnU-Net) under low-field
   degradation: does pretraining replace augmentation?
8. **Real low-field data:** any public paired high/low-field or labelled low-field datasets (e.g. the LISA
   ultra-low-field challenge, Hyperfine-related releases, M4Raw at 0.3 T). Can they serve as an external test set?
9. Anything better you find. Prefer directions where our existing infrastructure gives us a head start.

**Suggested searches:** "low-field MRI brain tumor segmentation", "portable MRI glioma segmentation",
"Hyperfine 0.064T segmentation deep learning", "ultra-low-field MRI segmentation challenge LISA",
"low-field MRI simulation contrast relaxation deep learning", "conformal prediction segmentation volume",
"acquisition-conditioned segmentation FiLM MRI", "degradation-aware brain MRI segmentation 2025 2026",
"SynthSeg low-field robustness", "paired 3T 0.064T dataset", "M4Raw low-field dataset". Use
`firecrawl_research_related_papers` on our closest references: Iglesias 2023 (doi 10.1148/radiol.220522),
SynthSeg (Billot et al.), Zhang 2020 BigAug (doi 10.1109/TMI.2020.2973595), Karani 2021 test-time adaptation
(doi 10.1016/j.media.2020.101907), HFF-Net (Shao 2025).

**Required output:**
1. A **ranked shortlist of 3-5 directions**. For each: the core idea in 2 sentences; **novelty evidence**
   (closest existing papers with links, and the exact gap left); why it should beat M1 (mechanism, not hope);
   feasibility on free T4s; which parts of our pipeline it reuses; datasets needed (with links); and risk.
2. For the **top pick:** 2-3 pre-registerable hypotheses with the comparator (M1), primary metric, worst
   condition, and a kill criterion (what result would make us stop). Add a minimal experiment plan and a
   rough timeline.
3. A list of every paper and dataset cited, with URLs/DOIs. **Only cite things you actually opened.** Mark
   anything unverified.

## 10. Key references already in our paper (verified DOIs)

- Iglesias et al., portable low-field MRI morphometry via super-resolution, *Radiology* 2023, 10.1148/radiol.220522
- Arnold et al., Low-field MRI: clinical promise and challenges, *JMRI* 2023, 10.1002/jmri.28408
- Marques, Simonis, Webb, Low-field MRI: an MR physics perspective, *JMRI* 2019, 10.1002/jmri.26637
- Sheth et al., portable low-field MRI at the bedside, *JAMA Neurol* 2021, 10.1001/jamaneurol.2020.3263
- Ogbole et al., MRI in West Africa, *Pan Afr Med J* 2018, 10.11604/pamj.2018.30.240.14000
- Anazodo et al., MRI access in Africa, *NMR Biomed* 2023, 10.1002/nbm.4846
- Menze et al., BRATS benchmark, *IEEE TMI* 2015, 10.1109/TMI.2014.2377694
- Bakas et al., TCGA glioma collections, *Sci Data* 2017, 10.1038/sdata.2017.117
- Ronneberger et al., U-Net, MICCAI 2015, 10.1007/978-3-319-24574-4_28
- Isensee et al., nnU-Net, *Nat Methods* 2021, 10.1038/s41592-020-01008-z
- Zhang et al., deep stacked transformation (BigAug), *IEEE TMI* 2020, 10.1109/TMI.2020.2973595
- Karani et al., test-time adaptable networks, *MedIA* 2021, 10.1016/j.media.2020.101907
- Yin et al., Fourier perspective on robustness (NeurIPS 2019); Wang et al., high-frequency component and
  CNN generalisation (CVPR 2020); Xu et al., learning in the frequency domain (CVPR 2020)
- Zhou et al. 2024 spatial-frequency dual progressive attention (SF-UNet); Shao et al. 2025 HFF-Net
  (closest prior frequency-split brain-tumour design)
- Visser et al., inter-rater agreement on glioma volume, *NeuroImage Clin* 2019, 10.1016/j.nicl.2019.101727
- Bland & Altman 1986, 10.1016/S0140-6736(86)90837-8; Shrout & Fleiss 1979, 10.1037/0033-2909.86.2.420
- Gudbjartsson & Patz, Rician MRI noise, *MRM* 1995, 10.1002/mrm.1910340618
- Full list: https://github.com/joshykoshy/Argus/blob/master/docs/paper/references.bib
