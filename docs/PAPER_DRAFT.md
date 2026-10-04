# Paper draft - sections that do not depend on results

Status: **draft for the team** (roadmap step 1.5). Written from the code as
it stands, so every number here is traceable to a file. Square-bracket notes:

- `[CITE: ...]` - a reference is needed; the suggested source is listed in
  the reference list at the end, **marked "verify"**: check each one against
  the actual paper before submission (titles, years and venues were written
  from memory).
- `[TEAM: ...]` - a decision the team still has to make (see
  `LAB_NOTEBOOK.md`, Critical findings C1-C7).
- `[RESULT]` - filled in after Stage 3-4.

Working title: *Does frequency-band decomposition make brain tumor
segmentation robust to low-field MRI degradation? A controlled comparison
with morphometric validation.*

---

## 1. Introduction

### 1.1 Low-field MRI and why segmentation robustness matters

Magnetic resonance imaging is the reference modality for brain tumor
diagnosis, treatment planning and follow-up, but conventional high-field
scanners (1.5-3 T) are expensive to buy, site and run, which concentrates
them in large hospitals in high-income settings [CITE: global MRI access].
Low-field systems (below ~0.5 T, including recent portable scanners around
0.064 T) cost a fraction as much, need little shielding, and can be brought
to the bedside [CITE: Marques 2019; Sheth 2021]. The price is image quality:
signal grows roughly with field strength, so a low-field image has a lower
signal-to-noise ratio (SNR), and scanners compensate by acquiring larger
voxels, i.e. lower spatial resolution [CITE: Marques 2019].

Automated tumor segmentation networks, typically U-Net variants
[CITE: Ronneberger 2015], are trained and benchmarked almost entirely on
high-field data such as the BraTS challenge [CITE: Menze 2015; Bakas 2017].
If they are to be useful on low-field scanners, their accuracy must hold up
when SNR and resolution fall. A drop in overlap scores is only part of the
concern: tumor volume and shape measurements derived from the segmentation
are what inform clinical decisions (for example response assessment), so a
model that loses little Dice but systematically mis-sizes tumors would still
fail clinically.

### 1.2 Why decompose by spatial frequency

Noise and resolution loss do not affect all spatial frequencies equally. In
brain MRI, most image energy is at low spatial frequencies (smooth tissue
contrast, the overall shape of the tumor), while fine detail and edges sit at
high frequencies (Section 2.6 measures 92-97 % of clean-image power below
the cutoff used here). Thermal noise, in contrast, is spread evenly across
all frequencies. Lowering SNR therefore hurts the high-frequency band
proportionally far more than the low band, and resolution loss removes the
highest frequencies outright.

This motivates the hypothesis tested here: separating each image into a
low-frequency and a high-frequency band *before* the network, and letting the
network process them in separate streams, might let it rely on the robust
low band when the high band becomes unreliable.

### 1.3 Contributions

1. A controlled comparison of eight capacity-matched 2D U-Net
   configurations (Table 1) under 13 simulated low-field conditions
   (4 SNR levels x 3 resolution levels + clean) applied in k-space.
   `[TEAM: C1 - three configurations are currently duplicates in code; the
   final count and descriptions depend on that fix.]`
2. Pre-registered hypotheses H1-H3 (Section 2.10) with a paired
   non-parametric test design.
3. A morphometric validation layer: tumor volume, surface, sphericity,
   elongation and fragmentation for every prediction, with agreement
   analysis (Bland-Altman, ICC) that was itself validated on synthetically
   damaged masks with known errors.
4. An independent MATLAB re-implementation of the degradation and
   decomposition physics, numerically matched to the Python training code.

---

## 2. Methods

### 2.1 Dataset and splits

We used the BraTS 2020 training set [CITE: Menze 2015; Bakas 2017; Bakas
2018]: 369 pre-operative glioma patients (293 high-grade, HGG; 76
low-grade, LGG), each with four co-registered, skull-stripped MRI sequences
(T1, contrast-enhanced T1 [T1ce], T2, FLAIR) resampled to 1 mm isotropic
voxels on a 240 x 240 x 155 grid, and expert labels for necrotic/non-enhancing
core (label 1), edema (2) and enhancing tumor (4). Evaluation follows the
three BraTS regions: whole tumor (WT = 1 + 2 + 4), tumor core (TC = 1 + 4) and
enhancing tumor (ET = 4).

Patients were split once, stratified by grade with seed 42, into training
(258; 205 HGG / 53 LGG), validation (37; 29 / 8) and test (74; 59 / 15) sets
(`data/splits/split_v1.json`); the split was fixed before any model was
trained and never changed. The BraTS 2020 validation set was not used
(no public labels).

Each axial slice was cropped in-plane to 192 x 192 (rows 24-215, columns
28-219, zero-based). This crop retains at least 99.9 % of brain voxels for
every patient (checked in preprocessing) and loses 18 tumor voxels in total
across all 369 patients (one patient; 0.00005 % of tumor volume;
`matlab/check_crop_retention.m`).

Ground-truth tumor sizes in the test set (median, range): WT 81.3 cm^3
(16.6-230.9), TC 28.6 cm^3 (0.8-159.9), ET 13.2 cm^3 (0-79.0); ET is absent
in 5 test patients, all LGG (Section 2.7).

### 2.2 Simulated low-field degradation

Degradation is applied per 2D axial slice and per sequence, to raw
(un-normalised) intensities, in two steps (`python/physics/degradation.py`;
MATLAB port `matlab/reference/degrade_slice.m`).

**Resolution loss by k-space truncation.** The slice x is Fourier
transformed and centred, K = fftshift(FFT2(x)). A rectangular mask keeps the
central fraction r of k-space along each axis (half-widths round(r H / 2) and
round(r W / 2) around the centre, symmetric so the result stays real), and
zeroes the rest; the inverse transform gives the lower-resolution image
x_r = Re(IFFT2(ifftshift(K . M_r))). With r = 0.5 the effective in-plane
resolution is halved (2 mm). Because the transform is applied to the
magnitude image rather than raw complex scanner data, this is a
"pseudo-k-space" simulation, a common simplification.

**Thermal noise.** Independent zero-mean Gaussian noise of standard
deviation sigma is added to the real and imaginary channels and the
magnitude is taken:

  x_deg = sqrt((x_r + n_re)^2 + n_im^2),  n_re, n_im ~ N(0, sigma^2),

which yields Rician-distributed magnitudes in tissue and Rayleigh-distributed
background, as in real magnitude MR images [CITE: Gudbjartsson 1995]. The
noise level is set relative to the mean clean intensity of the brain for
that patient and sequence, sigma = mu_brain / SNR, where mu_brain is
computed over all brain voxels of the cropped volume. **SNR is a linear
ratio** (SNR = 8 means noise SD is 1/8 of the mean brain signal), not decibels.

**Conditions.** SNR in {30, 20, 12, 8} crossed with r in {1.0, 0.75, 0.5},
plus clean, gives 13 conditions. During training (all models except M0),
one condition is drawn uniformly at random per sample. For validation and
test, noise is deterministic: each (patient, slice, channel, condition)
tuple is hashed with SHA-256 and the first 4 bytes seed the noise generator,
so every model sees exactly the same noisy test images.

End-to-end check: measuring SNR directly in degraded slices (brain mean
over noise SD estimated from the Rayleigh background, sigma = mean(bg) /
sqrt(pi/2)) gives 8.1-9.8 at nominal SNR 8 (Section 2.9 cases).

`[TEAM: C4 - noise is added after truncation, so it occupies the full
spectrum including frequencies removed by truncation; a real low-resolution
acquisition would carry noise only in the acquired band. Either change the
order (noise in k-space before truncation) or state this in Limitations.]`

`[TEAM: C2 - at the strongest condition a clean-trained U-Net loses ~0.006
mean Dice on validation; decide whether to extend the range (lower SNR,
through-plane resolution loss) before final evaluation.]`

### 2.3 Frequency-band decomposition

Each (degraded, brain-z-score-normalised) slice is split into two bands with
complementary Gaussian filters in the centred 2D Fourier domain
(`python/physics/decomposition.py`; MATLAB `decompose_slice.m`):

  D(u, v) = sqrt((u / (H/2))^2 + (v / (W/2))^2)     (normalised radius, 1 = Nyquist along an axis)
  H_lp = exp(-D^2 / (2 D0^2)),   H_hp = 1 - H_lp
  x_low = Re(IFFT2(F . H_lp)),   x_high = Re(IFFT2(F . H_hp))

with D0 = 0.20. A Gaussian rather than an ideal (hard circular) cutoff avoids
ringing artefacts. Because H_lp + H_hp = 1 at every frequency, the bands sum
exactly to the input (maximum reconstruction error 5.1e-7, float32
rounding), so the decomposition discards no information; it only rearranges
it.

`[TEAM: C5 - the D0 selection evidence (proxy Dice ~0.03 on 3 validation
patients) does not support a data-driven choice; describe D0 = 0.20 as a
design choice motivated by the spectral analysis (Section 2.6), optionally
with a post-hoc sensitivity analysis.]`

### 2.4 Models (outline - Python workstream)

All models are 2D U-Nets with five resolution levels, instance normalisation
and LeakyReLU(0.2), outputting three sigmoid channels (WT, TC, ET), with
parameter counts matched within 4.5 % (4.17-4.37 M; `tables/model_complexity.csv`).

| Model | Intended design | Params |
|---|---|---|
| M0 | U-Net, trained on clean data only | 4.37 M |
| M1 | U-Net + degradation augmentation | 4.37 M |
| M2 | NLM denoiser + U-Net | 4.37 M |
| M3 | low-pass + CLAHE + U-Net | 4.37 M |
| M4 | **dual-stream U-Net** (low and high band encoders, shared decoder), D0 = 0.20 | 4.17 M |
| M5 | low band only (ablation) | 4.37 M |
| M6 | early fusion: low + high as 8 input channels | 4.37 M |
| M7 | dual-stream with enhanced high band | 4.17 M |

`[TEAM: C1 - in the current code M2 = M1, M3 = M5 and M7 = M4 (identical
forward passes). Implement or remove before reporting.]`

Training: AdamW, learning rate 3e-4 with 1-epoch linear warm-up and cosine
decay to 1e-6, soft Dice + BCE loss, batch size 32, 10 largest-tumor slices
per training patient, 5 epochs, seed 0 (`docs/DEVIATIONS.md`).
`[TEAM: C7 - configs/base.yaml specifies 20 epochs and seeds {0, 1, 2}; only
seed 0 at 5 epochs was run. With expected differences of ~0.005 Dice, at
least 3 seeds are needed to separate model effects from training noise.]`

### 2.5 Implementation verification

The degradation and decomposition were implemented independently in MATLAB
(`degrade_slice.m`, `decompose_slice.m`) and compared with the PyTorch code
on five random 192 x 192 slices with shared noise arrays, across all 12
degraded conditions and four cutoffs D0 in {0.10, 0.15, 0.20, 0.30}. The
Python test (`tests/test_degradation.py::test_matlab_python_parity`) requires
a maximum absolute difference below 1e-4 `[TEAM: report the measured value
from that test run]`. Rerunning the MATLAB side on a second machine and
MATLAB version (R2026b vs R2025a) reproduced the stored MATLAB outputs with
a maximum difference of exactly 0 (`matlab/env_check.m`).

### 2.6 Spectral analysis

To quantify how the degradation distributes across frequency bands, we
analysed the axial slice with the largest tumor area for the first 12 test
patients (FLAIR and T1ce; same 192 x 192 crop and noise calibration as
training; `matlab/spectral_check.m`, noise seed 42).

**Where the image energy is.** The fraction of clean-image spectral power at
normalised radius D < 0.20 was 95.1 % for FLAIR (minimum 92.5 %) and 91.9 %
for T1ce (minimum 88.4 %), excluding the zero-frequency (DC) term
(96.9 % and 95.2 % including it). Under the smooth Gaussian split actually
used by the models, the low band carries 97.8 % (FLAIR) and 96.6 % (T1ce) of
the band energy.

**How the degradation hits each band.** For each condition we computed a
band SNR: clean-image power in the band divided by the power of the
degradation error (degraded minus clean image, which contains both noise and
lost resolution) in the same band, in dB.

| Nominal SNR | r | FLAIR low / high band (dB) | T1ce low / high band (dB) |
|---|---|---|---|
| 30 | 1.0 | 28.0 / 14.1 | 26.6 / 14.8 |
| 12 | 0.75 | 19.9 / 4.9 | 18.5 / 5.6 |
| 8 | 1.0 | 16.5 / 2.7 | 15.1 / 3.4 |
| 8 | 0.5 | 16.4 / **1.0** | 15.1 / **1.7** |

(All 12 conditions: `results/spectral_check.csv`, Figure
`figures/spectral_band_snr.png`.)

The low band stays clean throughout (>= 15 dB, i.e. the error is under 3 %
of the anatomical power), while the high band degrades by 12-13 dB from the
mildest to the strongest condition, ending close to, but still above,
0 dB. Resolution loss mainly affects the high band (FLAIR at SNR 30: 14.1,
9.3 and 5.5 dB for r = 1, 0.75, 0.5) and leaves the low band almost
untouched.

Two consequences for interpreting the results: (i) the premise of band
separation holds - degradation is concentrated in the high band; (ii) even
the strongest condition leaves the high band with more anatomical signal than
error *in aggregate*, and leaves the low band nearly pristine, which is
consistent with the small Dice losses observed for all models (C2).
A caveat: the high-band anatomical power includes the sharp brain/background
edge of skull-stripped images, which inflates high-band SNR relative to the
tumor interior.

*Note for the team:* `docs/FINDINGS_SPECTRAL.md` states that the high band
"drops below 0 dB" at SNR 8-12 and r <= 0.75. Measured band-aggregate values
are +1.0 to +5.6 dB in those conditions, so that sentence should not go into
the paper. Those statements are hard-coded text in `spectral_analysis.m`
rather than computed values.

### 2.7 Morphometric biomarkers

For every segmentation (ground truth or prediction, identical code:
`matlab/tumor_biomarkers.m`) and each region (WT, TC, ET) we computed:
volume (voxel count x voxel volume from the NIfTI header), surface area,
sphericity psi = pi^(1/3) (6V)^(2/3) / A [CITE: Wadell 1935], elongation
(ratio of the second to the first principal axis length), the number of
26-connected components, and the fraction of volume outside the largest
component. Empty regions are recorded as volume 0 with undefined (NaN) shape
measures.

Surface area on a voxel grid needs care: counting exposed voxel faces
overestimates the true area of a smooth surface by 45-55 % regardless of
size (validated on digital spheres, `figures/surface_area_validation.png`).
We used MATLAB's `regionprops3` SurfaceArea, which was within 1.7 % of the
analytic area for spheres of radius >= 8 mm, and cross-checked it with a
marching-cubes mesh [CITE: Lorensen 1987] after Gaussian smoothing
(sigma = 0.5 voxel; +2-4 % on spheres). Because `regionprops3` reports the
surface of only one component when a single label spans several
disconnected components, surface area was computed per connected component
and summed (regression-tested). The two estimators then agreed closely on
the 74 test tumors (Spearman rho 0.992-0.996 for sphericity).

### 2.8 Agreement analysis

Agreement between predicted and ground-truth biomarkers is reported per
model and condition as (i) percentage volume error, (ii) Bland-Altman bias
and 95 % limits of agreement [CITE: Bland 1986], and (iii) the intraclass
correlation ICC(A,1) (two-way, absolute agreement, single measure;
[CITE: McGraw 1996], equivalent to ICC(2,1) of [CITE: Shrout 1979]),
alongside the consistency form ICC(C,1). Confidence intervals are
percentile bootstrap intervals over patients (2000 resamples, seed 42).
The implementation (`matlab/agreement_stats.m`) reproduces the worked
example of Shrout & Fleiss (ICC(2,1) = 0.29, ICC(3,1) = 0.71).

The full pipeline was validated before use on real predictions by applying
known synthetic errors to the 74 test ground truths (dilation/erosion by 1
and 2 mm, a 2-voxel shift, deletion of an 8 mm ball, three spurious 3 mm
blobs; `matlab/agreement_validation.m`). Exactly known volume changes were
recovered to the voxel; dilation-induced volume changes matched a
surface-area (Steiner) prediction calibrated on a sphere to within 1-3 %
(median ratio 1.006-1.032).

### 2.9 Qualitative case selection

Five test patients were selected for qualitative display before any
predictions were inspected, using only ground-truth WT volume, ET presence
and grade: the HGG patients (with ET) closest to the 10th, 50th and 90th
percentile of HGG WT volume, the LGG patient with ET closest to that
subgroup's median, and the LGG patient without ET closest to that subgroup's
median (ties: lowest ID). Selected: BraTS20_Training_008, _167, _236, _292,
_281 (`results/diagnostic_selection.csv`).

### 2.10 Hypotheses and statistics (pre-specified)

- **H1** M4 exceeds M0 by >= 0.05 mean Dice at SNR = 8.
- **H2** M4's clean-to-SNR-8 Dice drop is smaller than M0's (paired
  Wilcoxon signed-rank over the 74 test patients, p < 0.05).
- **H3** Frequency decomposition alone (M5) accounts for >= 50 % of M4's
  improvement relative to M6. `[TEAM: M6 also uses both bands, so "M4
  without decomposition" is not M6; restate H3 before computing it.]`

Per-patient paired comparisons use the Wilcoxon signed-rank test with Holm
correction across comparisons [CITE: Wilcoxon 1945; Holm 1979]. Segmentation
metrics: Dice per region and 95th-percentile Hausdorff distance (mm).

`[TEAM: C3 - M4 is trained with degradation augmentation, M0 without. A
robustness gain of M4 over M0 confounds architecture with training data;
M4 vs M1 isolates the architecture. Consider adding that comparison.]`

---

## 3. Results  [RESULT]

To be written after Stage 3-4. Planned structure:
3.1 Segmentation accuracy across the 13 conditions (Dice, HD95 curves).
3.2 Hypothesis tests H1-H3.
3.3 Morphometric agreement per model and condition (volume error, Bland-
Altman, ICC; fragmentation).
3.4 Qualitative cases and 3D comparisons (M0 vs M4).

## 4. Discussion - points already established

- **ET is intrinsically fragile.** A 1 mm boundary error changes ET volume
  by a median 48 % but WT by 17 % (synthetic validation), because ET is a
  thin, small shell with a high surface-to-volume ratio. Expect ET volume and
  Dice to degrade first, independent of model.
- **Overlap, volume and false positives are different questions.** In the
  synthetic validation a 2 mm shift kept ICC = 1.000 while ET Dice fell to
  0.71, and three spurious blobs kept Dice at 0.997 and ICC at 1.000; only the
  component count detected them. Report all three kinds of metric.
- **ICC can flatter a biased method.** A systematic +17 % volume error still
  gave ICC(A,1) = 0.957; report Bland-Altman bias alongside ICC.

## 5. Limitations (known now)

- Simulated, 2D in-plane degradation; real low-field scans also have thick
  slices, different contrast, and field inhomogeneity.
- Noise model order (C4).
- Single training seed and short training (C7).
- Test noise realisations in MATLAB figures differ from the Python test set
  (same statistics).

---

## References (all to verify before submission)

- Bakas S, et al. Advancing The Cancer Genome Atlas glioma MRI collections
  with expert segmentation labels and radiomic features. *Scientific Data*
  2017. (verify)
- Bakas S, et al. Identifying the best machine learning algorithms for brain
  tumor segmentation, progression assessment, and overall survival
  prediction in the BRATS challenge. arXiv:1811.02629, 2018. (verify)
- Bland JM, Altman DG. Statistical methods for assessing agreement between
  two methods of clinical measurement. *Lancet* 1986. (verify)
- Gudbjartsson H, Patz S. The Rician distribution of noisy MRI data.
  *Magnetic Resonance in Medicine* 1995. (verify)
- Holm S. A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics* 1979. (verify)
- Lorensen WE, Cline HE. Marching cubes: a high resolution 3D surface
  construction algorithm. *SIGGRAPH* 1987. (verify)
- Marques JP, Simonis FFJ, Webb AG. Low-field MRI: an MR physics perspective.
  *Journal of Magnetic Resonance Imaging* 2019. (verify)
- McGraw KO, Wong SP. Forming inferences about some intraclass correlation
  coefficients. *Psychological Methods* 1996. (verify)
- Menze BH, et al. The multimodal brain tumor image segmentation benchmark
  (BRATS). *IEEE Transactions on Medical Imaging* 2015. (verify)
- Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for
  biomedical image segmentation. *MICCAI* 2015. (verify)
- Sheth KN, et al. Assessment of brain injury using portable, low-field
  magnetic resonance imaging at the bedside of critically ill patients.
  *JAMA Neurology* 2021. (verify)
- Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater
  reliability. *Psychological Bulletin* 1979. (verify)
- Wadell H. Volume, shape, and roundness of quartz particles. *Journal of
  Geology* 1935. (verify)
- Wilcoxon F. Individual comparisons by ranking methods. *Biometrics
  Bulletin* 1945. (verify)
- [CITE: global MRI access] - a source on MRI scanner density by country
  income level is still needed.
