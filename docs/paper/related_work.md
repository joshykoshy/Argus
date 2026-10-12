# Related Work

Citation keys in `[@key]` form refer to `docs/paper/references.bib`. Every cited paper was opened (abstract or
landing page) while writing this section; see `docs/paper/references_check.md` for what "opened" covers.

## Low-field and portable MRI

Conventional 1.5-3 T scanners are expensive to install and operate, which makes them scarce outside
high-income countries and major population centres [@arnold2022lowfield]. Low-field systems, defined by Marques
et al. as 0.25-1 T, trade cost for signal: the signal-to-noise ratio (SNR) depends on field strength and this
constrains the achievable contrast, resolution and acquisition time [@marques2019lowfield]. Portable scanners
push this further. A portable low-field scanner has been used at the bedside of 50 intensive-care patients
[@sheth2021portable], and 0.064 T portable systems are limited by low spatial resolution and low SNR
[@iglesias2023superres].

The access gap that motivates these devices is documented. A survey in West Africa found 84 MRI units serving
about 373 million people, and 45 of the 58 units in Nigeria were low-field systems [@ogbole2018westafrica]. The
CAMERA consortium describes MRI as largely inaccessible in low-resource areas such as Africa
[@anazodo2023camera], and a broader review frames limited MRI access as a global disparity across low- and
middle-income countries [@murali2023lmic]. Low-field devices have drawn interest because their lower cost and
portability could increase imaging access [@arnold2022lowfield]; whether automated analysis tools keep working
on their lower-quality images is the question this study addresses.

Most computational work on low-field images improves the image before analysis. Iglesias et al. trained a
super-resolution network to synthesise 1 mm isotropic images from 0.064 T scans so that standard morphometry
tools could be applied; in the eleven participants analysed, direct segmentation of the low-field scans gave
nonsignificant correlations with the high-field reference [@iglesias2023superres]. That study concerns
brain-region morphometry, not tumours.

## Deep-learning brain tumour segmentation

The BraTS benchmark [@menze2015brats] and its glioma label releases [@bakas2017tcga] underpin most work on
glioma segmentation; the 2018 retrospective analysis covers methods submitted from 2012 to 2018
[@bakas2018brats]. Menze et al. reported inter-rater Dice between 74 % and 85 % for tumour sub-regions, which
is the reference point for what a segmentation model can be expected to reach [@menze2015brats]. Most methods
descend from the U-Net [@ronneberger2015unet]. The self-configuring nnU-Net [@isensee2021nnunet] and its BraTS
2020 variant, which won that year's challenge with region-based training and stronger data augmentation, reached
Dice of about 88.95, 85.06 and 82.03 for whole tumour, tumour core and enhancing tumour
[@isensee2020bratsnnunet]. These results are for BraTS data. We did not find, in the sources read for this
section, a glioma-segmentation evaluation on low-field scans; the closest prior work concerns anatomy
[@iglesias2023superres].

## Robustness to noise, resolution and domain shift

Performance loss under acquisition shift is well documented for medical segmentation networks
[@karani2021testtime]. Three families of remedies are relevant here. *Augmentation:* Zhang et al. simulate the
expected domain shift of an imaging modality with a stack of image-level transformations applied to a single
source domain, and report generalisation to unseen domains without retraining on the target
[@zhang2020bigaug]; the strongest BraTS 2020 pipeline also relied on heavier augmentation
[@isensee2020bratsnnunet]. *Test-time adaptation:* Karani et al. place a small normalisation sub-network in front
of the segmenter and adapt it per test image, guided by a denoising autoencoder trained on plausible label maps
[@karani2021testtime]. *Image restoration:* super-resolution, as above, repairs the input rather than the
network [@iglesias2023superres].

Evidence from natural-image classification warns that the answer depends on where the corruption lives in
frequency. Gaussian-noise augmentation improves robustness to high-frequency corruptions but reduces it for
low-frequency ones, and a more diverse augmentation set can mitigate the trade-off [@yin2019fourier]. Our
degradation, which adds noise and removes high spatial frequencies, falls mostly in the first category.

## Frequency-domain and multi-band networks

Wang et al. link the generalisation of CNNs to the high-frequency content of images, including components
almost imperceptible to humans, and discuss a robustness-accuracy trade-off [@wang2020highfreq]. Xu et al.
feed frequency-domain input to standard architectures and learn which frequency components to keep, improving
ImageNet accuracy at equal or reduced input size [@xu2020frequency]. In medical segmentation, SF-UNet adds a
lightweight spatial-frequency attention block to learn texture and boundary features, arguing that
spatial-only networks are biased towards low frequencies [@zhou2024sfunet]. HFF-Net is closest to our design:
it splits brain MRI into low-frequency components (smooth tumour contours) and high-frequency components
(textures and edges), emphasises the high band for boundary sensitivity, and reports a 4.48 % average relative
Dice gain over baselines on four public datasets [@shao2025hffnet]. As far as the abstracts we read show,
these designs are motivated and evaluated by accuracy on standard-quality data; none tests whether the
decomposition helps when the image is deliberately degraded in a controlled way.

## Volumetric agreement in tumour assessment

Clinical response assessment uses the RANO criteria [@wen2010rano], which mainly rely on manual
two-dimensional measurements of contrast-enhancing lesions; this restricts both reliability and accurate
assessment of tumour burden, which motivates automated volumetry [@kickingereder2019automated]. Automated volumetry has been compared against manual measurement using
intraclass correlation: Chang et al. report ICC of 0.915 to 0.965 between manual and automated tumour volumes
and above 0.97 for repeatability on double-baseline scans [@chang2019autorano]. Agreement should not be
summarised by correlation alone, because correlation is misleading when judging whether one method can replace
another; Bland and Altman proposed limits of agreement instead [@bland1986agreement], and the form of ICC
used must be chosen to match the rating design [@shrout1979icc]. A segmentation can keep a high Dice while
mis-sizing tumours, so volume-level agreement is a separate question from overlap.

## The gap this study addresses

Work on low-field MRI mostly repairs image quality or builds hardware, frequency-aware tumour networks are
evaluated on clean, high-quality data, and augmentation methods are rarely compared against architectural
alternatives under the same simulated acquisition conditions. This study compares a clean-trained U-Net,
degradation-augmented variants and frequency-band architectures under one pre-registered set of simulated
low-field degradations, with paired statistics over test patients and training seeds, and links the Dice
results to tumour volume agreement.

---

## Notes for the team (not part of the section)

- **Not cited because not opened:** search snippets pointed to 2025-2026 preprints on BraTS image-corruption
  robustness and to other frequency-domain segmentation networks (FMISeg, SLf-UNet, FRCNet). Open them before
  adding. They are the most likely missing citations for the robustness paragraph, since the section currently
  has no brain-tumour-specific noise-robustness paper.
- **"We did not find a glioma evaluation on low-field scans"** is a statement about the search done for this
  task, not a proof. Repeat a PubMed search ("low-field" / "portable MRI" with glioma segmentation) before
  submission and soften or remove the sentence accordingly.
- **Result-dependent wording:** the gap paragraph is deliberately neutral. With the real results
  (`docs/RESULTS_STATS.md`: augmentation is the effect; two-stream M4 adds nothing detectable over M1 or M6),
  the paper's framing becomes "a controlled negative result for decomposition", and the final sentence of
  this section should say so in the Discussion rather than here.
- **Word count** of the body above is about 1,000.
