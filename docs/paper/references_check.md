# Reference verification (Task 3, part 1)

Checked 2026-10-12 against Crossref, PubMed and arXiv records. "Opened" means the abstract or landing page was
read; "metadata only" means the bibliographic details were confirmed but the paper's content was not read.

## References listed in `docs/PAPER_DRAFT.md`

| Reference as drafted | Result | Evidence | Status |
|---|---|---|---|
| Bakas 2017, *Scientific Data* | Correct | doi 10.1038/sdata.2017.117, vol 4, art. 170117 | opened |
| Bakas 2018, arXiv:1811.02629 | Correct; title on arXiv has the quotation marks around two parts, use the plain title | arXiv abs page, 427 authors, v1 Nov 2018 | opened |
| Bland & Altman, *Lancet* 1986 | Correct; add vol 327(8476), 307-310 | doi 10.1016/S0140-6736(86)90837-8; PubMed 2868172 | opened (abstract) |
| Gudbjartsson & Patz, *MRM* 1995 | Correct; add 34(6), 910-914 | doi 10.1002/mrm.1910340618; PubMed 8598820 | opened (abstract) |
| Holm, *Scand J Stat* 1979 | Correct; vol 6(2), 65-70. No Crossref DOI record (JSTOR 4615733) | Details agree across several secondary sources; no primary record opened | metadata only |
| Lorensen & Cline, *SIGGRAPH* 1987 | Correct; published in *Computer Graphics* 21(4), 163-169 | doi 10.1145/37402.37422 | metadata only |
| Marques, Simonis, Webb, *JMRI* 2019 | Correct; 49(6), 1528-1542. **See note 1** | doi 10.1002/jmri.26637 | opened (abstract) |
| McGraw & Wong, *Psychol Methods* 1996 | Correct; 1(1), 30-46 | doi 10.1037/1082-989X.1.1.30 | metadata only |
| Menze et al., *IEEE TMI* 2015 | Correct; 34(10), 1993-2024 | doi 10.1109/TMI.2014.2377694 | opened (abstract) |
| Ronneberger, Fischer, Brox, *MICCAI* 2015 | Correct; LNCS pp. 234-241 | doi 10.1007/978-3-319-24574-4_28; arXiv:1505.04597 | opened (arXiv abstract) |
| Sheth et al., *JAMA Neurology* 2021 | Correct; 78(1), starts p. 41 | doi 10.1001/jamaneurol.2020.3263 | opened (abstract) |
| Shrout & Fleiss, *Psychol Bull* 1979 | Correct; 86(2), 420-428 | doi 10.1037/0033-2909.86.2.420; PubMed 18839484 | opened (abstract) |
| Wadell, *J Geology* 1935 | Correct; 43(3), 250-280 | doi 10.1086/624298 | metadata only |
| Wilcoxon, *Biometrics Bulletin* 1945 | Correct; 1(6), starts p. 80 | doi 10.2307/3001968 | metadata only |

**No reference was wrong.** The draft's own warning that titles and years "were written from memory" turned out
to be unfounded for all 14 entries. Missing detail (volumes, pages, DOIs) is filled in `references.bib`.

### Note 1: low-field definition in Section 1.1 (needs a text fix)

The draft says low-field systems are "below ~0.5 T, including recent portable scanners around 0.064 T" and cites
Marques 2019 and Sheth 2021. Marques et al. define low field as **0.25-1 T** (their abstract), not "below 0.5 T",
and the 0.064 T figure is stated in Iglesias et al. 2023 (Radiology), not in the Marques abstract. The suggested
wording is applied to `PAPER_DRAFT.md`.

### Note 2: citation-use checks

- Menze 2015 and Bakas 2017/2018 are cited in Section 2.1 for the BraTS 2020 dataset. This is the usual BraTS
  citation trio, but none of the three describes the 2020 release specifically; the BraTS organisers ask for
  exactly these three, so keep them.
- Section 2.8 reports ICC(A,1) with McGraw & Wong and Shrout & Fleiss. Shrout & Fleiss (abstract read) covers six
  ICC forms; the A,1 absolute-agreement form is McGraw & Wong's terminology. McGraw & Wong's content was not
  opened here, so that mapping is unconfirmed.

## The open placeholder: `[CITE: global MRI access]`

Resolved with three sources, all opened:

- **Ogbole et al. 2018**, *Pan African Medical Journal* 30:240, doi 10.11604/pamj.2018.30.240.14000. Survey
  (Sept 2015 to Sept 2016) found 84 MRI units in West Africa for a population of 372,551,411 (abstract figure; the
  results text says "more than 350 million"). Ghana 14 units (0.48 per million), Nigeria 58 units (0.30 per million),
  45 of Nigeria's units (77.6 %) low-field. The paper does not state a sub-region rate; 84 / 372.6 M is about
  0.23 per million (our arithmetic).
- **Anazodo et al. 2023**, CAMERA consortium, *NMR in Biomedicine* 36(3):e4846, doi 10.1002/nbm.4846. Abstract:
  MRI remains largely inaccessible in low-resource areas such as Africa. A related interview (Nat Commun 2024,
  doi 10.1038/s41467-024-46567-3) quotes "less than one MRI scanner per million people across Sub-Saharan Africa"
  citing this survey; that figure was read in the interview, **not** in the NMR Biomed abstract, so do not
  attribute the number to the NMR Biomed paper without opening its full text.
- **Murali et al. 2023**, *NMR in Biomedicine* 37(7):e4992, doi 10.1002/nbm.4992. Abstract frames the global
  disparity in MRI and limited LMIC access.

The draft's claim that high-field scanners are "concentrated in large hospitals in high-income settings" is also
supported by Arnold et al. 2023 (*JMRI* 57(1):25-44, doi 10.1002/jmri.28408): high-field devices are "scarce
outside of high-income countries and major population centers".

## What was not checked

- The full text of any paper. Claims in `related_work.md` are limited to what abstracts and landing pages state.
- Sheth 2021's abstract was read only up to the results; the 0.064 T field strength is not attributed to it.
- Candidate papers found only through search snippets (see the notes at the end of `related_work.md`).
