# Spectral Power & Band-SNR Diagnostic Findings

Based on empirical radial power spectral density (PSD) analysis across 12 test-split BraTS 2020 patients:

1. Clean brain MRI spectra exhibit exponential power roll-off with frequency, concentrating over 92% of total anatomical signal power in the low radial band (D < 0.20).
2. At high SNR (30 dB), true anatomical edge and boundary high-frequency signal remains measurable above the noise floor across the spectrum.
3. Under severe low-field degradation (SNR 8 to 12, r <= 0.75), the high-frequency band (D > 0.20) drops below 0 dB SNR, meaning noise power exceeds anatomical signal power.
4. Pseudo k-space truncation at r = 0.5 creates a sharp spectral cliff at Nyquist/2, with high frequencies dominated exclusively by post-truncation Rician noise.
5. Consequently, dual-stream models must learn to extract macro-structural contrast from the low band while selectively filtering noise or attenuating high-frequency stream weights under low SNR conditions.
