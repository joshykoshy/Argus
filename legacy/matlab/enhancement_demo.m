%% Enhancement Demo: CLAHE, Top-Hat, Sobel, and Canny Filtering
% ECTE408 Low-Field MRI Brain Tumor Segmentation Study
% Demonstrates spatial and frequency-domain enhancement techniques side by side.

clear; clc; close all;

%% 1. Setup & Load Sample
project_root = 'C:/Users/Mayan/.gemini/antigravity/scratch/ecte408';
mat_file = fullfile(project_root, 'data', 'spectral_samples.mat');
figures_dir = fullfile(project_root, 'figures');
ref_dir = fullfile(project_root, 'matlab', 'reference');

addpath(ref_dir);

data = load(mat_file);
slices = single(data.slices);
mu_brains = single(data.mu_brains);

% Choose test case 1, FLAIR modality (ch 4) and T1ce modality (ch 2)
img_flair_clean = squeeze(slices(1, 4, :, :));
img_t1ce_clean  = squeeze(slices(1, 2, :, :));
mu_flair = mu_brains(1, 4);
mu_t1ce  = mu_brains(1, 2);

% Degraded realization: SNR 12, r = 0.75
img_flair_deg = degrade_slice(img_flair_clean, mu_flair, 12.0, 0.75);
img_t1ce_deg  = degrade_slice(img_t1ce_clean, mu_t1ce, 12.0, 0.75);

% Frequency decomposition: D0 = 0.20
[flair_low, flair_high] = decompose_slice(img_flair_deg, 0.20);
[t1ce_low, t1ce_high]   = decompose_slice(img_t1ce_deg, 0.20);

%% 2. Apply Classical Spatial Enhancements
% Normalize to [0, 1]
norm_flair_clean = (img_flair_clean - min(img_flair_clean(:))) / (max(img_flair_clean(:)) - min(img_flair_clean(:)) + 1e-6);
norm_flair_deg   = (img_flair_deg - min(img_flair_deg(:))) / (max(img_flair_deg(:)) - min(img_flair_deg(:)) + 1e-6);

% CLAHE
clahe_clean = adapthisteq(norm_flair_clean, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
clahe_deg   = adapthisteq(norm_flair_deg, 'ClipLimit', 0.02, 'NumTiles', [8 8]);

% Top-Hat morphological filtering (radius 3 disk)
se = strel('disk', 3);
tophat_clean = imtophat(norm_flair_clean, se);
tophat_deg   = imtophat(norm_flair_deg, se);

% Edge detection: Sobel & Canny
edge_sobel_clean = edge(norm_flair_clean, 'sobel');
edge_sobel_deg   = edge(norm_flair_deg, 'sobel');
edge_canny_clean = edge(norm_flair_clean, 'canny');
edge_canny_deg   = edge(norm_flair_deg, 'canny');

%% 3. Plot Comparison Grid
fig = figure('Position', [100, 100, 1200, 900], 'Visible', 'off');

% Row 1: Clean vs Degraded vs Bands
subplot(3, 4, 1); imshow(norm_flair_clean, []); title('Clean FLAIR (r=1.0)', 'FontSize', 10, 'FontWeight', 'bold');
subplot(3, 4, 2); imshow(norm_flair_deg, []);   title('Degraded (SNR 12, r=0.75)', 'FontSize', 10, 'FontWeight', 'bold');
subplot(3, 4, 3); imshow(flair_low, []);        title('Low-Pass Band (D0=0.20)', 'FontSize', 10, 'FontWeight', 'bold');
subplot(3, 4, 4); imshow(flair_high, []);       title('High-Pass Band (D0=0.20)', 'FontSize', 10, 'FontWeight', 'bold');

% Row 2: Contrast & Morphological Enhancement
subplot(3, 4, 5); imshow(clahe_clean, []);   title('CLAHE (Clean)', 'FontSize', 10, 'FontWeight', 'bold');
subplot(3, 4, 6); imshow(clahe_deg, []);     title('CLAHE (Degraded)', 'FontSize', 10, 'FontWeight', 'bold');
subplot(3, 4, 7); imshow(tophat_clean, []);  title('White Top-Hat (Clean)', 'FontSize', 10, 'FontWeight', 'bold');
subplot(3, 4, 8); imshow(tophat_deg, []);    title('White Top-Hat (Degraded)', 'FontSize', 10, 'FontWeight', 'bold');

% Row 3: Edge Detection
subplot(3, 4, 9);  imshow(edge_sobel_clean, []); title('Sobel (Clean)', 'FontSize', 10, 'FontWeight', 'bold');
subplot(3, 4, 10); imshow(edge_sobel_deg, []);   title('Sobel (Degraded - Noise sensitive)', 'FontSize', 10, 'FontWeight', 'bold');
subplot(3, 4, 11); imshow(edge_canny_clean, []); title('Canny (Clean)', 'FontSize', 10, 'FontWeight', 'bold');
subplot(3, 4, 12); imshow(edge_canny_deg, []);   title('Canny (Degraded)', 'FontSize', 10, 'FontWeight', 'bold');

sgtitle('Spatial & Frequency-Domain Image Enhancement Under Simulated Low-Field MRI', 'FontSize', 13, 'FontWeight', 'bold');

saveas(fig, fullfile(figures_dir, 'enhancement_demo.png'));
close(fig);

fprintf('Enhancement demo generated successfully: figures/enhancement_demo.png\n');
