% FIGURE_PIPELINE_PHYSICS  Explainer figure: degradation and band split, in
% image space and in k-space, on one real slice. Candidate paper Figure 1.
%
% Run from repo root:  matlab -batch "run('matlab/figure_pipeline_physics.m')"
%
% Top row (image space):    clean -> resolution loss (r = 0.5) -> + noise (SNR 8)
%                           -> z-score -> low band -> high band
% Bottom row (k-space, log power): the same six images' 2D spectra, with the
% truncation box and the Gaussian D0 = 0.20 cutoff drawn on.
% Uses the parity-verified degrade_slice.m / decompose_slice.m, patient
% BraTS20_Training_167 (diagnostic case 2, the median HGG), seed 42.

cd(fileparts(fileparts(mfilename('fullpath'))));
addpath('matlab/reference');
DATA_ROOT = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';
ID = 'BraTS20_Training_167';
ROWS = 25:216;  COLS = 29:220;  SNR = 8;  R = 0.5;  D0 = 0.20;  SEED = 42;

fl  = single(niftiread(fullfile(DATA_ROOT, ID, [ID '_flair.nii'])));
seg = niftiread(fullfile(DATA_ROOT, ID, [ID '_seg.nii']));
fl = fl(ROWS, COLS, :);  seg = seg(ROWS, COLS, :);
mu = mean(fl(fl > 0));
[~, z] = max(squeeze(sum(sum(seg > 0, 1), 2)));
x = fl(:, :, z);  bm = x > 0;
rs = RandStream('mt19937ar', 'Seed', SEED);
nr = randn(rs, size(x), 'single');  ni = randn(rs, size(x), 'single');

xr  = degrade_slice(x, mu, [], R);              % truncation only
xd  = degrade_slice(x, mu, SNR, R, nr, ni);      % truncation + noise
zs  = @(im) (im - mean(im(bm))) / std(im(bm));
xz  = zs(xd);
[lo, hi] = decompose_slice(xz, D0);

ims = {zs(x), zs(xr), xz, xz, lo, hi};
ttl = {'1  Clean FLAIR', sprintf('2  k-space truncated (r = %.1f)', R), ...
       sprintf('3  + Rician noise (SNR %d)', SNR), '4  z-scored = network input', ...
       sprintf('5  Low band (D_0 = %.2f)', D0), '6  High band'};
lims = {[-2 4], [-2 4], [-2 4], [-2 4], [-2 4], [-1.5 1.5]};
cap = {'what a 3 T scanner gives', 'outer k-space removed: blur', ...
       'noise added: grain', 'same image, normalised', ...
       'smooth contrast, tumor shape', 'edges + most of the noise'};

H = 192;  ch = H / 2;
[V, U] = meshgrid((0:H-1) - ch, (0:H-1) - ch);
D = sqrt((U / ch).^2 + (V / ch).^2);
kr = round(H * R / 2);

set(groot, 'defaultAxesFontSize', 10, 'defaultTextFontSize', 10);
f = figure('Visible', 'off', 'Position', [30 30 1700 640]);  theme(f, 'light');
tl = tiledlayout(f, 2, 6, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:6
    ax = nexttile(i);
    imshow(rot90(ims{i}), lims{i}, 'Parent', ax);
    title(ax, ttl{i}, 'FontWeight', 'bold', 'FontSize', 11);
    xlabel(ax, cap{i});  ax.XLabel.Visible = 'on';
end
for i = 1:6
    ax = nexttile(6 + i);
    P = 10 * log10(abs(fftshift(fft2(ims{i}))).^2 + 1e-3);
    imshow(P, [-10 70], 'Parent', ax);  colormap(ax, parula);  hold(ax, 'on');
    if i >= 2         % truncation box
        rectangle(ax, 'Position', [ch - kr + 1, ch - kr + 1, 2 * kr, 2 * kr], ...
            'EdgeColor', 'w', 'LineStyle', '--', 'LineWidth', 1);
    end
    if i >= 5         % Gaussian cutoff: D = D0 circle (filter at exp(-1/2) = 61%)
        t = linspace(0, 2 * pi, 200);
        plot(ax, ch + 1 + D0 * ch * cos(t), ch + 1 + D0 * ch * sin(t), 'r-', 'LineWidth', 1.5);
    end
    if i == 1, ylabel(ax, 'k-space (log power)');  ax.YLabel.Visible = 'on'; end
end
annotation(f, 'textbox', [0 0.005 1 0.04], 'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
    'FontSize', 10, 'String', ['Bottom row: centre = low spatial frequency, edges = high. ' ...
    'White dashed box = k-space kept at r = 0.5. Red circle = D_0 = 0.20 (Gaussian split). ' ...
    'Panel 3: noise fills ALL of k-space, including outside the box (see C4).']);
exportgraphics(f, 'figures/pipeline_physics.png', 'Resolution', 200);  close(f);
fprintf('Saved figures/pipeline_physics.png (%s, slice %d)\n', ID, z);
