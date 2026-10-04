% ENV_CHECK  One-shot environment check for the MATLAB workstream.
%
% Run from the repo root:  matlab -batch "run('matlab/env_check.m')"
%
% 1. Toolbox: Image Processing Toolbox must be licensed (regionprops3,
%    imgaussfilt3, bwdist, niftiread all live there).
% 2. Parity: rerun decompose_slice / degrade_slice on the shared test slices
%    (parity_test_data.mat, generated in Python) and compare against the
%    reference results (parity_matlab_results.mat). Tolerance 1e-5.
% 3. Data: every test ID in split_v1.json has all 5 NIfTI files, and the
%    header voxel spacing is read (not assumed) and reported.

%% ---- Configuration (paths relative to repo root) ----
% run() cd's into the script folder, so anchor on the repo root explicitly.
cd(fileparts(fileparts(mfilename('fullpath'))));
DATA_ROOT  = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';
SPLIT_FILE = 'data/splits/split_v1.json';
REF_DIR    = 'matlab/reference';
TOL        = 1e-5;

%% ---- 1. Toolbox ----
fprintf('MATLAB %s\n', version);
assert(license('test', 'image_toolbox') == 1, 'Image Processing Toolbox not licensed');
fprintf('[OK] Image Processing Toolbox licensed\n');

%% ---- 2. Parity ----
% Same loop structure as tests/test_degradation.py::test_matlab_python_parity,
% so result arrays line up index-for-index with the stored reference.
addpath(REF_DIR);
d   = load(fullfile(REF_DIR, 'parity_test_data.mat'));
ref = load(fullfile(REF_DIR, 'parity_matlab_results.mat'));

snrs = [30 20 12 8];  rs = [1.0 0.75 0.5];  d0s = [0.10 0.15 0.20 0.30];
nS = size(d.slices, 1);
max_deg = 0; max_low = 0; max_high = 0; max_recon = 0;
for i = 1:nS
    x  = squeeze(d.slices(i,:,:));
    nr = squeeze(d.n_real(i,:,:));
    ni = squeeze(d.n_imag(i,:,:));
    for s = 1:4
        for r = 1:3
            y = degrade_slice(x, d.mu_brains(i), snrs(s), rs(r), nr, ni);
            max_deg = max(max_deg, max(abs(y - squeeze(ref.deg_results(i,s,r,:,:))), [], 'all'));
        end
    end
    for k = 1:4
        [lo, hi] = decompose_slice(x, d0s(k));
        max_low   = max(max_low,   max(abs(lo - squeeze(ref.low_results(i,k,:,:))),  [], 'all'));
        max_high  = max(max_high,  max(abs(hi - squeeze(ref.high_results(i,k,:,:))), [], 'all'));
        % Gaussian LP + (1 - LP) HP are complementary, so low + high must equal x.
        max_recon = max(max_recon, max(abs(lo + hi - x), [], 'all'));
    end
end
fprintf('Parity max |diff|: degrade %.2e, low %.2e, high %.2e, low+high-x %.2e\n', ...
    max_deg, max_low, max_high, max_recon);
assert(all([max_deg max_low max_high max_recon] < TOL), 'Parity FAILED (tol %g)', TOL);
fprintf('[OK] Parity within %g\n', TOL);

%% ---- 3. Data ----
split = jsondecode(fileread(SPLIT_FILE));
ids = split.test;
fprintf('Test IDs in split: %d\n', numel(ids));
mods = {'flair', 't1', 't1ce', 't2', 'seg'};
missing = {};
for i = 1:numel(ids)
    for m = 1:numel(mods)
        f = fullfile(DATA_ROOT, ids{i}, sprintf('%s_%s.nii', ids{i}, mods{m}));
        if ~isfile(f), missing{end+1} = f; end %#ok<AGROW>
    end
end
if ~isempty(missing), error('Missing %d files, first: %s', numel(missing), missing{1}); end
fprintf('[OK] All %d test patients have all 5 files\n', numel(ids));

% Header check on every test seg: spacing and size, so later scripts can
% trust (or know not to trust) the 1 mm isotropic assumption.
sp = zeros(numel(ids), 3); sz = zeros(numel(ids), 3);
for i = 1:numel(ids)
    info = niftiinfo(fullfile(DATA_ROOT, ids{i}, [ids{i} '_seg.nii']));
    sp(i,:) = info.PixelDimensions(1:3);
    sz(i,:) = info.ImageSize(1:3);
end
fprintf('Voxel spacing (mm): unique rows =\n'); disp(unique(sp, 'rows'));
fprintf('Volume size: unique rows =\n');        disp(unique(sz, 'rows'));
labs = unique(niftiread(fullfile(DATA_ROOT, ids{1}, [ids{1} '_seg.nii'])))';
fprintf('Labels in %s seg: %s\n', ids{1}, mat2str(labs));
fprintf('ENV CHECK PASSED\n');
