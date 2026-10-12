% CHECK_PRED_GRID  Stage 3.0: do the Python-written NIfTIs sit on the raw BraTS grid?
%
% Run from repo root:  matlab -batch "run('matlab/check_pred_grid.m')"
%
% evaluation/evaluate_v2.py crops to 192x192, predicts, pads back to 240x240x155
% and writes predictions/ground_truth/GT_<id>.nii.gz with labels 1 = NCR,
% 2 = edema, 3 = ET (BraTS raw uses 4 for ET). If padding, orientation or the
% label mapping were wrong, every Stage 3 measurement would be wrong, so compare
% each GT_<id> with the raw seg voxel by voxel. Expected: identical except the
% 18 tumor voxels outside the crop (one patient, check_crop_retention.m).
%
% Output: results/pred_grid_check.csv (one row per test patient)

cd(fileparts(fileparts(mfilename('fullpath'))));
DATA_ROOT = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';
PRED_DIR  = 'results/predictions';

split = jsondecode(fileread('data/splits/split_v1.json'));
ids = split.test;
n = numel(ids);
same_size = false(n, 1); same_spacing = false(n, 1);
mismatch = zeros(n, 1); d_wt = zeros(n, 1); d_tc = zeros(n, 1); d_et = zeros(n, 1);
for i = 1:n
    f = dir(fullfile(DATA_ROOT, ids{i}, '*eg*.nii'));      % patient 355's seg is misnamed
    rawf = fullfile(f(1).folder, f(1).name);
    gtf  = fullfile(PRED_DIR, 'ground_truth', ['GT_' ids{i} '.nii.gz']);
    raw = niftiread(rawf);  raw(raw == 4) = 3;              % same label code as the GT file
    gt  = niftiread(gtf);
    ri = niftiinfo(rawf);  gi = niftiinfo(gtf);
    same_size(i)    = isequal(size(raw), size(gt));
    same_spacing(i) = max(abs(ri.PixelDimensions - gi.PixelDimensions)) < 1e-6;
    mismatch(i) = nnz(raw ~= gt);
    % Raw minus written voxel count per region (positive = lost in the pipeline)
    d_wt(i) = nnz(raw > 0) - nnz(gt > 0);
    d_tc(i) = nnz(raw == 1 | raw == 3) - nnz(gt == 1 | gt == 3);
    d_et(i) = nnz(raw == 3) - nnz(gt == 3);
end
T = table(ids, same_size, same_spacing, mismatch, d_wt, d_tc, d_et, 'VariableNames', ...
    {'patient_id', 'same_size', 'same_spacing', 'voxels_differing', 'wt_lost', 'tc_lost', 'et_lost'});
writetable(T, 'results/pred_grid_check.csv');
fprintf('Size and spacing match raw: %d / %d\n', nnz(same_size & same_spacing), n);
fprintf('Patients with any differing voxel: %d (total %d voxels)\n', nnz(mismatch), sum(mismatch));
fprintf('Tumor voxels lost: WT %d, TC %d, ET %d\n', sum(d_wt), sum(d_tc), sum(d_et));
