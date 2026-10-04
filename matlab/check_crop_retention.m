% CHECK_CROP_RETENTION  Does the Python 192x192 in-plane crop cut off tumor?
%
% Run from repo root:  matlab -batch "run('matlab/check_crop_retention.m')"
%
% python/data/preprocess.py crops rows 24:216, cols 28:220 (0-based, end
% exclusive) = MATLAB rows 25:216, cols 29:220. The models only ever see the
% crop, and evaluate.py pads predictions back with zeros, so any tumor voxel
% outside the crop is a guaranteed miss. preprocess.py checks brain-voxel
% retention but not tumor retention; this script checks tumor, all 369 patients.

cd(fileparts(fileparts(mfilename('fullpath'))));
DATA_ROOT = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';
ROWS = 25:216;  COLS = 29:220;

d = dir(fullfile(DATA_ROOT, 'BraTS20_Training_*'));
ids = {d([d.isdir]).name}';
lost = zeros(numel(ids), 1);  total = zeros(numel(ids), 1);
for i = 1:numel(ids)
    % Patient 355's seg is misnamed in the Kaggle copy (W39_1998.09.19_Segm.nii).
    f = dir(fullfile(DATA_ROOT, ids{i}, '*eg*.nii'));
    seg = niftiread(fullfile(f(1).folder, f(1).name)) > 0;
    inside = false(size(seg));  inside(ROWS, COLS, :) = true;
    total(i) = nnz(seg);
    lost(i) = nnz(seg & ~inside);
end
T = table(ids, total, lost, lost ./ total, 'VariableNames', ...
    {'patient_id', 'tumor_voxels', 'voxels_outside_crop', 'fraction_lost'});
writetable(T, 'results/crop_tumor_retention.csv');
fprintf('Patients with any tumor outside crop: %d / %d\n', nnz(lost), numel(ids));
fprintf('Total tumor voxels lost: %d of %d (%.5f%%)\n', sum(lost), sum(total), 100 * sum(lost) / sum(total));
if any(lost), disp(T(lost > 0, :)); end
