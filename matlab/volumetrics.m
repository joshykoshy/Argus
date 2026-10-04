% VOLUMETRICS  Tumor shape biomarkers for the 74 test patients.
% ECTE408 Low-Field MRI Brain Tumor Segmentation Study
%
% Run from repo root:  matlab -batch "run('matlab/volumetrics.m')"
%
% SOURCE = 'gt'   -> ground truth from the raw BraTS seg files (now).
% SOURCE = 'pred' -> model predictions in results/predictions/ (Stage 3, only
%                    once evaluation has written ALL files; the script refuses
%                    to run on a partially filled folder so models never mix).
%
% Every mask, truth or prediction, goes through tumor_biomarkers.m, so the
% measurement method is identical by construction. Output is one row per
% (source, condition, patient, region); see tumor_biomarkers.m for columns.

%% ---- Configuration (paths relative to repo root) ----
cd(fileparts(fileparts(mfilename('fullpath'))));
SOURCE     = 'gt';
DATA_ROOT  = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';
SPLIT_FILE = 'data/splits/split_v1.json';
PRED_DIR   = 'results/predictions';
MODELS     = {'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7'};
% python/evaluate.py saves NIfTI predictions for these 3 conditions only.
CONDITIONS = {'clean', 'snr12_r0.75', 'snr8_r0.5'};
OUT_CSV    = sprintf('results/volumetrics_%s.csv', SOURCE);

addpath('matlab');

%% ---- Patients and grades ----
split = jsondecode(fileread(SPLIT_FILE));
ids = split.test;
map = readtable(fullfile(DATA_ROOT, 'name_mapping.csv'), 'TextType', 'string');
[~, loc] = ismember(ids, map.BraTS_2020_subject_ID);
assert(all(loc > 0), 'Some test IDs missing from name_mapping.csv');
grades = cellstr(map.Grade(loc));

%% ---- Build the job list: {source, condition, patient index, file} ----
switch SOURCE
    case 'gt'
        jobs = cell(numel(ids), 4);
        for p = 1:numel(ids)
            jobs(p, :) = {'GroundTruth', 'clean', p, ...
                fullfile(DATA_ROOT, ids{p}, [ids{p} '_seg.nii'])};
        end
    case 'pred'
        jobs = cell(0, 4);
        for m = 1:numel(MODELS)
            for c = 1:numel(CONDITIONS)
                for p = 1:numel(ids)
                    jobs(end+1, :) = {MODELS{m}, CONDITIONS{c}, p, fullfile(PRED_DIR, ...
                        sprintf('%s_%s_%s.nii.gz', MODELS{m}, ids{p}, CONDITIONS{c}))}; %#ok<SAGROW>
                end
            end
        end
    otherwise
        error('SOURCE must be ''gt'' or ''pred''');
end
missing = jobs(~cellfun(@isfile, jobs(:, 4)), 4);
if ~isempty(missing)
    error('%d of %d input files missing (first: %s). Not running on a partial set.', ...
        numel(missing), size(jobs, 1), missing{1});
end

%% ---- Measure ----
fprintf('Measuring %d volumes (%s)...\n', size(jobs, 1), SOURCE);
tables = cell(size(jobs, 1), 1);
tic;
for j = 1:size(jobs, 1)
    info = niftiinfo(jobs{j, 4});            % spacing from the header, not assumed
    T = tumor_biomarkers(niftiread(info), info.PixelDimensions(1:3));
    p = jobs{j, 3};
    n = height(T);
    T = [table(repmat(jobs(j, 1), n, 1), repmat(jobs(j, 2), n, 1), ...
               repmat(ids(p), n, 1), repmat(grades(p), n, 1), ...
               'VariableNames', {'source', 'condition', 'patient_id', 'grade'}), T]; %#ok<AGROW>
    tables{j} = T;
    if mod(j, 10) == 0 || j == size(jobs, 1)
        fprintf('  %d/%d  (%.0f s)\n', j, size(jobs, 1), toc);
    end
end
out = vertcat(tables{:});
writetable(out, OUT_CSV);
fprintf('Saved %d rows to %s\n', height(out), OUT_CSV);

%% ---- Quick summary to sanity-check by eye ----
for reg = {'WT', 'TC', 'ET'}
    r = out(strcmp(out.region, reg{1}), :);
    fprintf(['%s: median volume %.1f cm3 [%.1f-%.1f], empty in %d patients, ', ...
             'median sphericity %.3f (mesh %.3f), median pieces %d, max %d\n'], ...
        reg{1}, median(r.volume_cm3), min(r.volume_cm3), max(r.volume_cm3), ...
        sum(r.voxels == 0), median(r.sphericity, 'omitnan'), ...
        median(r.sphericity_mesh, 'omitnan'), median(r.n_components), max(r.n_components));
end
