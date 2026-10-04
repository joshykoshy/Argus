% TEST_STAGE3_PIPELINE  End-to-end dry run of Stage 3 on FAKE predictions.
%
% Run from repo root:  matlab -batch "run('matlab/test_stage3_pipeline.m')"
% Runtime ~15-20 min. Writes only to a temporary folder (deleted at the end);
% never touches results/predictions/.
%
% Builds a fake predictions folder in the exact format python/evaluate.py
% writes (<model>_<patient>_<condition>.nii.gz, labels 1/2/3 with ET = 3),
% from synthetically damaged ground truth with KNOWN errors:
%   fake "M0":  clean = dilate 1 mm, snr12_r0.75 = dilate 2 mm,
%               snr8_r0.5 = dilate 2 mm + 3 spurious blobs
%   fake "M4":  clean = perfect,     snr12_r0.75 = erode 1 mm,
%               snr8_r0.5 = erode 1 mm
% then runs volumetrics.m (pred mode) and agreement_report.m on it and checks
% the report reproduces the Step 1.2 validation numbers. Also checks that
% volumetrics.m refuses a partially filled folder.

cd(fileparts(fileparts(mfilename('fullpath'))));
addpath('matlab');
DATA_ROOT = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';
ids = jsondecode(fileread('data/splits/split_v1.json')).test;
TMP = tempname;  mkdir(TMP);
cleanup = onCleanup(@() rmdir(TMP, 's'));
fprintf('Fake predictions in %s\n', TMP);

plan = struct('M0', {{'dilate', 1; 'dilate', 2; 'dilate+blobs', 2}}, ...
              'M4', {{'none', 0; 'erode', 1; 'erode', 1}});
conds = {'clean', 'snr12_r0.75', 'snr8_r0.5'};
tic;
for p = 1:numel(ids)
    id = ids{p};
    lab = niftiread(fullfile(DATA_ROOT, id, [id '_seg.nii']));
    brain = niftiread(fullfile(DATA_ROOT, id, [id '_flair.nii'])) > 0;
    for mname = {'M0', 'M4'}
        steps = plan.(mname{1});
        for c = 1:3
            rs = RandStream('mt19937ar', 'Seed', 42 + p);
            switch steps{c, 1}
                case 'none',   d = lab;
                case 'dilate+blobs'
                    d = damage_mask(lab, 'dilate', steps{c, 2});
                    d = damage_mask(d, 'blobs', [3 3], rs, brain);
                otherwise,     d = damage_mask(lab, steps{c, 1}, steps{c, 2});
            end
            d(d == 4) = 3;                         % Python label convention
            % Explicit .nii: otherwise niftiwrite treats the '.5' in 'r0.5' as
            % the extension and writes '..._r0.nii.gz'.
            f = fullfile(TMP, sprintf('%s_%s_%s.nii', mname{1}, id, conds{c}));
            niftiwrite(uint8(d), f, 'Compressed', true);   % -> f.gz
        end
    end
    if mod(p, 10) == 0, fprintf('  wrote %d/%d patients (%.0f s)\n', p, numel(ids), toc); end
end

% 1. Partial-folder guard: hide one file, volumetrics.m must refuse.
victim = fullfile(TMP, sprintf('M4_%s_snr8_r0.5.nii.gz', ids{end}));
movefile(victim, [victim '.bak']);
SOURCE = 'pred';  PRED_DIR = TMP;  MODELS = {'M0', 'M4'};
OUT_CSV = fullfile(TMP, 'vol_pred.csv');
try
    run('matlab/volumetrics.m');  error('volumetrics.m ran on a partial folder');
catch ME
    assert(contains(ME.message, 'Not running on a partial set'), ME.message);
    fprintf('[OK] partial folder refused: %s\n', ME.message);
end
movefile([victim '.bak'], victim);

% 2. Full run.
SOURCE = 'pred';  PRED_DIR = TMP;  MODELS = {'M0', 'M4'};
OUT_CSV = fullfile(TMP, 'vol_pred.csv');
run('matlab/volumetrics.m');
PRED_CSV = OUT_CSV;  OUT_DIR = TMP;  FIG_DIR = TMP;  TAG = 'dryrun';  COMPARE = {'M0', 'M4'};
run('matlab/agreement_report.m');
if ~isfolder('docs/dryrun'), mkdir('docs/dryrun'); end
copyfile(fullfile(TMP, 'agreement_dryrun_*.png'), 'docs/dryrun');   % FAKE data: kept out of figures/

% 3. Known answers (Step 1.2 values, results/agreement_validation.csv)
S = readtable(fullfile(TMP, 'agreement_dryrun.csv'), 'TextType', 'string');
V = readtable('results/agreement_validation.csv', 'TextType', 'string');
row = @(T, m, c, k) T(T.model == m & T.condition == c & T.region == k, :);
vrow = @(d, k) V(V.damage == d & V.region == k, :);
for k = ["WT", "TC", "ET"]
    a = row(S, "M0", "clean", k);  b = vrow("dilate 1 mm", k);
    assert(abs(a.vol_bias_cm3 - b.bias_cm3) < 1e-6, 'M0 clean = dilate 1 mm (%s)', k);
    a = row(S, "M4", "clean", k);
    assert(abs(a.vol_bias_cm3) < 1e-12 && abs(a.icc_a1 - 1) < 1e-12, 'M4 clean perfect (%s)', k);
    a = row(S, "M4", "snr8_r0.5", k);  b = vrow("erode 1 mm", k);
    assert(abs(a.vol_bias_cm3 - b.bias_cm3) < 1e-6, 'M4 snr8 = erode 1 mm (%s)', k);
end
a = row(S, "M0", "snr8_r0.5", "WT");
assert(a.extra_pieces_median >= 1, 'blobs should add pieces');
fprintf('[OK] report reproduces the Step 1.2 validation numbers exactly\n');
fprintf('STAGE 3 DRY RUN PASSED\n');
