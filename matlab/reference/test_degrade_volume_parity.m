% TEST_DEGRADE_VOLUME_PARITY  degrade_volume.m vs pilot.pilot.degrade_volume (Python).
%
% Run from repo root:  matlab -batch "run('matlab/reference/test_degrade_volume_parity.m')"
% Needs python with torch on the PATH (it runs export_degrade_parity.py).
%
% Both sides get the same input volume and the same N(0,1) noise arrays, so
% for each of the 13 DEC-005 conditions the outputs must agree to float32
% rounding. Pass criterion: max |difference| < 1e-4 relative to the mean
% brain intensity. Writes results/degrade_volume_parity.csv.

cd(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath('matlab/reference');
TOL = 1e-4;

tmp = tempname;  mkdir(tmp);
cleanup = onCleanup(@() rmdir(tmp, 's'));
[st, msg] = system(sprintf('python matlab/reference/export_degrade_parity.py "%s"', tmp));
assert(st == 0, 'Python export failed:\n%s', msg);

meta = strsplit(strtrim(fileread(fullfile(tmp, 'meta.txt'))), newline);
dims = sscanf(meta{1}, '%d')';                    % Z C H W (torch order)
rd = @(name, n) fread_f32(fullfile(tmp, [name '.f32']), n);
% torch (Z,C,H,W) in C order = MATLAB [W H C Z] in column order -> [H W Z C]
to_m = @(v) permute(reshape(v, dims([4 3 2 1])), [2 1 4 3]);
x  = to_m(rd('x', prod(dims)));
nr = to_m(rd('nr', prod(dims)));
ni = to_m(rd('ni', prod(dims)));
mu = rd('mu', dims(2));

ids = strings(0);  maxdiff = [];  reldiff = [];
for k = 2:numel(meta)
    p = strsplit(strtrim(meta{k}));
    cond = struct('snr', str2double(p{2}), 'r', str2double(p{3}), 'thick', str2double(p{4}));
    if cond.snr == 0, cond.snr = []; end
    yp = to_m(rd(p{1}, prod(dims)));
    ym = degrade_volume(x, mu, cond, nr, ni);
    d = max(abs(double(ym(:)) - double(yp(:))));
    ids(end + 1) = p{1};  maxdiff(end + 1) = d;  reldiff(end + 1) = d / mean(mu); %#ok<SAGROW>
    fprintf('%-24s max |diff| %.3g  (%.2g of mean brain intensity)\n', p{1}, d, d / mean(mu));
end
T = table(ids', maxdiff', reldiff', 'VariableNames', {'condition', 'max_abs_diff', 'rel_to_mu'});
writetable(T, 'results/degrade_volume_parity.csv');
assert(all(reldiff < TOL), 'degrade_volume parity FAILED');
fprintf('DEGRADE_VOLUME PARITY PASSED (13 conditions, worst %.2g of mean brain intensity)\n', max(reldiff));

function v = fread_f32(f, n)
fid = fopen(f, 'r', 'ieee-le');  v = fread(fid, n, 'single=>single');  fclose(fid);
end
