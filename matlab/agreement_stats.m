function S = agreement_stats(ref, test, nboot)
% AGREEMENT_STATS  How well a measurement (test) agrees with the truth (ref).
%
%   S = agreement_stats(ref, test)          e.g. GT volume vs predicted volume
%   S = agreement_stats(ref, test, nboot)   bootstrap resamples for CIs (default 2000)
%
%   ref, test - vectors, one value per patient. Pairs with a NaN are dropped.
%
%   S fields:
%     n                    patients used
%     bias                 mean(test - ref)       Bland-Altman bias
%     sd_diff              std(test - ref)
%     loa_low, loa_high    bias -/+ 1.96 sd       Bland-Altman 95% limits of agreement
%     bias_ci              bootstrap 95% CI of the bias
%     icc_a1               ICC(A,1): two-way, ABSOLUTE agreement, single measure
%                          (McGraw & Wong 1996; = Shrout & Fleiss ICC(2,1)).
%                          Penalises systematic offsets. Primary agreement index.
%     icc_a1_ci            bootstrap 95% CI of icc_a1 (percentile)
%     icc_c1               ICC(C,1): CONSISTENCY (= Shrout & Fleiss ICC(3,1)).
%                          Ignores a constant offset; high C1 with low A1 means
%                          "right ranking, wrong level".
%     pct_err_mean, pct_err_median, pct_abs_err_median
%                          100 (test - ref) / ref, over patients with ref > 0
%
%   Implemented by hand: the Statistics and Machine Learning Toolbox is not
%   installed. Verified against Shrout & Fleiss (1979) Table 2 in
%   test_agreement_stats.m.

if nargin < 3, nboot = 2000; end
ref = double(ref(:));  test = double(test(:));
ok = ~isnan(ref) & ~isnan(test);
ref = ref(ok);  test = test(ok);
d = test - ref;

S.n = numel(d);
S.bias = mean(d);
S.sd_diff = std(d);
S.loa_low = S.bias - 1.96 * S.sd_diff;
S.loa_high = S.bias + 1.96 * S.sd_diff;
[S.icc_a1, S.icc_c1] = icc([ref test]);

pos = ref > 0;
pe = 100 * d(pos) ./ ref(pos);
S.pct_err_mean = mean(pe);
S.pct_err_median = median(pe);
S.pct_abs_err_median = median(abs(pe));

% Bootstrap over patients (resample patients with replacement), fixed seed.
s = RandStream('mt19937ar', 'Seed', 42);
bb = zeros(nboot, 1);  ba = zeros(nboot, 1);
for b = 1:nboot
    idx = randi(s, S.n, S.n, 1);
    bb(b) = mean(d(idx));
    ba(b) = icc([ref(idx) test(idx)]);
end
S.bias_ci = prctile_(bb, [2.5 97.5]);
S.icc_a1_ci = prctile_(ba, [2.5 97.5]);
end

function [a1, c1] = icc(Y)
% Two-way ANOVA mean squares on an n-subjects x k-raters matrix.
[n, k] = size(Y);
g = mean(Y(:));
msr = k * sum((mean(Y, 2) - g).^2) / (n - 1);          % between subjects
msc = n * sum((mean(Y, 1) - g).^2) / (k - 1);          % between raters
sse = sum((Y(:) - g).^2) - msr * (n - 1) - msc * (k - 1);
mse = sse / ((n - 1) * (k - 1));                       % residual
a1 = (msr - mse) / (msr + (k - 1) * mse + k / n * (msc - mse));
c1 = (msr - mse) / (msr + (k - 1) * mse);
end

function q = prctile_(x, p)
% Percentiles by linear interpolation (prctile is in the Statistics Toolbox).
x = sort(x(~isnan(x)));
q = interp1(linspace(0, 100, numel(x)), x, p);
end
