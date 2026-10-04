% TEST_AGREEMENT_STATS  Known-answer tests for agreement_stats.m.
%
% Run from repo root:  matlab -batch "run('matlab/test_agreement_stats.m')"

cd(fileparts(mfilename('fullpath')));

% 1. Shrout & Fleiss (1979), Psychol Bull 86:420, Table 2: 6 targets x 4
%    judges. Published: ICC(2,1) = 0.29, ICC(3,1) = 0.71. Our function takes
%    two raters, so call the ICC core through a 4-judge matrix via a local copy
%    of the same formula (identical code path, k = 4).
Y = [9 2 5 8; 6 1 3 2; 8 4 6 8; 7 1 2 6; 10 5 6 9; 6 2 4 7];
[n, k] = size(Y);  g = mean(Y(:));
msr = k * sum((mean(Y, 2) - g).^2) / (n - 1);
msc = n * sum((mean(Y, 1) - g).^2) / (k - 1);
mse = (sum((Y(:) - g).^2) - msr * (n - 1) - msc * (k - 1)) / ((n - 1) * (k - 1));
a1 = (msr - mse) / (msr + (k - 1) * mse + k / n * (msc - mse));
c1 = (msr - mse) / (msr + (k - 1) * mse);
fprintf('Shrout-Fleiss: ICC(2,1)=%.3f (pub 0.29), ICC(3,1)=%.3f (pub 0.71)\n', a1, c1);
assert(abs(a1 - 0.29) < 0.005 && abs(c1 - 0.71) < 0.005, 'Shrout-Fleiss values');
% Same data, judges 1 and 2 only, through the real function vs the formula.
S = agreement_stats(Y(:, 1), Y(:, 2), 10);
Y2 = Y(:, 1:2);  k = 2;  g = mean(Y2(:));
msr = k * sum((mean(Y2, 2) - g).^2) / (n - 1);
msc = n * sum((mean(Y2, 1) - g).^2) / (k - 1);
mse = (sum((Y2(:) - g).^2) - msr * (n - 1) - msc * (k - 1)) / ((n - 1) * (k - 1));
assert(abs(S.icc_a1 - (msr - mse) / (msr + mse + 2 / n * (msc - mse))) < 1e-12, 'function = formula');

% 2. Perfect agreement: bias 0, ICC 1.
rng(0);  x = 10 + 50 * rand(74, 1);
S = agreement_stats(x, x, 200);
assert(S.bias == 0 && abs(S.icc_a1 - 1) < 1e-12 && S.loa_low == 0, 'perfect');

% 3. Constant offset +5: bias exactly 5, SD 0, consistency ICC 1, absolute ICC < 1.
S = agreement_stats(x, x + 5, 200);
fprintf('Offset +5: bias=%.3f, ICC(A,1)=%.3f, ICC(C,1)=%.3f\n', S.bias, S.icc_a1, S.icc_c1);
assert(abs(S.bias - 5) < 1e-12 && S.sd_diff < 1e-12, 'offset bias');
assert(abs(S.icc_c1 - 1) < 1e-12 && S.icc_a1 < 1, 'offset ICC');

% 4. Proportional +10%: percent error exactly 10 for every patient.
S = agreement_stats(x, 1.1 * x, 200);
assert(abs(S.pct_err_median - 10) < 1e-9 && abs(S.pct_abs_err_median - 10) < 1e-9, 'pct');

% 5. NaN pairs dropped; ref = 0 excluded from percent error only.
S = agreement_stats([0; 1; 2; NaN], [1; 1; 2; 3], 50);
assert(S.n == 3 && abs(S.bias - 1/3) < 1e-12 && S.pct_err_median == 0, 'NaN / zero');

% 6. Bootstrap CI brackets the estimate.
S = agreement_stats(x, x + randn(74, 1), 500);
assert(S.bias_ci(1) <= S.bias && S.bias <= S.bias_ci(2), 'bias CI');
assert(S.icc_a1_ci(1) <= S.icc_a1 && S.icc_a1 <= S.icc_a1_ci(2), 'ICC CI');

fprintf('ALL AGREEMENT_STATS TESTS PASSED\n');
