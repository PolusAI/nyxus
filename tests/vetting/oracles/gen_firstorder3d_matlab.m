% MATLAB-semantics reference for the 3D first-order features, run as GNU Octave (SPEC section 4:
% the `matlab` token names the semantics, Octave supplies them license-free).
%
% Input:  voxels.csv  - the label-57 ROI voxel vector in the LOADER DOMAIN, written by
%                       gen_firstorder3d_matlab.py (see that file for how the domain is defined).
% Output: KEY=VALUE lines on stdout, consumed by the same driver.
%
% Every statistic below is an Octave/MATLAB built-in or a one-line closed form. Nothing here
% re-implements a Nyxus code path: where Nyxus uses its own binned-histogram estimator the built-in
% order statistic is emitted instead, and the driver reports the divergence rather than hiding it.

pkg load statistics;

args = argv();
voxfile = args{1};

x = dlmread(voxfile);
x = x(:);
n = numel(x);

mu    = mean(x);
med   = median(x);
mn    = min(x);
mx    = max(x);
sd_u  = std(x, 0);      % unbiased, N-1  -> Nyxus 3STANDARD_DEVIATION
sd_b  = std(x, 1);      % biased,   N    -> Nyxus 3STANDARD_DEVIATION_BIASED

printf('N=%d\n', n);

% ---- exact statistics: Nyxus and MATLAB agree on the definition ----
printf('3MIN=%.17g\n',                   mn);
printf('3MAX=%.17g\n',                   mx);
printf('3RANGE=%.17g\n',                 mx - mn);
printf('3MEAN=%.17g\n',                  mu);
printf('3MEDIAN=%.17g\n',                med);
printf('3MODE=%.17g\n',                  mode(x));
printf('3INTEGRATED_INTENSITY=%.17g\n',  sum(x));
printf('3ENERGY=%.17g\n',                sum(x .^ 2));
printf('3ROOT_MEAN_SQUARED=%.17g\n',     sqrt(mean(x .^ 2)));
printf('3VARIANCE=%.17g\n',              var(x, 0));
printf('3VARIANCE_BIASED=%.17g\n',       var(x, 1));
printf('3STANDARD_DEVIATION=%.17g\n',    sd_u);
printf('3STANDARD_DEVIATION_BIASED=%.17g\n', sd_b);
printf('3STANDARD_ERROR=%.17g\n',        sd_u / sqrt(n));
printf('3COV=%.17g\n',                   sd_u / mu);
printf('3MEAN_ABSOLUTE_DEVIATION=%.17g\n', mean(abs(x - mu)));
printf('3UNIFORMITY_PIU=%.17g\n',        (1 - (mx - mn) / (mx + mn)) * 100);

% Nyxus 3MEDIAN_ABSOLUTE_DEVIATION is mean|x - median|, not MATLAB mad(x,1) = median|x - median|.
% Both are emitted so the driver can show which definition the golden follows.
printf('3MEDIAN_ABSOLUTE_DEVIATION=%.17g\n', mean(abs(x - med)));
printf('MATLAB_mad_1=%.17g\n',               mad(x, 1));

% Population moments: Octave's skewness()/kurtosis() are the same estimators Moments4 computes
% (sqrt(n)*M3/M2^1.5 and n*M4/M2^2), so these are a direct comparison, not a convention match.
printf('3SKEWNESS=%.17g\n',        skewness(x));
printf('3KURTOSIS=%.17g\n',        kurtosis(x));
printf('3EXCESS_KURTOSIS=%.17g\n', kurtosis(x) - 3);

% Hyperskewness / hyperflatness are closed forms over the UNBIASED sd (3d_intensity.cpp).
printf('3HYPERSKEWNESS=%.17g\n', sum((x - mu) .^ 5) / (n * sd_u ^ 5));
printf('3HYPERFLATNESS=%.17g\n', sum((x - mu) .^ 6) / (n * sd_u ^ 6));

% ---- order statistics: MATLAB prctile, the definition Nyxus approximates ----
p01 = prctile(x,  1); p10 = prctile(x, 10); p25 = prctile(x, 25);
p75 = prctile(x, 75); p90 = prctile(x, 90); p99 = prctile(x, 99);
printf('3P01=%.17g\n', p01);
printf('3P10=%.17g\n', p10);
printf('3P25=%.17g\n', p25);
printf('3P75=%.17g\n', p75);
printf('3P90=%.17g\n', p90);
printf('3P99=%.17g\n', p99);
printf('3INTERQUARTILE_RANGE=%.17g\n', p75 - p25);
printf('3QCOD=%.17g\n', (p75 - p25) / (p75 + p25));

% Robust window [P10,P90] on the exact percentiles.
r = x(x >= p10 & x <= p90);
printf('3ROBUST_MEAN=%.17g\n', mean(r));
printf('3ROBUST_MEAN_ABSOLUTE_DEVIATION=%.17g\n', mean(abs(r - mean(r))));

% ---- histogram statistics, on the recipe the Nyxus settings define ----
% Equal-width binning of [min,max] into NB bins, top value folded into the last bin. That is a
% standard histogram, not a Nyxus-specific estimator; only the bin count is a Nyxus setting, so it
% is passed in as a recipe parameter the same way a PyRadiomics binCount would be.
NB = str2double(args{2});
g = floor((x - mn) / (mx - mn) * NB);
g(g >= NB) = NB - 1;
counts = accumarray(g + 1, 1, [NB 1]);
p = counts / n;
nz = p(p > 0);
printf('3ENTROPY=%.17g\n',    -sum(nz .* log2(nz)));
printf('3UNIFORMITY=%.17g\n',  sum(p .^ 2));

printf('OCTAVE_FO3D_DONE\n');
