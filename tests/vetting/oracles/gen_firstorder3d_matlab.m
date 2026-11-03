% MATLAB reference generator for native 3D first-order statistics.
% MATLAB version: R2026a, Statistics and Machine Learning Toolbox 26.1.
%
% Source fixture and pinned goldens (read from the checkout this runs in):
%   tests/data/nifti/phantoms/ut_inten.nii, ut_mask57.nii   label 57
%   tests/test_3d_firstorder_matlab.h                        firstorder_3d_matlab_ref_vals
%
% Every reported value uses MATLAB built-ins; derived statistics apply only their defining
% normalization to those built-in results.
%
% The percentiles take prctile(x, p, Method="inclusive"), numpy's default 'linear' rule: ranks
% (k-1)/(n-1).
%
% Run from the repository root:
%   matlab -batch "run('tests/vetting/oracles/gen_firstorder3d_matlab.m')"

% run() changes into this script's folder, so the checkout is found from here
repo_root = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
phantoms = fullfile(repo_root, 'tests', 'data', 'nifti', 'phantoms');
intensity = niftiread(fullfile(phantoms, 'ut_inten.nii'));
mask = niftiread(fullfile(phantoms, 'ut_mask57.nii'));
header_text = fileread(fullfile(repo_root, 'tests', 'test_3d_firstorder_matlab.h'));

% Match what Nyxus reports: the loader offsets a volume holding negative values by its floored
% minimum and truncates to integer grey levels, and the intensity family adds that offset back,
% so the reported domain is the volume's own with the truncation kept. Fixture setup, not a
% feature formula.
intensity = double(intensity);
offset = floor(min(intensity, [], 'all'));
if offset < 0
    intensity = fix(intensity - offset) + offset;
else
    intensity = fix(intensity);
end
voxels = intensity(mask == 57);

sample_mean = mean(voxels);
sample_std = std(voxels, 0);
p25 = prctile(voxels, 25, Method="inclusive");
p75 = prctile(voxels, 75, Method="inclusive");
voxel_min = min(voxels);
voxel_max = max(voxels);

matlab_values = containers.Map('KeyType', 'char', 'ValueType', 'double');
matlab_values('3COV') = sample_std / sample_mean;
matlab_values('3EXCESS_KURTOSIS') = kurtosis(voxels, 1) - 3;
matlab_values('3HYPERFLATNESS') = moment(voxels, 6) / sample_std^6;
matlab_values('3HYPERSKEWNESS') = moment(voxels, 5) / sample_std^5;
matlab_values('3INTEGRATED_INTENSITY') = sum(voxels);
matlab_values('3INTERQUARTILE_RANGE') = p75 - p25;
matlab_values('3KURTOSIS') = kurtosis(voxels, 1);
matlab_values('3MAX') = voxel_max;
matlab_values('3MEAN') = sample_mean;
matlab_values('3MEAN_ABSOLUTE_DEVIATION') = mad(voxels, 0);
matlab_values('3MEDIAN') = median(voxels);
matlab_values('3MIN') = voxel_min;
matlab_values('3MODE') = mode(voxels);
matlab_values('3P01') = prctile(voxels, 1, Method="inclusive");
matlab_values('3P10') = prctile(voxels, 10, Method="inclusive");
matlab_values('3P25') = p25;
matlab_values('3P75') = p75;
matlab_values('3P90') = prctile(voxels, 90, Method="inclusive");
matlab_values('3P99') = prctile(voxels, 99, Method="inclusive");
matlab_values('3QCOD') = (p75 - p25) / (p75 + p25);
matlab_values('3RANGE') = range(voxels);
matlab_values('3ROOT_MEAN_SQUARED') = rms(voxels);
matlab_values('3SKEWNESS') = skewness(voxels, 1);
matlab_values('3STANDARD_DEVIATION') = sample_std;
matlab_values('3STANDARD_DEVIATION_BIASED') = std(voxels, 1);
matlab_values('3STANDARD_ERROR') = sample_std / sqrt(numel(voxels));
matlab_values('3UNIFORMITY_PIU') = ...
    (1 - (voxel_max - voxel_min) / (voxel_max + voxel_min)) * 100;
matlab_values('3VARIANCE') = var(voxels, 0);
matlab_values('3VARIANCE_BIASED') = var(voxels, 1);

matlab_names = sort(keys(matlab_values));
fprintf('MATLAB R%s: native 3D first-order values of %d voxels\n', ...
    version('-release'), numel(voxels));

pinned_values = parse_pins(header_text, 'firstorder_3d_matlab_ref_vals');
pinned_names = sort(keys(pinned_values));
if ~isequal(matlab_names, pinned_names)
    missing_from_matlab = setdiff(pinned_names, matlab_names);
    missing_from_header = setdiff(matlab_names, pinned_names);
    error('MATLAB/header feature mismatch. Missing from MATLAB: %s. Missing from header: %s.', ...
        strjoin(missing_from_matlab, ', '), strjoin(missing_from_header, ', '));
end

percentile_names = {'3INTERQUARTILE_RANGE', '3P01', '3P10', '3P25', '3P75', '3P90', '3P99', ...
    '3QCOD'};
% the voxel range a zero golden's residual is measured against (firstorder_3d_matlab_span)
span = pinned_values('3RANGE');
failures = 0;
fprintf('\nGenerated values vs pinned C++ oracle goldens\n');
fprintf('%-32s %22s %22s %12s %10s\n', ...
    'feature', 'MATLAB', 'C++ golden', 'residual', 'tolerance');
for name_index = 1:numel(matlab_names)
    name = matlab_names{name_index};
    matlab_value = matlab_values(name);
    pinned_value = pinned_values(name);
    % the bands of assert_3d_firstorder_value_matlab() in test_3d_firstorder_matlab.h
    if pinned_value == 0
        residual = abs(matlab_value) / span;
        relative_tolerance = 1e-3;
    else
        residual = abs(matlab_value - pinned_value) / abs(pinned_value);
        if ismember(name, percentile_names)
            relative_tolerance = 1e-9;
        else
            relative_tolerance = 1e-3;
        end
    end
    fprintf('%-32s %22.17g %22.17g %12.3e %10.1e\n', ...
        name, matlab_value, pinned_value, residual, relative_tolerance);
    failures = failures + (residual > relative_tolerance);
end

fprintf('\nPaste-ready MATLAB R2026a goldens\n');
for name_index = 1:numel(matlab_names)
    name = matlab_names{name_index};
    fprintf('    { "%s", %.17g },\n', name, matlab_values(name));
end

if failures > 0
    error('%d of %d MATLAB goldens exceed their declared tolerance.', ...
        failures, numel(matlab_names));
end
fprintf('\nAll %d MATLAB goldens agree within their declared tolerance.\n', ...
    numel(matlab_names));

function pins = parse_pins(text, table_name)
    start_index = regexp(text, [table_name '\s*\{'], 'end', 'once');
    if isempty(start_index)
        error('Table %s not found.', table_name);
    end
    remainder = text(start_index + 1:end);
    end_index = strfind(remainder, '};');
    body = regexprep(remainder(1:end_index(1) - 1), '//[^\n]*', '');
    entries = regexp(body, '\{\s*"([A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', 'tokens');
    pins = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for entry_index = 1:numel(entries)
        pins(entries{entry_index}{1}) = str2double(entries{entry_index}{2});
    end
end
