%% MATLAB oracle generator for native 2D first-order statistics
% MATLAB version: R2026a, Statistics and Machine Learning Toolbox 26.1.
%
% Source fixture and pinned goldens:
%   https://github.com/PolusAI/nyxus/blob/main/tests/test_data.h
%   https://github.com/PolusAI/nyxus/blob/main/tests/test_2d_firstorder_matlab.h
%
% Every reported value uses MATLAB built-ins. Derived statistics apply only
% their defining normalization to those built-in results.
%
% Run:
%   matlab -batch "run('tests/vetting/oracles/gen_firstorder2d_matlab.m')"

repo_main = 'https://github.com/PolusAI/nyxus/blob/main/';
fixture_text = webread([repo_main 'tests/test_data.h?raw=1']);
header_text = webread([repo_main 'tests/test_2d_firstorder_matlab.h?raw=1']);
intensities = load_intensities(fixture_text);

sample_mean = mean(intensities);
sample_std = std(intensities, 0);
p25 = prctile(intensities, 25, Method="midpoint");
p75 = prctile(intensities, 75, Method="midpoint");
roi_min = min(intensities);
roi_max = max(intensities);

matlab_values = containers.Map('KeyType', 'char', 'ValueType', 'double');
matlab_values('COV') = sample_std / sample_mean;
matlab_values('COVERED_IMAGE_INTENSITY_RANGE') = range(intensities) / 65535;
matlab_values('ENERGY') = dot(intensities, intensities);
matlab_values('EXCESS_KURTOSIS') = kurtosis(intensities, 1) - 3;
matlab_values('HYPERFLATNESS') = moment(intensities, 6) / sample_std^6;
matlab_values('HYPERSKEWNESS') = moment(intensities, 5) / sample_std^5;
matlab_values('INTEGRATED_INTENSITY') = sum(intensities);
matlab_values('INTERQUARTILE_RANGE') = iqr(intensities);
matlab_values('KURTOSIS') = kurtosis(intensities, 1);
matlab_values('MAX') = roi_max;
matlab_values('MEAN') = sample_mean;
matlab_values('MEAN_ABSOLUTE_DEVIATION') = mad(intensities, 0);
matlab_values('MEDIAN') = median(intensities);
matlab_values('MIN') = roi_min;
matlab_values('MODE') = mode(intensities);
matlab_values('P01') = prctile(intensities, 1, Method="midpoint");
matlab_values('P10') = prctile(intensities, 10, Method="midpoint");
matlab_values('P25') = p25;
matlab_values('P75') = p75;
matlab_values('P90') = prctile(intensities, 90, Method="midpoint");
matlab_values('P99') = prctile(intensities, 99, Method="midpoint");
matlab_values('QCOD') = (p75 - p25) / (p75 + p25);
matlab_values('RANGE') = range(intensities);
matlab_values('ROOT_MEAN_SQUARED') = rms(intensities);
matlab_values('SKEWNESS') = skewness(intensities, 1);
matlab_values('STANDARD_DEVIATION') = sample_std;
matlab_values('STANDARD_DEVIATION_BIASED') = std(intensities, 1);
matlab_values('STANDARD_ERROR') = sample_std / sqrt(numel(intensities));
matlab_values('UNIFORMITY_PIU') = ...
    (1 - (roi_max - roi_min) / (roi_max + roi_min)) * 100;
matlab_values('VARIANCE') = var(intensities, 0);
matlab_values('VARIANCE_BIASED') = var(intensities, 1);

matlab_names = sort(keys(matlab_values));
fprintf('MATLAB R%s: native 2D first-order values\n', version('-release'));
fprintf('%-32s %22s\n', 'feature', 'MATLAB');
for name_index = 1:numel(matlab_names)
    name = matlab_names{name_index};
    fprintf('%-32s %22.17g\n', name, matlab_values(name));
end

pinned_values = parse_pins(header_text, 'firstorder_2d_matlab_ref_vals');
pinned_names = sort(keys(pinned_values));
if ~isequal(matlab_names, pinned_names)
    missing_from_matlab = setdiff(pinned_names, matlab_names);
    missing_from_header = setdiff(matlab_names, pinned_names);
    error('MATLAB/header feature mismatch. Missing from MATLAB: %s. Missing from header: %s.', ...
        strjoin(missing_from_matlab, ', '), strjoin(missing_from_header, ', '));
end

percentile_names = {'INTERQUARTILE_RANGE', 'P01', 'P10', 'P25', 'P75', 'P90', 'P99', 'QCOD'};
failures = 0;
fprintf('\nGenerated values vs pinned C++ oracle goldens\n');
fprintf('%-32s %22s %22s %12s %10s\n', ...
    'feature', 'MATLAB', 'C++ golden', 'relative error', 'tolerance');
for name_index = 1:numel(matlab_names)
    name = matlab_names{name_index};
    matlab_value = matlab_values(name);
    pinned_value = pinned_values(name);
    relative_error = abs(matlab_value - pinned_value) / max(abs(pinned_value), 1e-12);
    if ismember(name, percentile_names)
        relative_tolerance = 3e-2;
    else
        relative_tolerance = 1e-3;
    end
    fprintf('%-32s %22.17g %22.17g %12.3e %10.1e\n', ...
        name, matlab_value, pinned_value, relative_error, relative_tolerance);
    failures = failures + (relative_error > relative_tolerance);
end

fprintf('\nPaste-ready MATLAB R2026a goldens\n');
for name_index = 1:numel(matlab_names)
    name = matlab_names{name_index};
    fprintf('\t{ "%s", %.17g },\n', name, matlab_values(name));
end

if failures > 0
    error('%d of %d MATLAB goldens exceed their declared tolerance.', ...
        failures, numel(matlab_names));
end
fprintf('\nAll %d MATLAB goldens agree within their declared tolerance.\n', ...
    numel(matlab_names));

function intensities = load_intensities(text)
    marker = 'pixelIntensityFeaturesTestData[] = {';
    start_index = strfind(text, marker);
    if isempty(start_index)
        error('pixelIntensityFeaturesTestData not found in tests/test_data.h.');
    end
    remainder = text(start_index(1) + length(marker):end);
    end_index = strfind(remainder, '};');
    body = remainder(1:end_index(1) - 1);
    entries = regexp(body, '\{\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*\}', 'tokens');
    intensities = zeros(numel(entries), 1);
    for entry_index = 1:numel(entries)
        intensities(entry_index) = str2double(entries{entry_index}{1});
    end
end

function pins = parse_pins(text, table_name)
    marker = [table_name newline '{'];
    start_index = strfind(text, marker);
    if isempty(start_index)
        marker = [table_name ' {'];
        start_index = strfind(text, marker);
    end
    if isempty(start_index)
        error('Table %s not found in tests/test_2d_firstorder_matlab.h.', table_name);
    end
    remainder = text(start_index(1) + length(marker):end);
    end_index = strfind(remainder, '};');
    body = remainder(1:end_index(1) - 1);
    body = regexprep(body, '//[^\n]*', '');
    entries = regexp(body, ...
        '\{\s*"([A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', 'tokens');
    pins = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for entry_index = 1:numel(entries)
        pins(entries{entry_index}{1}) = str2double(entries{entry_index}{2});
    end
end
