%% GNU Octave surrogate generator for the percentile rows of the 2D first-order MATLAB table
% Octave version: 11.3.0 (core quantile; no package needed).
%
% Source fixture and pinned goldens (read from the checkout this runs in):
%   tests/test_data.h                   pixelIntensityFeaturesTestData
%   tests/test_2d_firstorder_matlab.h   firstorder_2d_matlab_ref_vals
%
% Octave's prctile takes no Method argument. The rule MATLAB's prctile(x, p, Method="inclusive")
% takes, numpy's default 'linear' ranks (k-1)/(n-1), is quantile(x, p/100, 1, 7) here. The
% interquartile range is P75 - P25 and QCOD is (P75 - P25) / (P75 + P25). The table's other
% rows are pinned from licensed MATLAB by gen_firstorder2d_matlab.m.
%
% Run from the repository root:
%   octave-cli tests/vetting/oracles/gen_firstorder2d_percentiles_octave.m

1;

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

intensities = load_intensities(fileread('tests/test_data.h'));
pinned = parse_pins(fileread('tests/test_2d_firstorder_matlab.h'), 'firstorder_2d_matlab_ref_vals');

percents = [1 10 25 75 90 99];
q = quantile(intensities, percents / 100, 1, 7);
values = containers.Map('KeyType', 'char', 'ValueType', 'double');
for k = 1:numel(percents)
    values(sprintf('P%02d', percents(k))) = q(k);
end
values('INTERQUARTILE_RANGE') = values('P75') - values('P25');
values('QCOD') = (values('P75') - values('P25')) / (values('P75') + values('P25'));

% the band of firstorder_2d_matlab_rel_tol() in test_2d_firstorder_matlab.h for these rows
relative_tolerance = 1e-9;
names = sort(keys(values));
failures = 0;
printf('GNU Octave %s: 2D first-order percentiles of %d pixels\n', OCTAVE_VERSION, numel(intensities));
printf('%-22s %24s %24s %12s\n', 'feature', 'Octave', 'C++ golden', 'rel. error');
for k = 1:numel(names)
    name = names{k};
    if ~isKey(pinned, name)
        error('%s is not pinned in firstorder_2d_matlab_ref_vals.', name);
    end
    relative_error = abs(values(name) - pinned(name)) / max(abs(pinned(name)), 1e-12);
    printf('%-22s %24.17g %24.17g %12.3e\n', name, values(name), pinned(name), relative_error);
    failures = failures + (relative_error > relative_tolerance);
end
if failures > 0
    error('%d of %d percentile goldens exceed rel=%g.', failures, numel(names), relative_tolerance);
end
printf('\nAll %d percentile goldens agree within rel=%g.\n', numel(names), relative_tolerance);
