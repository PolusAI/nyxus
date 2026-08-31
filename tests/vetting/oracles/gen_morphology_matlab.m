%% MATLAB oracle generator for 2D morphology
% MATLAB version: R2026a.
%
% Source fixture and pinned goldens:
%   https://github.com/PolusAI/nyxus/blob/main/tests/test_data.h
%   https://github.com/PolusAI/nyxus/blob/main/tests/test_2d_morphology_matlab.h
%
% Built-ins:
%   regionprops(..., 'Area', 'Centroid', 'BoundingBox', 'Extent',
%               'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'Extrema')
%   regionprops(..., intensity, 'WeightedCentroid')
%   bweuler(..., 8)
%
% Run:
%   matlab -batch "run('tests/vetting/oracles/gen_morphology_matlab.m')"

repo_main = 'https://github.com/PolusAI/nyxus/blob/main/';
fixture_text = webread([repo_main 'tests/test_data.h?raw=1']);
header_text = webread([repo_main 'tests/test_2d_morphology_matlab.h?raw=1']);

[mask_values, intensity_values] = load_shape2d(fixture_text);
mask = logical(mask_values);
pixel_size_um = 2.0;

shape = regionprops(mask, 'Area', 'Centroid', 'BoundingBox', 'Extent', ...
    'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'Extrema');
weighted = regionprops(mask, intensity_values, 'WeightedCentroid');

matlab_values = containers.Map('KeyType', 'char', 'ValueType', 'double');
matlab_values('AREA_PIXELS_COUNT') = shape.Area;
matlab_values('AREA_UM2') = shape.Area * pixel_size_um^2;
matlab_values('CENTROID_X') = shape.Centroid(1) - 1;
matlab_values('CENTROID_Y') = shape.Centroid(2) - 1;
matlab_values('WEIGHTED_CENTROID_X') = weighted.WeightedCentroid(1) - 1;
matlab_values('WEIGHTED_CENTROID_Y') = weighted.WeightedCentroid(2) - 1;
matlab_values('BBOX_XMIN') = shape.BoundingBox(1) - 0.5;
matlab_values('BBOX_YMIN') = shape.BoundingBox(2) - 0.5;
matlab_values('BBOX_WIDTH') = shape.BoundingBox(3);
matlab_values('BBOX_HEIGHT') = shape.BoundingBox(4);
matlab_values('ASPECT_RATIO') = shape.BoundingBox(3) / shape.BoundingBox(4);
matlab_values('EXTENT') = shape.Extent;
matlab_values('MAJOR_AXIS_LENGTH') = shape.MajorAxisLength;
matlab_values('MINOR_AXIS_LENGTH') = shape.MinorAxisLength;
matlab_values('ELONGATION') = shape.MinorAxisLength / shape.MajorAxisLength;
matlab_values('ECCENTRICITY') = shape.Eccentricity;
matlab_values('EULER_NUMBER') = bweuler(mask, 8);

% Extrema are 1-based sub-pixel corners; Nyxus reports 0-based pixel centres.
% The fixed regionprops order determines whether a coordinate is a left/top
% edge (-0.5) or right/bottom edge (-1.5) in the Nyxus frame.
extrema = shape.Extrema;
x_offsets = [-0.5, -1.5, -1.5, -1.5, -1.5, -0.5, -0.5, -0.5];
y_offsets = [-0.5, -0.5, -0.5, -1.5, -1.5, -1.5, -1.5, -0.5];
for point = 1:8
    matlab_values(sprintf('EXTREMA_P%d_X', point)) = extrema(point, 1) + x_offsets(point);
    matlab_values(sprintf('EXTREMA_P%d_Y', point)) = extrema(point, 2) + y_offsets(point);
end

tables = {'morphology_2d_matlab_regionprops_ref_vals', ...
    'morphology_2d_matlab_extrema_ref_vals'};
pinned_values = containers.Map('KeyType', 'char', 'ValueType', 'double');
for table_index = 1:numel(tables)
    table_values = parse_pins(header_text, tables{table_index});
    table_names = keys(table_values);
    for name_index = 1:numel(table_names)
        name = table_names{name_index};
        pinned_values(name) = table_values(name);
    end
end

matlab_names = sort(keys(matlab_values));
pinned_names = sort(keys(pinned_values));
if ~isequal(matlab_names, pinned_names)
    missing_from_matlab = setdiff(pinned_names, matlab_names);
    missing_from_header = setdiff(matlab_names, pinned_names);
    error('MATLAB/header feature mismatch. Missing from MATLAB: %s. Missing from header: %s.', ...
        strjoin(missing_from_matlab, ', '), strjoin(missing_from_header, ', '));
end

relative_tolerance = 1e-3;
failures = 0;
fprintf('MATLAB R%s: generated values vs pinned C++ oracle goldens\n', version('-release'));
fprintf('%-28s %22s %22s %12s\n', 'feature', 'MATLAB', 'C++ golden', 'relative error');
for name_index = 1:numel(matlab_names)
    name = matlab_names{name_index};
    matlab_value = matlab_values(name);
    nyxus_value = pinned_values(name);
    relative_error = abs(matlab_value - nyxus_value) / max(abs(nyxus_value), 1e-12);
    fprintf('%-28s %22.17g %22.17g %12.3e\n', ...
        name, matlab_value, nyxus_value, relative_error);
    failures = failures + (relative_error > relative_tolerance);
end

fprintf('\nPaste-ready MATLAB R2026a goldens\n');
for name_index = 1:numel(matlab_names)
    name = matlab_names{name_index};
    fprintf('\t{"%s", %.17g},\n', name, matlab_values(name));
end

if failures > 0
    error('%d of %d MATLAB goldens exceed rel=1e-3.', failures, numel(matlab_names));
end
fprintf('\nAll %d MATLAB goldens agree within rel=1e-3.\n', numel(matlab_names));

function [mask, intensity] = load_shape2d(text)
    mask = parse_pixels(text, 'shape2d_morphology_mask');
    intensity = parse_pixels(text, 'shape2d_morphology_intensity');
end

function values = parse_pixels(text, array_name)
    marker = [array_name '[] = {'];
    start_index = strfind(text, marker);
    if isempty(start_index)
        error('Array %s not found in tests/test_data.h.', array_name);
    end
    remainder = text(start_index(1) + length(marker):end);
    end_index = strfind(remainder, '};');
    body = remainder(1:end_index(1) - 1);
    entries = regexp(body, '\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}', 'tokens');
    values = zeros(8, 8);
    for entry_index = 1:numel(entries)
        x = str2double(entries{entry_index}{1});
        y = str2double(entries{entry_index}{2});
        value = str2double(entries{entry_index}{3});
        values(y + 1, x + 1) = value;
    end
end

function pins = parse_pins(text, table_name)
    marker = [table_name '{'];
    start_index = strfind(text, marker);
    if isempty(start_index)
        error('Table %s not found in tests/test_2d_morphology_matlab.h.', table_name);
    end
    remainder = text(start_index(1) + length(marker):end);
    end_index = strfind(remainder, '};');
    body = remainder(1:end_index(1) - 1);
    body = regexprep(body, '//[^\n]*', '');
    entries = regexp(body, '\{\s*"([A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', 'tokens');
    pins = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for entry_index = 1:numel(entries)
        pins(entries{entry_index}{1}) = str2double(entries{entry_index}{2});
    end
end
