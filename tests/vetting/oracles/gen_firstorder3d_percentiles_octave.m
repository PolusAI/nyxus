%% GNU Octave surrogate generator for the percentile rows of the 3D first-order MATLAB table
% Octave version: 11.3.0 (core quantile; no package needed).
%
% Source fixture and pinned goldens (read from the checkout this runs in):
%   tests/data/nifti/phantoms/ut_inten.nii, ut_mask57.nii   label 57
%   tests/test_3d_firstorder_matlab.h                        firstorder_3d_matlab_ref_vals
%
% Octave's prctile takes no Method argument. The rule MATLAB's prctile(x, p, Method="inclusive")
% takes, numpy's default 'linear' ranks (k-1)/(n-1), is quantile(x, p/100, 1, 7) here. The
% interquartile range is P75 - P25 and QCOD is (P75 - P25) / (P75 + P25). The table's other
% rows are pinned by gen_firstorder3d_matlab.m.
%
% Run from the repository root:
%   octave-cli tests/vetting/oracles/gen_firstorder3d_percentiles_octave.m

1;

% A single-file NIfTI-1 volume as stored, the way MATLAB's niftiread returns it: no scaling is
% applied. Only the little-endian uint8 and float32 types the fixture uses are read.
function volume = read_nifti(path)
    fid = fopen(path, 'r', 'ieee-le');
    if fid < 0
        error('cannot open %s', path);
    end
    cleanup = onCleanup(@() fclose(fid));
    if fread(fid, 1, 'int32') ~= 348
        error('%s is not a little-endian NIfTI-1 file', path);
    end
    fseek(fid, 40, 'bof');
    dims = fread(fid, 8, 'int16');
    fseek(fid, 70, 'bof');
    datatype = fread(fid, 1, 'int16');
    fseek(fid, 108, 'bof');
    vox_offset = fread(fid, 1, 'float32');
    switch datatype
        case 2
            precision = 'uint8=>double';
        case 16
            precision = 'float32=>double';
        otherwise
            error('%s: NIfTI datatype %d is not read here', path, datatype);
    end
    shape = dims(2:dims(1) + 1)';
    fseek(fid, vox_offset, 'bof');
    volume = reshape(fread(fid, prod(shape), precision), shape);
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

intensity = read_nifti('tests/data/nifti/phantoms/ut_inten.nii');
mask = read_nifti('tests/data/nifti/phantoms/ut_mask57.nii');
pinned = parse_pins(fileread('tests/test_3d_firstorder_matlab.h'), 'firstorder_3d_matlab_ref_vals');

% Match what Nyxus reports, as gen_firstorder3d_matlab.m does: the loader offsets a volume holding
% negative values by its floored minimum and truncates to integer grey levels, and the intensity
% family adds that offset back. Fixture setup, not a feature formula.
offset = floor(min(intensity(:)));
if offset < 0
    intensity = fix(intensity - offset) + offset;
else
    intensity = fix(intensity);
end
voxels = intensity(mask == 57);

percents = [1 10 25 75 90 99];
q = quantile(voxels, percents / 100, 1, 7);
values = containers.Map('KeyType', 'char', 'ValueType', 'double');
for k = 1:numel(percents)
    values(sprintf('3P%02d', percents(k))) = q(k);
end
values('3INTERQUARTILE_RANGE') = values('3P75') - values('3P25');
values('3QCOD') = (values('3P75') - values('3P25')) / (values('3P75') + values('3P25'));

% the band of firstorder_3d_matlab_rel_tol() in test_3d_firstorder_matlab.h for these rows
relative_tolerance = 1e-9;
names = sort(keys(values));
failures = 0;
printf('GNU Octave %s: 3D first-order percentiles of %d voxels\n', OCTAVE_VERSION, numel(voxels));
printf('%-22s %24s %24s %12s\n', 'feature', 'Octave', 'C++ golden', 'rel. error');
for k = 1:numel(names)
    name = names{k};
    if ~isKey(pinned, name)
        error('%s is not pinned in firstorder_3d_matlab_ref_vals.', name);
    end
    relative_error = abs(values(name) - pinned(name)) / max(abs(pinned(name)), 1e-12);
    printf('%-22s %24.17g %24.17g %12.3e\n', name, values(name), pinned(name), relative_error);
    failures = failures + (relative_error > relative_tolerance);
end
if failures > 0
    error('%d of %d percentile goldens exceed rel=%g.', failures, numel(names), relative_tolerance);
end
printf('\nAll %d percentile goldens agree within rel=%g.\n', numel(names), relative_tolerance);
