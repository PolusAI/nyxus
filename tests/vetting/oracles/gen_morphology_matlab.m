%% OFFLINE MATLAB/Octave oracle for the 2D morphology features vetted against regionprops.
%
% Run from the repository root:
%     octave tests/vetting/oracles/gen_morphology_matlab.m
%
% Prints the paste-ready golden table AND re-verifies every golden currently pinned in
% tests/test_2d_morphology_matlab.h -- both the regionprops table and the extrema table. Any pin the
% generator cannot produce is reported by name, and the script exits non-zero on any mismatch. A
% hand-picked validation list is the failure mode this avoids: it silently stops covering whatever
% is added later.
%
% Provenance: tool=GNU Octave 11.3.0 + image package 2.20.0, used as the license-free MATLAB
% stand-in (tests/vetting/TOOLS.md); config=make_shape2d_settings() in tests/test_main_nyxus.h,
% i.e. PIXELSIZEUM=2.0, IBSI=false, single ROI. The fixture is read out of tests/test_data.h so the
% generator and the C++ tests share one copy of the pixels -- same discipline as
% tests/vetting/oracles/ibsi_phantom.py. CI never invokes this script.

1;

function [M, I] = load_shape2d(hdr)
  txt = slurp(hdr);
  M = parse_pixels(txt, 'shape2d_morphology_mask');
  I = parse_pixels(txt, 'shape2d_morphology_intensity');
end

function txt = slurp(path)
  fid = fopen(path, 'r');
  if fid < 0
    error('cannot open %s (run this script from the repository root)', path);
  end
  txt = fread(fid, Inf, '*char')';
  fclose(fid);
end

function A = parse_pixels(txt, name)
  key = [name '[] = {'];
  i = strfind(txt, key);
  if isempty(i), error('array %s not found in test_data.h', name); end
  rest = txt(i(1)+length(key):end);
  j = strfind(rest, '};');
  body = rest(1:j(1)-1);
  tok = regexp(body, '\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}', 'tokens');
  A = zeros(8, 8);
  for k = 1:numel(tok)
    x = str2double(tok{k}{1}); y = str2double(tok{k}{2}); v = str2double(tok{k}{3});
    A(y+1, x+1) = v;                      % header is {x, y, value}; Octave indexes (row, col)
  end
end

% Every {"NAME", value} entry of one named ref_vals_map in a test header.
function pins = parse_pins(txt, table_name)
  key = [table_name '{'];
  i = strfind(txt, key);
  if isempty(i), error('table %s not found', table_name); end
  rest = txt(i(1)+length(key):end);
  j = strfind(rest, '};');
  body = rest(1:j(1)-1);
  body = regexprep(body, '//[^\n]*', '');          % a commented-out golden is not a pin
  tok = regexp(body, '\{\s*"([A-Z0-9_]+)"\s*,\s*([-0-9.eE+]+)\s*\}', 'tokens');
  pins = containers.Map();
  for k = 1:numel(tok)
    pins(tok{k}{1}) = str2double(tok{k}{2});
  end
end

pkg load image;

hdr_data = fullfile('tests', 'test_data.h');
hdr_test = fullfile('tests', 'test_2d_morphology_matlab.h');

[Mv, Iv] = load_shape2d(hdr_data);
M = logical(Mv);
PIXELSIZEUM = 2.0;                        % NyxSetting::PIXELSIZEUM in make_shape2d_settings()

s = regionprops(M, 'Area', 'Centroid', 'BoundingBox', 'Extent', 'MajorAxisLength', ...
                   'MinorAxisLength', 'Eccentricity', 'Extrema');
w = regionprops(M, Iv, 'WeightedCentroid');

got = containers.Map();
got('AREA_PIXELS_COUNT')   = s(1).Area;
got('AREA_UM2')            = s(1).Area * PIXELSIZEUM^2;
got('CENTROID_X')          = s(1).Centroid(1) - 1;          % 1-based centres -> 0-based
got('CENTROID_Y')          = s(1).Centroid(2) - 1;
got('WEIGHTED_CENTROID_X') = w(1).WeightedCentroid(1) - 1;
got('WEIGHTED_CENTROID_Y') = w(1).WeightedCentroid(2) - 1;
got('BBOX_XMIN')           = s(1).BoundingBox(1) - 0.5;     % corner at min-0.5, 1-based
got('BBOX_YMIN')           = s(1).BoundingBox(2) - 0.5;
got('BBOX_WIDTH')          = s(1).BoundingBox(3);
got('BBOX_HEIGHT')         = s(1).BoundingBox(4);
got('ASPECT_RATIO')        = s(1).BoundingBox(3) / s(1).BoundingBox(4);
got('EXTENT')              = s(1).Extent;
got('MAJOR_AXIS_LENGTH')   = s(1).MajorAxisLength;
got('MINOR_AXIS_LENGTH')   = s(1).MinorAxisLength;
got('ELONGATION')          = s(1).MinorAxisLength / s(1).MajorAxisLength;
got('ECCENTRICITY')        = s(1).Eccentricity;
got('EULER_NUMBER')        = bweuler(M, 8);

% regionprops('Extrema') returns 8 sub-pixel CORNER points, 1-based, in the order
% top-left, top-right, right-top, right-bottom, bottom-right, bottom-left, left-bottom, left-top.
% Nyxus returns 0-based pixel CENTRES, so the corner offset is direction-specific: a left or top
% coordinate maps as (matlab - 0.5), a right or bottom one as (matlab - 1.5).
ex = s(1).Extrema;
dx = [-0.5, -1.5, -1.5, -1.5, -1.5, -0.5, -0.5, -0.5];
dy = [-0.5, -0.5, -0.5, -1.5, -1.5, -1.5, -1.5, -0.5];
for p = 1:8
  got(sprintf('EXTREMA_P%d_X', p)) = ex(p, 1) + dx(p);
  got(sprintf('EXTREMA_P%d_Y', p)) = ex(p, 2) + dy(p);
end

printf('# octave %s, image %s\n', version(), ver('image')(1).Version);
printf('# paste-ready goldens (17 significant digits)\n');
names = sort(keys(got));
for k = 1:numel(names)
  printf('\t{"%s", %.17g},\n', names{k}, got(names{k}));
end

% ---- verify EVERY pin in the header this generator feeds -------------------------------------
txt = slurp(hdr_test);
tables = {'morphology_2d_matlab_regionprops_ref_vals', 'morphology_2d_matlab_extrema_ref_vals'};
all_pins = containers.Map();
for t = 1:numel(tables)
  p = parse_pins(txt, tables{t});
  kk = keys(p);
  for k = 1:numel(kk)
    all_pins(kk{k}) = p(kk{k});
  end
end

RELTOL = 1e-3;             % SPEC 7 same-definition-oracle tier, matching the C++ assertions
nfail = 0; nmiss = 0; nok = 0;
pin_names = sort(keys(all_pins));
printf('\n# verifying %d pinned goldens against this run\n', numel(pin_names));
for k = 1:numel(pin_names)
  nm = pin_names{k};
  want = all_pins(nm);
  if ~isKey(got, nm)
    printf('  MISSING %s: pinned %.17g but this generator produces no such value\n', nm, want);
    nmiss++;
    continue;
  end
  have = got(nm);
  denom = max(abs(want), 1e-12);
  rel = abs(have - want) / denom;
  if rel <= RELTOL
    printf('  OK   %s: octave=%.17g pinned=%.17g rel=%.3g\n', nm, have, want, rel);
    nok++;
  else
    printf('  FAIL %s: octave=%.17g pinned=%.17g rel=%.3g\n', nm, have, want, rel);
    nfail++;
  end
end

printf('\n%d verified, %d failed, %d unproducible\n', nok, nfail, nmiss);
if nfail > 0 || nmiss > 0
  printf('SOME CHECKS FAILED -- do not promote\n');
  exit(1);
end
printf('ALL CHECKS PASSED\n');
exit(0);
