% MATLAB reference generator for native 3D first-order statistics.
% MATLAB version: R2026a.
%
% Source fixture (label 57):
%   https://github.com/PolusAI/nyxus/tree/main/tests/data/nifti/phantoms
%
% Every reported value is returned directly by the named MATLAB function.

intensity_permalink = ['https://github.com/PolusAI/nyxus/blob/main/' ...
    'tests/data/nifti/phantoms/ut_inten.nii'];
mask_permalink = ['https://github.com/PolusAI/nyxus/blob/main/' ...
    'tests/data/nifti/phantoms/ut_mask57.nii'];

intensity_file = [tempname '.nii'];
mask_file = [tempname '.nii'];
websave(intensity_file, [intensity_permalink '?raw=1']);
websave(mask_file, [mask_permalink '?raw=1']);
intensity = niftiread(intensity_file);
mask = niftiread(mask_file);
delete(intensity_file);
delete(mask_file);

% Match Nyxus' default float-NIfTI loader domain: shift a negative volume minimum to zero,
% then cast the nonnegative values to integers. This is fixture setup, not a feature formula.
intensity = double(intensity);
intensity = fix(intensity - min(intensity, [], 'all'));
voxels = intensity(mask == 57);

fprintf('3INTEGRATED_INTENSITY       sum            %.17g\n', sum(voxels));
fprintf('3INTERQUARTILE_RANGE        iqr            %.17g\n', iqr(voxels));
fprintf('3KURTOSIS                   kurtosis       %.17g\n', kurtosis(voxels, 1));
fprintf('3MAX                        max            %.17g\n', max(voxels));
fprintf('3MEAN                       mean           %.17g\n', mean(voxels));
fprintf('3MEAN_ABSOLUTE_DEVIATION    mad            %.17g\n', mad(voxels, 0));
fprintf('3MEDIAN                     median         %.17g\n', median(voxels));
fprintf('3MIN                        min            %.17g\n', min(voxels));
fprintf('3MODE                       mode           %.17g\n', mode(voxels));
fprintf('3P01                        prctile        %.17g\n', ...
    prctile(voxels, 1, Method="midpoint"));
fprintf('3P10                        prctile        %.17g\n', ...
    prctile(voxels, 10, Method="midpoint"));
fprintf('3P25                        prctile        %.17g\n', ...
    prctile(voxels, 25, Method="midpoint"));
fprintf('3P75                        prctile        %.17g\n', ...
    prctile(voxels, 75, Method="midpoint"));
fprintf('3P90                        prctile        %.17g\n', ...
    prctile(voxels, 90, Method="midpoint"));
fprintf('3P99                        prctile        %.17g\n', ...
    prctile(voxels, 99, Method="midpoint"));
fprintf('3RANGE                      range          %.17g\n', range(voxels));
fprintf('3ROOT_MEAN_SQUARED          rms            %.17g\n', rms(voxels));
fprintf('3SKEWNESS                   skewness       %.17g\n', skewness(voxels, 1));
fprintf('3STANDARD_DEVIATION         std            %.17g\n', std(voxels, 0));
fprintf('3STANDARD_DEVIATION_BIASED  std            %.17g\n', std(voxels, 1));
fprintf('3VARIANCE                   var            %.17g\n', var(voxels, 0));
fprintf('3VARIANCE_BIASED            var            %.17g\n', var(voxels, 1));
