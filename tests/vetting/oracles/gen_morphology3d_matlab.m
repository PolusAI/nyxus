% MATLAB reference generator for 3D morphology Volume and ConvexVolume features.
% MATLAB version: R2026a.
%
% Source fixture (label 57):
%   https://github.com/PolusAI/nyxus/blob/main/tests/data/nifti/phantoms/ut_mask57.nii
%
% Built-ins:
%   niftiread
%   regionprops3(..., 'Volume', 'ConvexVolume')

fixture_permalink = ['https://github.com/PolusAI/nyxus/blob/main/' ...
    'tests/data/nifti/phantoms/ut_mask57.nii'];

mask_file = [tempname '.nii'];
websave(mask_file, [fixture_permalink '?raw=1']);
mask = niftiread(mask_file);
delete(mask_file);

matlab_regionprops3 = regionprops3(mask == 57, 'Volume', 'ConvexVolume');

fprintf('3VOXEL_VOLUME      regionprops3.Volume       %.17g\n', ...
    matlab_regionprops3.Volume);
fprintf('3VOLUME_CONVEXHULL regionprops3.ConvexVolume %.17g\n', ...
    matlab_regionprops3.ConvexVolume);
fprintf('3MESH_VOLUME       regionprops3.ConvexVolume %.17g  (Nyxus alias)\n', ...
    matlab_regionprops3.ConvexVolume);
