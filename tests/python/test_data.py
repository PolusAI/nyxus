import numpy as np

intens = np.array([
       [[1, 4, 4, 1, 1],
        [1, 4, 6, 1, 1],
        [4, 1, 6, 4, 1],
        [4, 4, 6, 4, 1]],
                   
       [[1, 4, 4, 1, 1],
        [1, 1, 6, 1, 1],
        [1, 1, 3, 1, 1],
        [4, 4, 6, 1, 1]],
       
       [[1, 4, 4, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 6, 1, 1],
        [1, 1, 6, 1, 1]],
       
       [[1, 4, 4, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 6, 1, 1]],
])

seg = np.array([
       [[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]],
                
       [[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]],
       
       [[1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1]],
                
       [[1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]]
])

# z-slice of a CT scan (Hounsfeld units)
# -------------------------------------------
# steps to reproduce:
#       n = load_nii ('/data/ct_abdomen/CT3.nii.gz')
#       p14 = n.img (:, :, 14) ;
#       p14s = imresize (imrotate(p14, 90), [16 16]);
#       format bank;
#       disp (p14s)
#
ct_zslice_hounsfeld_inten = np.array([
      [-1024.00 ,     -1023.97 ,     -1024.36 ,     -1024.74 ,     -1018.48 ,     -1008.29 ,     -1001.06 ,      -997.09 ,      -996.73 ,     -1000.47 ,     -1008.45 ,     -1018.48 ,     -1024.73 ,     -1024.35 ,     -1023.97 ,     -1024.00],
      [-1023.97 ,     -1024.54 ,     -1022.72 ,     -1007.38 ,      -997.02 ,      -995.63 ,      -997.26 ,      -998.29 ,      -998.21 ,      -997.30 ,      -995.94 ,      -996.74 ,     -1007.35 ,     -1022.81 ,     -1024.52 ,     -1023.97],
      [-1024.35 ,     -1022.84 ,     -1003.64 ,      -996.22 ,      -997.13 ,      -996.50 ,      -996.76 ,      -998.36 ,      -999.37 ,      -997.95 ,      -996.63 ,      -997.13 ,     -996.08 ,     -1004.13 ,     -1022.90 ,     -1024.33],
      [-1024.71 ,     -1007.52 ,      -996.43 ,      -998.23 ,      -998.17 ,     -1001.24 ,     -1005.85 ,     -1006.61 ,     -1008.09 ,     -1009.41 ,     -1004.43 ,      -998.62 ,     -997.87 ,     -997.31 ,     -1008.30 ,     -1024.68],
      [-1018.56 ,      -997.07 ,      -997.88 ,     -1001.62 ,     -1010.75 ,      -983.53 ,      -930.37 ,      -904.36 ,      -879.86 ,      -882.79 ,      -953.21 ,     -1011.84 ,     -1002.44 ,     -999.35 ,      -999.27 ,     -1019.10],
      [-1008.42 ,      -997.96 ,     -1001.60 ,      -987.54 ,      -699.31 ,      -295.02 ,      -143.10 ,      -137.07 ,       -97.02 ,       -65.87 ,      -170.26 ,      -616.72 ,     -1003.51 ,     -1001.49 ,      -999.08 ,     -1010.09],
      [-1003.24 ,      -993.12 ,      -967.44 ,      -742.50 ,        -3.13 ,        87.65 ,        -4.51 ,       -22.25 ,       113.07 ,       108.63 ,       131.14 ,        62.04 ,     -610.30 ,     -999.93 ,      -997.24 ,     -1005.20],
      [-972.51 ,      -945.79 ,      -969.52 ,      -233.06 ,        36.31 ,        14.13 ,        22.76 ,        -1.42 ,        96.20 ,       104.25 ,       101.75 ,       120.68 ,     -4.56 ,     -855.61 ,     -1014.04 ,     -1007.62],
      [-852.37 ,      -360.25 ,      -547.89 ,       -15.64 ,        65.50 ,         7.91 ,        79.59 ,        22.13 ,        55.91 ,        98.84 ,       101.73 ,        98.18 ,     107.50 ,     -417.97 ,      -738.23 ,      -918.31],
      [-511.83 ,       222.33 ,      -114.05 ,         4.93 ,        94.08 ,        36.73 ,        35.32 ,       104.67 ,       118.84 ,        80.51 ,        95.52 ,        87.93 ,     77.17 ,     -67.45 ,        81.24 ,      -336.73],
      [-450.69 ,       278.17 ,      -135.16 ,        -5.34 ,        81.48 ,        73.26 ,        35.98 ,       106.48 ,       219.95 ,        85.04 ,        71.01 ,        76.55 ,     38.58 ,     -125.60 ,       250.75 ,      -296.22],
      [-912.35 ,      -648.92 ,      -623.68 ,      -103.38 ,        85.66 ,       112.12 ,        20.32 ,        52.57 ,       115.67 ,        17.68 ,        82.00 ,        88.66 ,     -47.38 ,     -391.65 ,      -371.17 ,      -844.75],
      [-1047.43 ,      -836.18 ,      -695.41 ,      -727.57 ,      -525.43 ,      -207.95 ,       -85.49 ,       -60.48 ,       -51.08 ,       -91.02 ,      -200.76 ,      -476.63 ,     -710.48 ,     -717.23 ,      -867.83 ,     -1048.78],
      [-1025.64 ,     -1024.21 ,      -874.48 ,      -669.48 ,      -692.81 ,      -729.16 ,      -698.63 ,      -698.16 ,      -704.74 ,      -719.20 ,      -737.88 ,      -694.85 ,     -663.07 ,     -849.33 ,     -1011.60 ,     -1026.81],
      [-1023.85 ,     -1029.05 ,     -1003.33 ,      -825.02 ,      -754.70 ,      -724.03 ,      -714.48 ,      -711.62 ,      -717.40 ,      -722.80 ,      -733.36 ,      -761.48 ,     -842.31 ,     -1022.24 ,     -1028.23 ,     -1023.77],
      [-1024.00 ,     -1023.62 ,     -1027.94 ,     -1034.10 ,      -986.49 ,      -914.63 ,      -856.92 ,      -836.73 ,      -854.55 ,      -884.40 ,      -925.75 ,      -982.15 ,     -1028.24 ,     -1026.52 ,     -1023.79 ,     -1024.00]
      ], np.float64)

# trivial mask matching dimensions of 'ct_zslice_hounsfeld_inten'
ct_zslice_hounsfeld_mask = np.ones((16, 16), dtype=np.int32)



# ---------------------------------------------------------------------------------------------
# bench_disk64_diagonal_boundary (tests/vetting/benchmarks.md)
#
# One 64x64 disk. The boundary is genuinely DIAGONAL, which is the property the other out-of-core
# fixtures lack: around a rectangle every contour step is an axis-aligned unit step, so the contour
# pixel COUNT and the sum of Euclidean step lengths are the same number and the two contour
# implementations agree by construction rather than by correctness. On this disk they separate.
#
# Built here rather than copied into each test: three modules read it, and three copies would drift
# independently -- the pins in one would stop describing the image another builds.
DISK64_SIDE = 64
DISK64_CENTRE = 32
DISK64_RADIUS = 20


def disk64_arrays():
    """-> (intensity, mask) for bench_disk64_diagonal_boundary.

    1257 ROI pixels, intensity 117..397, total 323049; 112 of them lie on the 4-neighbour inner
    boundary -- and 112 is exactly what the out-of-core path returns for PERIMETER, which is what
    identifies that divergence as a difference of definition rather than of accumulation.
    """
    y, x = np.mgrid[0:DISK64_SIDE, 0:DISK64_SIDE]
    mask = (((y - DISK64_CENTRE) ** 2 + (x - DISK64_CENTRE) ** 2) <= DISK64_RADIUS ** 2)
    inten = (1 + x + y * 7) * mask
    return inten.astype(np.uint32), mask.astype(np.uint32)


def write_disk64_pair(tmp_path, tifffile, as_dirs=True):
    """Write the fixture as a TIFF pair. as_dirs -> (intdir, segdir) for featurize_directory;
    otherwise -> (intensity_path, mask_path) for featurize_files."""
    inten, mask = disk64_arrays()
    if as_dirs:
        intdir = tmp_path / "disk_int"
        segdir = tmp_path / "disk_seg"
        intdir.mkdir()
        segdir.mkdir()
        tifffile.imwrite(str(intdir / "img.tif"), inten)
        tifffile.imwrite(str(segdir / "img.tif"), mask)
        import os
        return str(intdir) + os.sep, str(segdir) + os.sep
    ip = tmp_path / "disk_img.tif"
    sp = tmp_path / "disk_seg.tif"
    tifffile.imwrite(str(ip), inten)
    tifffile.imwrite(str(sp), mask)
    return str(ip), str(sp)
