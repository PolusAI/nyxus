// Headless FracLac-style SHIFTING-GRID box-counting oracle for ImageJ/Fiji.
// FracLac's key improvement over a single grid is scanning multiple grid origins and
// taking the MINIMUM box count per scale (the true covering number), which removes the
// grid-registration bias. This macro reproduces that method headlessly (FracLac itself
// is GUI-only), so the box-count oracle runs with zero clicking.
//
// Usage: fiji-linux-x64 --headless -macro shiftgrid_boxcount.ijm "<masks_dir>"
// Prints: RESULT <file> D=<dim> boxes=<n>   (D = -slope of log(minCount) vs log(boxSize))

setBatchMode(true);
dir = getArgument();
if (dir == "") dir = "masks/";
if (!endsWith(dir, "/")) dir = dir + "/";
list = getFileList(dir);

for (f = 0; f < list.length; f++) {
    name = list[f];
    if (!endsWith(name, ".tif")) continue;
    open(dir + name);
    run("8-bit");
    W = getWidth(); H = getHeight();

    // Collect foreground (object) pixel coordinates once. Object = value > 127.
    nfg = 0;
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++)
            if (getPixel(x, y) > 127) nfg++;
    fx = newArray(nfg); fy = newArray(nfg);
    k = 0;
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++)
            if (getPixel(x, y) > 127) { fx[k] = x; fy[k] = y; k++; }

    // Box sizes: powers of two from 2 up to <= min(W,H).
    sizes = newArray(); s = 2;
    while (s <= minOf(W, H)) { sizes = Array.concat(sizes, s); s = s * 2; }

    logx = newArray(sizes.length);
    logy = newArray(sizes.length);
    for (si = 0; si < sizes.length; si++) {
        s = sizes[si];
        // shifting grids: sample origins {0, s/2} in each axis -> 4 grid positions; take MIN count
        offs = newArray(0, floor(s / 2));
        best = -1;
        for (oyi = 0; oyi < offs.length; oyi++) {
            for (oxi = 0; oxi < offs.length; oxi++) {
                ox = offs[oxi]; oy = offs[oyi];
                nbcols = floor((W + ox) / s) + 1;
                nbrows = floor((H + oy) / s) + 1;
                flags = newArray(nbcols * nbrows);   // zero-initialized
                c = 0;
                for (i = 0; i < fx.length; i++) {
                    bc = floor((fx[i] + ox) / s);
                    br = floor((fy[i] + oy) / s);
                    idx = br * nbcols + bc;
                    if (flags[idx] == 0) { flags[idx] = 1; c++; }
                }
                if (best < 0 || c < best) best = c;
            }
        }
        logx[si] = log(s);
        logy[si] = log(best);
    }

    D = -lstsqSlope(logx, logy);
    print("RESULT " + name + " D=" + D + " boxes=" + sizes.length);
    close();
}
run("Quit");

function lstsqSlope(x, y) {
    n = x.length; sx = 0; sy = 0; sxy = 0; sx2 = 0;
    for (i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxy += x[i]*y[i]; sx2 += x[i]*x[i]; }
    return (sxy*n - sx*sy) / (sx2*n - sx*sx);
}
