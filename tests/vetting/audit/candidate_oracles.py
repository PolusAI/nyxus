"""Map a feature (by name / family) to the external oracle that COULD vet it — the actionable
"how to move toward vetting" column. Ordered rules; first match wins."""
import re

# (regex on FEATURE or FAMILY, candidate oracle, note)
RULES = [
    # texture families -> IBSI reference values are the gold standard, pyradiomics second
    (r"GLCM",  "IBSI + pyradiomics", "IBSI has GLCM reference tables; pyradiomics computes all"),
    (r"GLRLM", "IBSI + pyradiomics", "IBSI GLRLM reference; pyradiomics"),
    (r"GLSZM", "IBSI + pyradiomics", "IBSI GLSZM reference; pyradiomics"),
    (r"GLDZM", "IBSI + pyradiomics", "IBSI GLDZM reference"),
    (r"GLDM|GLDN", "IBSI + pyradiomics", "IBSI GLDM/dependence; pyradiomics gldm"),
    (r"NGTDM", "IBSI + pyradiomics", "IBSI NGTDM reference; pyradiomics"),
    (r"NGLDM", "IBSI + pyradiomics", "IBSI NGLDM reference"),
    # first-order intensity
    (r"Intensity|INTENSITY|ENERGY|ENTROPY|KURTOSIS|SKEWNESS|MEAN|MEDIAN|VARIANCE|STANDARD_|RANGE|"
     r"UNIFORMITY|PERCENTILE|^P[0-9]|QCOD|MODE|ENERGY|ROOT_MEAN|MAX$|MIN$|COV",
     "IBSI first-order + pyradiomics", "IBSI first-order intensity reference; pyradiomics firstorder"),
    (r"histogram|HISTOGRAM", "numpy/scipy analytic", "histogram stats reproducible with numpy"),
    # morphology / shape -> scikit-image regionprops
    (r"AREA|PERIMETER|CENTROID|BBOX|EXTENT|ASPECT|ORIENTATION|ECCENTRICITY|ELONGATION|"
     r"MAJOR_AXIS|MINOR_AXIS|EULER|SOLIDITY|CONVEX|CIRCULARITY|COMPACTNESS|ROUNDNESS|DIAMETER|"
     r"Morphology|EQUIVALENT",
     "scikit-image regionprops", "skimage.measure.regionprops covers most 2D shape descriptors"),
    (r"EXTREMA", "scikit-image / analytic", "extrema points derivable analytically / skimage"),
    (r"convex hull|CONVEX_HULL", "scikit-image convex_hull_image", "skimage convex hull (offset_coordinates)"),
    (r"ROI radius|ROI_RADIUS|radius", "scipy distance transform", "max/mean radius via scipy edt"),
    (r"erosion|EROSION", "scikit-image binary_erosion", "erosion count via skimage morphology"),
    (r"Caliper|FERET|MARTIN|NASSENSTEIN|CALIPER", "scikit-image / imagej", "Feret via skimage regionprops feret_diameter_max; ImageJ calipers"),
    (r"GEODETIC|THICKNESS", "skimage skeleton / medial axis", "geodetic length & thickness via skimage medial_axis"),
    (r"NEIGHBOR|Neighbor|touching|TOUCH", "hand-computed adjacency", "no standard tool; verify on constructed adjacency fixtures"),
    (r"Chords|CHORD|MAXCHORD|ALLCHORDS", "analytic constructed shape", "chord stats on shapes with known chords"),
    (r"POLYGON|POLYGONALITY|HEXAGON", "analytic constructed shape", "polygonality/hexagonality on regular polygons"),
    # moments
    (r"Hu.?moment|HU_MOMENT|_HU_", "scikit-image moments_hu / opencv HuMoments", "skimage.measure.moments_hu; cv2.HuMoments"),
    (r"WEIGHTED.*MOMENT|MOMENT.*WEIGHTED", "scikit-image weighted moments", "skimage moments with intensity weights"),
    (r"NORM.*MOMENT|MOMENT.*NORM|NU[0-9]", "scikit-image moments_normalized", "skimage.measure.moments_normalized"),
    (r"CENTRAL.*MOMENT|MOMENT.*CENTRAL|MU[0-9]", "scikit-image moments_central", "skimage.measure.moments_central"),
    (r"RAW.*MOMENT|MOMENT|SPAT.*MOMENT|M[0-9]", "scikit-image moments", "skimage.measure.moments (raw)"),
    # fractal / gabor
    (r"FRACT_DIM|fractal", "ImageJ/FracLac + analytic", "shifting-grid box count + analytic shapes (done)"),
    (r"GABOR", "analytic gabor response", "gabor on synthetic gratings of known frequency"),
    (r"Low-frequency|LOWFREQ|GEODETIC", "analytic / skimage", "constructed fixture"),
    # image quality
    (r"IMQ|FOCUS|SHARP|BLUR|CONTRAST_IQ|BRISQUE|POWER_SPECTRUM|SATURATION|NOISE",
     "reference IQ library (e.g. skimage/BRISQUE) or analytic", "image-quality metrics: compare to a reference implementation or synthetic images"),
]

def candidate_oracle(feature, family):
    hay = (feature or "") + " || " + (family or "")
    for pat, oracle, note in RULES:
        if re.search(pat, hay, re.IGNORECASE):
            return oracle, note
    return "TBD", "no obvious external oracle; consider analytic fixture or manual review"

if __name__ == "__main__":
    for f, fam in [("GLCM_CONTRAST","GLCM"),("AREA_PIXELS_COUNT","Morphology"),("MEAN","Intensity"),
                   ("HU_M1","-- shape Hu's moments 1-7"),("FRACT_DIM_BOXCOUNT","-- fractal dimension"),
                   ("GABOR","Gabor"),("MAX_CHORD_LENGTH","-- Chords")]:
        print(f, "->", candidate_oracle(f, fam))
