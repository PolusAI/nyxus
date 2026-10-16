"""Generate the feature list and feature-group pages from the code.

    python docs/tools/gen_feature_reference.py            # write the pages
    python docs/tools/gen_feature_reference.py --check    # fail if a page is stale or the
                                                          # descriptions and the code disagree

Codes, families and groups come from the C++ sources (feature_registry.py). One-line
descriptions come from docs/feature_descriptions.toml, keyed by family and code; --check fails
on a code that has no entry there and on an entry that names no code. The validation column
comes from tests/vetting/oracle_coverage.csv, rolled up per feature: a feature takes the
strongest status of its rows (vetted, then invariant, then regression, then untested), and a
vetted feature lists the oracles of its vetted rows. tests/vetting/check_coverage.py ranks
invariant and regression equally for its counts; the feature list ranks invariant higher, as
docs/source/validation.rst describes.
"""

import argparse
import csv
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_registry as reg  # noqa: E402

REPO = reg.REPO
DESCRIPTIONS = REPO / "docs" / "feature_descriptions.toml"
REGISTRY = REPO / "tests" / "vetting" / "oracle_coverage.csv"
LIST_PAGE = REPO / "docs" / "source" / "featurelist.rst"
GROUPS_PAGE = REPO / "docs" / "source" / "featuregroups.rst"

# Families in page order: (slug, title, dimension, feature classes). A code belongs to the family
# whose class's featureset lists it; CODE_FAMILY places the codes no featureset lists.
FAMILIES = [
    ("intensity", "Intensity", "2D", ["PixelIntensityFeatures"]),
    ("intensity_histogram", "IBSI intensity histogram", "2D", ["IntensityHistogramFeatures"]),
    ("basic_morphology", "Basic morphology", "2D", ["BasicMorphologyFeatures"]),
    ("contour", "Contour", "2D", ["ContourFeature"]),
    ("convex_hull", "Convex hull", "2D", ["ConvexHullFeature"]),
    ("ellipse", "Ellipse fitting", "2D", ["EllipseFittingFeature"]),
    ("circles", "Inscribing and circumscribing circles", "2D", ["EnclosingInscribingCircumscribingCircleFeature"]),
    ("roi_radius", "ROI radius", "2D", ["RoiRadiusFeature"]),
    ("extrema", "Extrema", "2D", ["ExtremaFeature"]),
    ("euler", "Euler number", "2D", ["EulerNumberFeature"]),
    ("erosion", "Erosion", "2D", ["ErosionPixelsFeature"]),
    ("fractal", "Fractal dimension", "2D", ["FractalDimensionFeature"]),
    ("geodetic", "Geodetic length and thickness", "2D", ["GeodeticLengthThicknessFeature"]),
    ("feret", "Feret diameters", "2D", ["CaliperFeretFeature"]),
    ("martin", "Martin diameters", "2D", ["CaliperMartinFeature"]),
    ("nassenstein", "Nassenstein diameters", "2D", ["CaliperNassensteinFeature"]),
    ("chords", "Chords", "2D", ["ChordsFeature"]),
    ("neighbors", "Neighbors", "2D", ["NeighborsFeature"]),
    ("glcm", "Grey-level co-occurrence matrix (GLCM)", "2D", ["GLCMFeature"]),
    ("glrlm", "Grey-level run-length matrix (GLRLM)", "2D", ["GLRLMFeature"]),
    ("glszm", "Grey-level size-zone matrix (GLSZM)", "2D", ["GLSZMFeature"]),
    ("gldm", "Grey-level dependence matrix (GLDM)", "2D", ["GLDMFeature"]),
    ("gldzm", "Grey-level distance-zone matrix (GLDZM)", "2D", ["GLDZMFeature"]),
    ("ngldm", "Neighbouring grey-level dependence matrix (NGLDM)", "2D", ["NGLDMfeature"]),
    ("ngtdm", "Neighbourhood grey-tone difference matrix (NGTDM)", "2D", ["NGTDMFeature"]),
    ("shape_moments", "Shape moments", "2D", ["Smoms2D_feature"]),
    ("intensity_moments", "Intensity moments", "2D", ["Imoms2D_feature"]),
    ("gabor", "Gabor", "2D", ["GaborFeature"]),
    ("zernike", "Zernike", "2D", ["ZernikeFeature"]),
    ("radial", "Radial distribution", "2D", ["RadialDistributionFeature"]),
    ("intensity_3d", "Intensity", "3D", ["D3_VoxelIntensityFeatures"]),
    ("shape_3d", "Shape", "3D", ["D3_SurfaceFeature"]),
    ("glcm_3d", "GLCM", "3D", ["D3_GLCM_feature"]),
    ("glrlm_3d", "GLRLM", "3D", ["D3_GLRLM_feature"]),
    ("glszm_3d", "GLSZM", "3D", ["D3_GLSZM_feature"]),
    ("gldm_3d", "GLDM", "3D", ["D3_GLDM_feature"]),
    ("gldzm_3d", "GLDZM", "3D", ["D3_GLDZM_feature"]),
    ("ngldm_3d", "NGLDM", "3D", ["D3_NGLDM_feature"]),
    ("ngtdm_3d", "NGTDM", "3D", ["D3_NGTDM_feature"]),
    ("image_quality", "Image quality", "IMQ", ["*"]),
]
CODE_FAMILY = {("2D", "HISTOGRAM"): "intensity"}

# Families with a definition page. The generator writes the page's feature index (`index`, an
# include file) from docs/feature_extras.toml and the registry, and checks that the page defines an
# entry, `.. _feature-<code>:`, for every base code of the family.
EXTRAS = REPO / "docs" / "feature_extras.toml"
FAMILY_PAGES = {
    "glcm": {"page": "Math/f_glcm", "index": "Math/glcm_index.inc"},
    "glrlm": {"page": "Math/f_glrlm", "index": "Math/glrlm_index.inc"},
    "glszm": {"page": "Math/f_glszm", "index": "Math/glszm_index.inc"},
    "gldm": {"page": "Math/f_gldm", "index": "Math/gldm_index.inc"},
    "gldzm": {"page": "Math/f_gldzm", "index": "Math/gldzm_index.inc"},
    "ngldm": {"page": "Math/f_ngldm", "index": "Math/ngldm_index.inc"},
    "ngtdm": {"page": "Math/f_ngtdm", "index": "Math/ngtdm_index.inc"},
    "intensity": {"page": "Math/f_pixel_intensity", "index": "Math/intensity_index.inc"},
    "shape_moments": {"page": "Math/f_2dmoments", "index": "Math/shape_moments_index.inc"},
    "intensity_moments": {"page": "Math/f_2dmoments", "index": "Math/intensity_moments_index.inc"},
    "zernike": {"page": "Math/f_zernike", "index": "Math/zernike_index.inc"},
    "gabor": {"page": "Math/f_gabor", "index": "Math/gabor_index.inc"},
    "image_quality": {"page": "Math/f_image_quality", "index": "Math/image_quality_index.inc"},
    "fractal": {"page": "Math/f_fractal_dimension", "index": "Math/fractal_index.inc"},
    "basic_morphology": {"page": "Math/f_morphology", "index": "Math/basic_morphology_index.inc"},
    "contour": {"page": "Math/f_morphology", "index": "Math/contour_index.inc"},
    "convex_hull": {"page": "Math/f_morphology", "index": "Math/convex_hull_index.inc"},
    "ellipse": {"page": "Math/f_morphology", "index": "Math/ellipse_index.inc"},
    "circles": {"page": "Math/f_morphology", "index": "Math/circles_index.inc"},
    "roi_radius": {"page": "Math/f_morphology", "index": "Math/roi_radius_index.inc"},
    "extrema": {"page": "Math/f_morphology", "index": "Math/extrema_index.inc"},
    "euler": {"page": "Math/f_morphology", "index": "Math/euler_index.inc"},
    "geodetic": {"page": "Math/f_morphology", "index": "Math/geodetic_index.inc"},
    "feret": {"page": "Math/f_morphology", "index": "Math/feret_index.inc"},
    "martin": {"page": "Math/f_morphology", "index": "Math/martin_index.inc"},
    "nassenstein": {"page": "Math/f_morphology", "index": "Math/nassenstein_index.inc"},
    "chords": {"page": "Math/f_morphology", "index": "Math/chords_index.inc"},
    "neighbors": {"page": "Math/f_morphology", "index": "Math/neighbors_index.inc"},
    "erosion": {"page": "Math/f_erosion", "index": "Math/erosion_index.inc"},
    "ngldm_3d": {"page": "Math/f_ngldm_3d", "index": "Math/ngldm_3d_index.inc"},
    "glrlm_3d": {"page": "Math/f_glrlm_3d", "index": "Math/glrlm_3d_index.inc"},
    "glcm_3d": {"page": "Math/f_glcm_3d", "index": "Math/glcm_3d_index.inc"},
    "intensity_3d": {"page": "Math/f_intensity_3d", "index": "Math/intensity_3d_index.inc"},
    "ngtdm_3d": {"page": "Math/f_ngtdm_3d", "index": "Math/ngtdm_3d_index.inc"},
    "gldm_3d": {"page": "Math/f_gldm_3d", "index": "Math/gldm_3d_index.inc"},
    "glszm_3d": {"page": "Math/f_glszm_3d", "index": "Math/glszm_3d_index.inc"},
    "gldzm_3d": {"page": "Math/f_gldzm_3d", "index": "Math/gldzm_3d_index.inc"},
    "shape_3d": {"page": "Math/f_morphology_3d", "index": "Math/shape_3d_index.inc"},
    "intensity_histogram": {"page": "Math/f_intensity_histogram", "index": "Math/intensity_histogram_index.inc"},
    "radial": {"page": "Math/f_radial_distribution", "index": "Math/radial_index.inc"},
}


def label(code):
    return "feature-" + code.lower().replace("_", "-")


def base_codes(features, slug):
    # A code ending in _AVE is the angle average of a per-angle code only when that code is also
    # in the family; otherwise (POLYGONALITY_AVE, HEXAGONALITY_AVE) it is a base code of its own.
    members = [f for f in features if family_of(f) == slug]
    codes = {f.code for f in members}
    return [f for f in members if not (f.code.endswith("_AVE") and f.code[:-4] in codes)]
DIM_TITLES = {"2D": "2D features", "3D": "3D features", "IMQ": "Image quality features"}
STATUS_TEXT = {"vetted": "vetted", "invariant": "invariant", "regression": "regression", "untested": "untested"}
ORACLE_NAMES = {
    "pyradiomics": "PyRadiomics", "radiomicsj": "RadiomicsJ", "mirp": "MIRP", "matlab": "MATLAB",
    "cellprofiler": "CellProfiler", "mitk": "MITK", "feature2djava": "Feature2DJava",
    "wndcharm": "WND-CHARM", "imea": "IMEA", "imagej": "ImageJ", "fraclac": "FracLac",
    "ibsi": "IBSI", "analytic": "analytic", "skimage": "scikit-image", "pydicom": "pydicom",
    "opencv": "OpenCV",
}


def family_of(feature):
    if (feature.dim, feature.code) in CODE_FAMILY:
        return CODE_FAMILY[(feature.dim, feature.code)]
    for slug, _, dim, classes in FAMILIES:
        if dim == feature.dim and (classes == ["*"] or feature.feature_class in classes):
            return slug
    return None


def validation(with_tolerance=False):
    """{(dim, code): (status, [oracles])} rolled up from the registry; with_tolerance adds, as a
    third element, {oracle: [tolerances]} of the vetted rows."""
    rank = {"untested": 0, "regression": 1, "invariant": 2, "vetted": 3}
    best, oracles, tols = {}, {}, {}
    with open(REGISTRY, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["dim"].strip(), r["feature"].strip())
            st = r["status"].strip()
            if key not in best or rank[st] > rank[best[key]]:
                best[key] = st
            if st == "vetted" and r["oracle"].strip():
                oracles.setdefault(key, set()).add(r["oracle"].strip())
                tols.setdefault(key, {}).setdefault(r["oracle"].strip(), set())
                if r["tolerance"].strip():
                    tols[key][r["oracle"].strip()].add(r["tolerance"].strip())
    out = {}
    for k, v in best.items():
        ors = sorted(oracles.get(k, ())) if v == "vetted" else []
        per_oracle = {o: sorted(t) for o, t in tols.get(k, {}).items()} if v == "vetted" else {}
        out[k] = (v, ors, per_oracle) if with_tolerance else (v, ors)
    return out


from gen_cli_reference import escape  # noqa: E402  options and group names as literals, the rest escaped


def validation_cell(v):
    if v is None:
        return "untested"
    status, oracles = v
    if status == "vetted":
        return "vetted: " + ", ".join(ORACLE_NAMES.get(o, o) for o in oracles)
    return STATUS_TEXT[status]


def render_list(features, desc, notes, valid):
    out = [
        ".. This page is generated by docs/tools/gen_feature_reference.py. Edit",
        "   docs/feature_descriptions.toml and re-run the script; do not edit this file by hand.",
        "",
        "Feature list",
        "============",
        "",
        f"Nyxus computes {sum(f.dim == '2D' for f in features)} features for 2D images, "
        f"{sum(f.dim == '3D' for f in features)} for 3D volumes and "
        f"{sum(f.dim == 'IMQ' for f in features)} image quality features. Request a feature by its "
        "code, or a family of features by its group (see :doc:`featuregroups`).",
        "",
        "The *Validation* column says how each feature is tested:",
        "",
        "* **vetted** -- the test suite checks its value against an independent implementation,",
        "  published reference values (IBSI) or a closed-form calculation (analytic), named in the",
        "  column;",
        "* **invariant** -- the test suite checks a relation its value must satisfy, such as equality",
        "  between two Nyxus code paths;",
        "* **regression** -- the test suite pins its current value, so a change is noticed, but no",
        "  independent implementation confirms it;",
        "* **untested** -- no test checks its value.",
        "",
    ]
    for dim in ("2D", "3D", "IMQ"):
        out += [DIM_TITLES[dim], "-" * len(DIM_TITLES[dim]), ""]
        for slug, title, fdim, _ in FAMILIES:
            if fdim != dim:
                continue
            rows = [f for f in features if family_of(f) == slug]
            if not rows:
                continue
            heading = title if dim != "3D" else f"{title} (3D)"
            out += [f".. _features-{slug.replace('_', '-')}:", "", heading, "^" * len(heading), ""]
            if notes.get(slug):
                out += [escape(" ".join(notes[slug].split())), ""]
            if slug in FAMILY_PAGES:
                out += [f"Definitions, formulas and settings: :doc:`/{FAMILY_PAGES[slug]['page']}`.", ""]
            out += [".. list-table::", "   :header-rows: 1", "   :widths: 30 50 20", "",
                    "   * - Code", "     - Description", "     - Validation"]
            for f in rows:
                text = desc.get(slug, {}).get(f.code, "")
                out += [f"   * - ``{f.code}``",
                        f"     - {escape(' '.join(text.split())) or '—'}",
                        f"     - {validation_cell(valid.get((f.dim, f.code)))}"]
            out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def render_groups(features, groups):
    out = [
        ".. This page is generated by docs/tools/gen_feature_reference.py; do not edit it by hand.",
        "",
        "Feature groups",
        "==============",
        "",
        "A group name requests every feature in the group. Group names and feature codes can be",
        "mixed in one request, and are not case-sensitive on the command line:",
        "",
        ".. code-block:: bash",
        "",
        "   nyxus --features=\"*ALL_INTENSITY*,*ALL_GLCM*,ZERNIKE2D\" ...",
        "",
        "``*ALL*`` requests every 2D feature, ``*3D_ALL*`` every 3D feature. Image quality features",
        "are requested only through their own codes or ``*ALL_IMQ*``, and not together with other",
        "features. The intensity histogram features (``IH_*``, in ``*ALL*`` and ``*ALL_IH*``) are",
        "computed only in :doc:`IBSI mode <ibsi_mode>`; without it they are left out of the output.",
        "",
        "A group or feature name prefixed with ``-`` is removed from the request (except ``-*ALL*``",
        "and ``-COV``, which add, and image quality feature codes, where ``-`` is ignored and the",
        "feature is added; ``-*ALL_IMQ*`` stops with an error), and names are applied from left to",
        "right; see",
        ":ref:`requesting-feature-groups`. ``HISTOGRAM`` is in ``*ALL*`` but in no other group, not",
        "even ``*ALL_INTENSITY*``.",
        "",
    ]
    titles = {slug: (title if dim != "3D" else f"{title} (3D)") for slug, title, dim, _ in FAMILIES}
    unusable = []
    for dim in ("2D", "3D", "IMQ"):
        rows = [g for g in groups if g.dim == dim]
        out += [DIM_TITLES[dim].replace("features", "groups"), "-" * len(DIM_TITLES[dim].replace("features", "groups")), "",
                ".. list-table::", "   :header-rows: 1", "   :widths: 26 10 64", "",
                "   * - Group", "     - Features", "     - Families"]
        for g in rows:
            if not g.expanded:
                unusable.append(g)
                continue
            m = reg.members(g, features)
            fams = []
            for f in m:
                fam = family_of(f)
                if fam not in fams:
                    fams.append(fam)
            links = ", ".join(f":ref:`{titles[s]} <features-{s.replace('_', '-')}>`" for s in fams)
            out += [f"   * - ``{g.name}``", f"     - {len(m)}", f"     - {links}"]
        out.append("")
    if unusable:
        names = ", ".join(f"``{g.name}``" for g in unusable)
        out += [".. note::", "",
                f"   {names} are accepted as group names but no features are defined for them, so a",
                "   request that uses them stops with an error.", ""]
    return "\n".join(out).rstrip("\n") + "\n"


def render_family_index(features, slug, extras, valid):
    """The feature index table of a family's definition page."""
    out = [".. Generated by docs/tools/gen_feature_reference.py from docs/feature_extras.toml and",
           "   tests/vetting/oracle_coverage.csv; do not edit by hand.",
           "",
           ".. list-table::",
           "   :header-rows: 1",
           "   :widths: 24 34 8 26 8",
           "",
           "   * - Code",
           "     - Name",
           "     - IBSI",
           "     - Validation",
           "     - Example"]
    for f in base_codes(features, slug):
        e = extras[f.code]
        status, oracles, tols = valid.get((f.dim, f.code), ("untested", [], {}))
        if status == "vetted":
            # each oracle with its own tolerances, so a row with several oracles pairs them correctly
            parts = []
            for o in oracles:
                ts = ", ".join(t.replace("rel=", "rel. ").replace("abs=", "abs. ") for t in tols.get(o, []))
                parts.append(ORACLE_NAMES.get(o, o) + (f" ({ts})" if ts else ""))
            v = ", ".join(parts)
        else:
            v = status
        out += [f"   * - :ref:`{f.code} <{label(f.code)}>`",
                f"     - {escape(e['name'])}",
                f"     - {e['ibsi'] or '—'}",
                f"     - {v}",
                "     - " + f"{e['example']:g}".replace("-", "−")]
    return "\n".join(out) + "\n"


def check_family_pages(features, extras):
    problems = []
    for slug, cfg in FAMILY_PAGES.items():
        page = REPO / "docs" / "source" / (cfg["page"] + ".rst")
        text = page.read_text(encoding="utf-8") if page.exists() else ""
        codes = [f.code for f in base_codes(features, slug)]
        fam_extras = extras.get(slug, {})
        for c in codes:
            if c not in fam_extras:
                problems.append(f"{c}: no entry in [{slug}] of {EXTRAS.name}")
            if f".. _{label(c)}:" not in text:
                problems.append(f"{c}: {cfg['page']}.rst has no entry '.. _{label(c)}:'")
        for c in fam_extras:
            if c not in codes:
                problems.append(f"[{slug}] {c} in {EXTRAS.name} is not a base code of that family")
    return problems


def check_descriptions(features, desc):
    problems = []
    codes = {(family_of(f), f.code) for f in features}
    for f in features:
        fam = family_of(f)
        if fam is None:
            problems.append(f"{f.dim} {f.code}: no family in FAMILIES claims class {f.feature_class or '(none)'}")
        elif not " ".join(desc.get(fam, {}).get(f.code, "").split()):
            problems.append(f"{f.code}: no entry in [{fam}] of {DESCRIPTIONS.name}")
    for fam, entries in desc.items():
        for code in entries:
            if (fam, code) not in codes:
                problems.append(f"[{fam}] {code} in {DESCRIPTIONS.name} is not a feature code of that family")
    return problems


COUNT_PAGES = [REPO / "README.md", REPO / "docs" / "source" / "index.rst", REPO / "CLAUDE.md"]


def check_counts(features):
    # README.md, the landing page and CLAUDE.md state the feature counts in prose; keep them equal
    # to the registry.
    n = {d: sum(1 for f in features if f.dim == d) for d in ("2D", "3D", "IMQ")}
    sentence = (f"{sum(n.values())} features in all, {n['2D']} for 2D images, {n['3D']} for 3D volumes "
                f"and {n['IMQ']} image quality features")
    return [f"{p.relative_to(REPO)} does not say \"{sentence}\"" for p in COUNT_PAGES
            if sentence not in " ".join(p.read_text(encoding="utf-8").split())]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify instead of writing")
    args = ap.parse_args(argv)

    features, groups = reg.load()
    with open(DESCRIPTIONS, "rb") as f:
        data = tomllib.load(f)
    notes = data.pop("family_notes", {})
    desc = data
    with open(EXTRAS, "rb") as f:
        extras = tomllib.load(f)
    problems = check_descriptions(features, desc) + check_family_pages(features, extras) + check_counts(features)
    if any(family_of(f) is None for f in features):
        # the pages cannot be rendered without a family for every code
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return 1
    pages = {LIST_PAGE: render_list(features, desc, notes, validation()),
             GROUPS_PAGE: render_groups(features, groups)}
    valid_t = validation(with_tolerance=True)
    for slug, cfg in FAMILY_PAGES.items():
        if all(f.code in extras.get(slug, {}) for f in base_codes(features, slug)):
            pages[REPO / "docs" / "source" / cfg["index"]] = render_family_index(features, slug, extras.get(slug, {}), valid_t)

    if args.check:
        for page, text in pages.items():
            if not page.exists() or page.read_text(encoding="utf-8") != text:
                problems.append(f"{page.relative_to(REPO)} is stale; run docs/tools/gen_feature_reference.py")
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if problems:
        return 1
    if not args.check:
        for page, text in pages.items():
            page.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {page.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
