"""Read Nyxus' feature codes, families and groups out of the C++ sources. Stdlib only.

The documentation's feature pages are generated from what this module returns, so it reads the
same declarations the program uses:

  codes     the Feature2D / Feature3D / FeatureIMQ enums in src/nyx/featureset.h, with the
            user-facing names from the maps in src/nyx/featureset.cpp
  families  the feature class whose `featureset` lists a code (src/nyx/features/*.h)
  groups    the group names in featureset.cpp and the featuresets each one enables in
            Environment::expand_2D/3D/IMQ_featuregroup (src/nyx/env_features.cpp)

Code inside `#if 0` blocks and comments is ignored.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "nyx"

ENUMS = {"2D": "Feature2D", "3D": "Feature3D", "IMQ": "FeatureIMQ"}
GROUP_ENUMS = {"2D": "Fgroup2D", "3D": "Fgroup3D", "IMQ": "FgroupIMQ"}
EXPANDERS = {"2D": "expand_2D_featuregroup", "3D": "expand_3D_featuregroup", "IMQ": "expand_IMQ_featuregroup"}


@dataclass
class Feature:
    dim: str            # "2D", "3D" or "IMQ"
    code: str           # user-facing name, e.g. GLCM_ASM
    enum: str           # enumerator, normally equal to code
    feature_class: str  # class whose featureset lists it; "" if none does


@dataclass
class Group:
    dim: str
    name: str                     # e.g. *ALL_GLCM*
    enum: str                     # e.g. FG2_GLCM
    classes: list = field(default_factory=list)   # feature classes it enables
    everything: bool = False      # enables every code of its dimension
    expanded: bool = True         # False: the name is accepted but no expander handles it


def strip_cpp(text):
    """Drop comments and `#if 0 ... #endif` blocks (nesting-aware)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    out, depth = [], 0
    for line in text.splitlines():
        s = line.strip()
        if depth:
            if s.startswith("#if"):
                depth += 1
            elif s.startswith("#endif"):
                depth -= 1
            continue
        if re.match(r"#\s*if\s+0\b", s):
            depth = 1
            continue
        out.append(line)
    return "\n".join(out)


def _block(text, start):
    """The text between the first '{' at or after `start` and its matching '}'."""
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    raise ValueError("unbalanced braces")


def enumerators(text, enum_name):
    m = re.search(r"enum\s+class\s+" + enum_name + r"\b", text)
    body = _block(text, m.end())
    names = [re.match(r"\s*(\w+)", item).group(1) for item in body.split(",") if item.strip()]
    return [n for n in names if not (n.startswith("_") and n.endswith("_"))]


def name_maps(text, enum_name):
    """{user-facing name: enumerator} for every map entry naming an `enum_name` value.

    A name listed twice keeps its first entry, as std::map's initializer-list constructor does."""
    result = {}
    for name, e in re.findall(r'\{\s*"([^"]+)"\s*,\s*(?:Nyxus::)?' + enum_name + r"::(\w+)\s*\}", text):
        result.setdefault(name, e)
    return result


def class_featuresets():
    """{class name: (enum name, [enumerators])} from every feature header."""
    result = {}
    for header in sorted((SRC / "features").glob("*.h")):
        text = strip_cpp(header.read_text(encoding="utf-8", errors="replace"))
        for m in re.finditer(r"initializer_list\s*<\s*Nyxus::(\w+)\s*>\s*featureset\s*=", text):
            cls = re.findall(r"\b(?:class|struct)\s+(\w+)\s*(?::|\{)", text[:m.start()])
            body = _block(text, m.end())
            result[cls[-1]] = (m.group(1), re.findall(r"Nyxus::\w+::(\w+)", body))
    return result


def load():
    """-> (features, groups): every code and every group, in declaration order."""
    fs_h = strip_cpp((SRC / "featureset.h").read_text(encoding="utf-8"))
    fs_cpp = strip_cpp((SRC / "featureset.cpp").read_text(encoding="utf-8"))
    env = strip_cpp((SRC / "env_features.cpp").read_text(encoding="utf-8"))
    classes = class_featuresets()

    features = []
    for dim, enum_name in ENUMS.items():
        names = {v: k for k, v in reversed(list(name_maps(fs_cpp, enum_name).items()))}  # first entry wins, as in std::map
        owner = {}
        for cls, (e, codes) in classes.items():
            if e == enum_name:
                for c in codes:
                    owner.setdefault(c, cls)
        for e in enumerators(fs_h, enum_name):
            features.append(Feature(dim, names.get(e, e), e, owner.get(e, "")))

    groups = []
    for dim, enum_name in GROUP_ENUMS.items():
        expander = env[env.index("Environment::" + EXPANDERS[dim]):]
        expander = _block(expander, 0)
        for name, e in name_maps(fs_cpp, enum_name).items():
            g = Group(dim, name, e)
            if dim == "IMQ":
                g.everything = True
            else:
                m = re.search(r"\(\s*" + enum_name + r"\s*\)\s*fgcode\s*==\s*" + enum_name + "::" + e + r"\b", expander)
                if not m:
                    g.expanded = False
                else:
                    body = _block(expander, m.end())
                    g.classes = re.findall(r"enableFeatures\s*\(\s*(\w+)::featureset", body)
                    g.everything = "_FIRST_" in body and "_COUNT_" in body
            groups.append(g)
    return features, groups


def members(group, features):
    """Codes a group enables, in declaration order."""
    if not group.expanded:
        return []
    return [f for f in features if f.dim == group.dim and (group.everything or f.feature_class in group.classes)]
