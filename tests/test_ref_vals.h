#pragma once

#include <string>
#include <unordered_map>
#include <vector>

// Golden reference tables (SPEC 6.3.1).
//
// Declaring a table through one of these aliases rather than a bare std::unordered_map makes it
// identifiable by its TYPE. That matters for enforcement: check_test_names.py used to recognise a
// reference table by looking for "golden" / "oracle" / "ref_vals" in its name, which is circular --
// a table called glcm_values matched nothing and was invisible to the naming rule. A declaration
// spelled ref_vals_map<double> is a reference table whatever it is called, so the 6.3.1 name check
// applies to the complete set instead of a self-selected one.
//
// The value type is the one the assertion compares. agrees_gt() takes double, so double is the
// default; float appears only where a table predates that and is worth revisiting rather than
// copying.

// Keyed by feature name -- the common shape.
//   ref_vals_map<double> glcm_2d_ibsi_ref_vals { {"GLCM_ASM", 0.368}, ... };
template <typename T>
using ref_vals_map = std::unordered_map<std::string, T>;

// Keyed by ROI label, each label holding its own per-feature table.
//   ref_vals_map_by_label<double> neighbor_2d_analytic_ref_vals_by_label { {1, {...}}, ... };
template <typename T>
using ref_vals_map_by_label = std::unordered_map<int, ref_vals_map<T>>;

// An ordered list of records rather than a keyed lookup: the geometric-moment tables carry
// {enum, name, value} triples and are iterated in order, so there is no string key to map from.
template <typename T>
using ref_vals_list = std::vector<T>;
