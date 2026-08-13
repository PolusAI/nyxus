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
// The value type is the one the assertion compares, and it is double throughout: agrees_gt() and
// ASSERT_NEAR both take double, so a float table only round-tripped its literals through a narrower
// type before widening them again at the comparison. The five 3D texture tables that predated the
// aliases were the last float ones and are now double like the rest.
//
// Declaring the table through an alias is itself the rule, not a style preference -- a bare
// std::unordered_map is invisible to the type-based detection above, so check_test_names.py rejects
// a reference table spelled that way.
//
// Declare every table const, and read it with .at(). On a non-const map operator[] compiles, and a
// key the table does not hold is default-inserted as 0 rather than failing: the assertion then
// compares against a golden that does not exist, and passes whenever the computed value is also 0 --
// a feature that silently produced nothing, which is the case the table is here to catch. The
// inserted key then persists, so a later .count() guard on it succeeds too. const turns operator[]
// into a compile error and .at() throws naming the missing key. check_test_names.py rejects a
// reference table declared without const.

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
