#!/bin/bash
# Validate that all required prereq libraries (Arrow, Z5, DICOM) were installed.
# Mirrors the find_package / find_file checks in CMakeLists.txt.
# Usage: bash ci-utils/validate_prereqs.sh [local_install_dir]

set -uo pipefail

LOCAL_INSTALL="${1:-local_install}"
errors=0

check() {
  local label="$1"
  shift
  # Try each pattern; succeed if any match
  for pattern in "$@"; do
    if ls $pattern >/dev/null 2>&1; then
      echo "  OK: $label"
      return 0
    fi
  done
  echo "  FAIL: $label (looked for: $*)"
  errors=$((errors + 1))
}

echo "=== Validating prereq installation in $LOCAL_INSTALL ==="

echo ""
echo "-- Z5 support (CMakeLists.txt: find_package ZLIB, BLOSC, Boost, nlohmann_json; find_file z5/z5.hxx) --"
check "Z5 header"             "$LOCAL_INSTALL/include/z5/z5.hxx"
check "Boost headers"         "$LOCAL_INSTALL/include/boost/version.hpp"
check "nlohmann_json headers" "$LOCAL_INSTALL/include/nlohmann/json.hpp"
check "xtensor headers"       "$LOCAL_INSTALL/include/xtensor/xtensor.hpp"
check "blosc header"          "$LOCAL_INSTALL/include/blosc.h"

echo ""
echo "-- Arrow support (CMakeLists.txt: find_package Arrow, Parquet) --"
check "Arrow CMake config"    "$LOCAL_INSTALL/lib/cmake/Arrow/ArrowConfig.cmake" \
                              "$LOCAL_INSTALL/lib64/cmake/Arrow/ArrowConfig.cmake"
check "Parquet CMake config"  "$LOCAL_INSTALL/lib/cmake/Parquet/ParquetConfig.cmake" \
                              "$LOCAL_INSTALL/lib64/cmake/Parquet/ParquetConfig.cmake"

echo ""
echo "-- DICOM support (CMakeLists.txt: find_package DCMTK, fmjpeg2k) --"
check "DCMTK headers"         "$LOCAL_INSTALL/include/dcmtk/dcmdata/dcfilefo.h"
check "fmjpeg2k CMake config" "$LOCAL_INSTALL/lib/cmake/fmjpeg2k/fmjpeg2kConfig.cmake" \
                              "$LOCAL_INSTALL/lib64/cmake/fmjpeg2k/fmjpeg2kConfig.cmake"

echo ""
if [ $errors -gt 0 ]; then
  echo "=== FAILED: $errors prereq check(s) did not pass ==="
  exit 1
fi
echo "=== All prereq checks passed ==="
