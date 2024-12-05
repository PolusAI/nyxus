#ifdef __APPLE__
#define uint64 uint64_hack_
#define int64 int64_hack_
#include <tiffio.h>
#undef uint64
#undef int64
#else
#include <tiffio.h>
#endif
#include <cstring>
#include <sstream>
#include <limits.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "raw_tiff.h"
#include "raw_omezarr.h"
#include "raw_dicom.h"


