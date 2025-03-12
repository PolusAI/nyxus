#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include <tuple>
#include "globals.h"
#include "environment.h"
#include "features/radial_distribution.h"
#include "features/gabor.h"
#include "features/glcm.h"
#include "features/glrlm.h"
#include "features/zernike.h"
#include "helpers/fsystem.h"

namespace Nyxus
{
	static std::mutex mx1;

	std::tuple<bool, std::optional<std::string>> save_features_2_apache_wholeslide (const LR & wsi_roi, const std::string & wsi_path)
	{
		std::lock_guard<std::mutex> lg (mx1);
		return theEnvironment.arrow_stream.write_arrow_file (Nyxus::get_feature_values_roi (wsi_roi, wsi_path, ""));
	}
}