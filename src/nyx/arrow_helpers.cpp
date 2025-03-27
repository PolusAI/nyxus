#include <string>
#include "environment.h"
#include "helpers/fsystem.h"
#include "helpers/helpers.h"

namespace Nyxus
{

	std::string get_arrow_filename(const std::string& output_path, const std::string& default_filename, const SaveOption& arrow_file_type) {

		/*
					output_path			condition			verdict
		Case 1: 	/foo/bar		exist in fs				is a directory, append default filename with proper ext
					/foo/bar/		or ends with / or \
					\foo\bar\

		Case 2:		/foo/bar		does not exist in fs	assume the extension is missing, append proper ext
									but /foo exists

		Case 3: 	/foo/bar		neither /foo nor 		treat as directory, append default filename with proper ext
									/foo/bar exists in fs

		Case 4: 	/foo/bar.ext	exists in fs and is a 	append default filename with proper ext
									directory

		Case 5: 	/foo/bar.ext	does not exist in fs  	this is a file, check if ext is correct and modify if needed

		Case 6:		empty									default filename with proper ext


		*/
		std::string valid_ext = [&arrow_file_type]() {
			if (arrow_file_type == Nyxus::SaveOption::saveArrowIPC) {
				return ".arrow";
			}
			else if (arrow_file_type == Nyxus::SaveOption::saveParquet) {
				return ".parquet";
			}
			else { return ""; }
		}();

		if (output_path != "") {
			auto arrow_path = fs::path(output_path);
			if (fs::is_directory(arrow_path) // case 1, 4
				|| Nyxus::ends_with_substr(output_path, "/")
				|| Nyxus::ends_with_substr(output_path, "\\")) {
				arrow_path = arrow_path / default_filename;
			}
			else if (!arrow_path.has_extension()) {
				if (!fs::is_directory(arrow_path.parent_path())) { // case 3
					arrow_path = arrow_path / default_filename;
				}
				// else case 2, do nothing here	
			}
			// case 5 here, but also for 1-4, update extenstion here
			arrow_path.replace_extension(valid_ext);
			return arrow_path.string();
		}
		else { // case 6
			return default_filename + valid_ext;
		}
	}

}

