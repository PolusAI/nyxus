#pragma once

//
// Helper functions for manipulating directories and files
//

#include <map>
#include <string>
#include <vector>
#include "strpat.h"

namespace Nyxus
{
	/// @brief Returns the pure file name
	/// @param fpath File name possible having the directory part
	/// @return File name with extension
	std::string getPureFname(const std::string& fpath);

	/// @brief Checks if a directory exists
	/// @param dir 
	/// @return 
	bool existsOnFilesystem(const std::string& dir);
	
	/// @brief Reads all the files in a directory with respect to a file pattern
	/// @return 0 if success, otherwise an error code (error details in 'error_message')
	/// @param dir 
	/// @param file_pattern 
	/// @param files 
	/// 

	int read_2D_dataset(
		// input:
		const std::string& dirIntens,
		const std::string& dirLabels,
		const std::string& filePatt,
		const std::string& dirOut,
		const std::string& intLabMappingDir,
		const std::string& intLabMappingFile,
		bool mustCheckDirOut,
		// output:
		std::vector <std::string>& intensFiles,
		std::vector <std::string>& labelFiles,
		std::string & error_message);

	struct Imgfile3D_layoutA
	{
	public:
		
		Imgfile3D_layoutA (const std::string& possibly_full_path);
		Imgfile3D_layoutA();

		std::string fname, fdir;
		std::vector<std::string> z_indices;
	};
		
	int read_3D_dataset(
		// input:
		const std::string& dirIntens,
		const std::string& dirLabels,
		const StringPattern& filePatt,
		const std::string& dirOut,
		const std::string& intLabMappingDir,
		const std::string& intLabMappingFile,
		bool mustCheckDirOut,
		// output:
		std::vector <Imgfile3D_layoutA>& intensFiles,
		std::vector <Imgfile3D_layoutA>& labelFiles);

	int read_3D_dataset_wholevolume(
		// input:
		const std::string& dirIntens,
		const StringPattern& filePatt,
		const std::string& dirOut,
		// output:
		std::vector <std::string>& intensFiles);

	/// @brief checks if the Tiff file is tiled or not
	/// @param filePath File name with complete path
	bool check_tile_status(const std::string& filePath);
} // namespace Nyxus