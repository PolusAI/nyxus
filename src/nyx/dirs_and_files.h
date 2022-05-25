#pragma once

//
// Helper functions for manipulating directories and files
//

#include <string>
#include <vector>

namespace Nyxus
{
	/// @brief Returns the pure file name
	/// @param fpath 
	/// @return 
	std::string getPureFname(std::string fpath);

	/// @brief Checks if a directory exists
	/// @param dir 
	/// @return 
	bool directoryExists(const std::string& dir);
	
	/// @brief Reads all the files in a directory with respect to a file pattern
	/// @param dir 
	/// @param file_pattern 
	/// @param files 
	void readDirectoryFiles(const std::string& dir, const std::string& file_pattern, std::vector<std::string>& files);
	
	/// @brief Nyxus specific: checks if the intensity, mask, and output directories exist
	/// @param dirIntens 
	/// @param dirLabels 
	/// @param dirOut 
	/// @param mustCheckDirOut 
	/// @return 
	int datasetDirsOK(const std::string& dirIntens, const std::string& dirLabels, const std::string& dirOut, bool mustCheckDirOut);
	
	int read_dataset(
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
		std::vector <std::string>& labelFiles);

} // namespace Nyxus