//
// Helper functions for manipulating directories and files
//

#include <string>
#include <filesystem>
#include <vector>
#include <iostream>
#include <regex>
#include "environment.h"

namespace Nyxus
{
	bool directoryExists(const std::string& dir)
	{
		std::filesystem::path p(dir);
		return std::filesystem::exists(p);
	}

	void readDirectoryFiles(const std::string& dir, std::vector<std::string>& files)
	{
		std::regex re(theEnvironment.file_pattern);

		for (auto& entry : std::filesystem::directory_iterator(dir))
		{
			std::string fp = entry.path().string();
			if (std::regex_match(fp, re))
				files.push_back(fp);
			else
				std::cout << "Skipping file " << fp << " as not matching file pattern " << theEnvironment.file_pattern << "\n";
		}
	}

	int datasetDirsOK(const std::string& dirIntens, const std::string& dirLabels, const std::string& dirOut, bool mustCheckDirOut)
	{
		if (!directoryExists(dirIntens))
		{
			std::cout << "Error: " << dirIntens << " is not a directory" << std::endl;
			return 1;
		}

		if (!directoryExists(dirLabels))
		{
			std::cout << "Error: " << dirLabels << " is not a directory" << std::endl;
			return 2;
		}

		if (mustCheckDirOut && !directoryExists(dirOut))
		{
			std::cout << "Error: " << dirOut << " is not a directory" << std::endl;
			return 3;
		}
		return 0; // success
	}

	int checkAndReadDataset(
		// input
		const std::string& dirIntens, const std::string& dirLabels, const std::string& dirOut, bool mustCheckDirOut,
		// output
		std::vector <std::string>& intensFiles, std::vector <std::string>& labelFiles)
	{
		// Check the directories
		if (datasetDirsOK(dirIntens, dirLabels, dirOut, mustCheckDirOut) != 0)
			return 1;	// No need to issue console messages here, datasetDirsOK() does that

		readDirectoryFiles(dirIntens, intensFiles);
		readDirectoryFiles(dirLabels, labelFiles);

		// Check if the dataset is meaningful
		if (intensFiles.size() == 0 || labelFiles.size() == 0)
		{
			std::cout << "No intensity and/or label files to process" << std::endl;
			return 2;
		}
		if (intensFiles.size() != labelFiles.size())
		{
			std::cout << "The number of intensity directory files (" << intensFiles.size() << ") should match the number of label directory files (" << labelFiles.size() << ")" << std::endl;
			return 3;
		}

		return 0; // success
	}

	std::string getPureFname(std::string fpath)
	{
		std::filesystem::path p(fpath);
		return p.filename().string();
	}

} // namespace Nyxus