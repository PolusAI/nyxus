//
// Helper functions for manipulating directories and files
//

#include <fstream>
#include <string>
#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif
#include <vector>
#include <iostream>
#include <regex>
#include <sstream>
#include <tiffio.h>

namespace Nyxus
{
	bool existsOnFilesystem(const std::string& dir)
	{
		fs::path p(dir);
		return fs::exists(p);
	}

	void readDirectoryFiles(const std::string& dir, const std::string& file_pattern, std::vector<std::string>& files)
	{
		std::regex re(file_pattern);

		for (auto& entry : fs::directory_iterator(dir))
		{
			// Skip hidden objects, e.g. directories '.DS_store' in OSX
			if (entry.path().filename().string()[0] == '.')
			{
				std::cout << "Skipping " << entry.path().filename().string() << "\n";
				continue;
			}

			std::string fullPath = entry.path().string(),
				pureFname = entry.path().filename().string();

			if (std::regex_match(pureFname, re))
				files.push_back(fullPath);
		}
	}

	int datasetDirsOK(const std::string& dirIntens, const std::string& dirLabels, const std::string& dirOut, bool mustCheckDirOut)
	{
		if (!existsOnFilesystem(dirIntens))
		{
			std::cout << "Error: " << dirIntens << " is not a directory" << std::endl;
			return 1;
		}

		if (!existsOnFilesystem(dirLabels))
		{
			std::cout << "Error: " << dirLabels << " is not a directory" << std::endl;
			return 2;
		}

		if (mustCheckDirOut && !existsOnFilesystem(dirOut))
		{
			std::cout << "Error: " << dirOut << " is not a directory" << std::endl;
			return 3;
		}
		return 0; // success
	}

	int read_dataset (
		// input
		const std::string& dirIntens,
		const std::string& dirLabels,
		const std::string& filePatt,
		const std::string& dirOut,
		const std::string& intLabMappingDir,
		const std::string& intLabMappingFile,
		bool mustCheckDirOut,
		// output
		std::vector <std::string>& intensFiles,
		std::vector <std::string>& labelFiles)
	{
		// Check the directories
		if (datasetDirsOK(dirIntens, dirLabels, dirOut, mustCheckDirOut) != 0)
			return 1;	// No need to issue console messages here, datasetDirsOK() does that

		if (!existsOnFilesystem(dirIntens))
		{
			std::cout << "Error: nonexisting directory " << dirIntens << std::endl;
			return 1;
		}
		if (!existsOnFilesystem(dirLabels))
		{
			std::cout << "Error: nonexisting directory " << dirLabels << std::endl;
			return 1;
		}
		if (!existsOnFilesystem(dirOut))
		{
			std::cout << "Error: nonexisting directory " << dirOut << std::endl;
			return 1;
		}

		if (intLabMappingFile.empty())
		{
			// Common case - no ad hoc intensity-label file mapping, 1-to-1 correspondence instead
			readDirectoryFiles(dirIntens, filePatt, intensFiles);
			readDirectoryFiles(dirLabels, filePatt, labelFiles);

			// Check if the dataset is meaningful
			if (intensFiles.size() == 0 || labelFiles.size() == 0)
			{
				std::cout << "No intensity and/or label files to process, probably due to file pattern " << filePatt << std::endl;
				return 2;
			}
			if (intensFiles.size() != labelFiles.size())
			{
				std::cout << "Mismatch: " << intensFiles.size() << " intensity images vs " << labelFiles.size() << " mask images\n";
				return 3;
			}

			// Sort the files to produce an intuitive sequence
			std::sort(intensFiles.begin(), intensFiles.end());
			std::sort(labelFiles.begin(), labelFiles.end());
		}
		else
		{
			// Special case - using intensity and label file pairs defined with the mapping file
			if (!existsOnFilesystem(intLabMappingDir))
			{
				std::cout << "Error: nonexisting directory " << intLabMappingDir << std::endl;
				return 1;
			}

			std::string mapPath = intLabMappingDir + "/" + intLabMappingFile;
			if (!existsOnFilesystem(mapPath))
			{
				std::cout << "Error: nonexisting file " << mapPath << std::endl;
				return 1;
			}

			// Read
			std::ifstream file(mapPath);
			std::string ln, intFile, segFile;
			int lineNo = 1;
			while (std::getline(file, ln))
			{
				std::stringstream ss(ln);
				std::string intFname, segFname;
				bool pairOk = ss >> intFname && ss >> segFname;
				if (!pairOk)
				{
					std::cout << "Error: cannot recognize a file name pair in line #" << lineNo << " - " << ln << std::endl;
					return 1;
				}

				// We have a pair of file names. Let's check if they exist
				lineNo++;
				std::string intFpath = dirIntens + "/" + intFname;
				if (!existsOnFilesystem(intFpath))
				{
					std::cout << "Error: nonexisting file " << intFpath << std::endl;
					return 1;
				}

				std::string segFpath = dirLabels + "/" + segFname;
				if (!existsOnFilesystem(intFpath))
				{
					std::cout << "Error: nonexisting file " << intFpath << std::endl;
					return 1;
				}

				// Save the file pair
				intensFiles.push_back (intFpath);
				labelFiles.push_back (segFpath);
			}

			// Check if we have pairs to process
			if (intensFiles.size() == 0)
			{
				std::cout << "Special mapping " << mapPath << " produced no intensity-label file pairs" << std::endl;
				return 1;
			}

			// Inform the user
			std::cout << "Using special mapped intensity-label pairs:" << std::endl;
			for (int i = 0; i < intensFiles.size(); i++)
				std::cout << "\tintensity: " << intensFiles[i] << "\tlabels: " << labelFiles[i] << std::endl;
		}

		return 0; // success
	}

	std::string getPureFname(const std::string& fpath)
	{
		fs::path p(fpath);
		return p.filename().string();
	}

	// Helper function to determine Tile Status
	bool check_tile_status(const std::string& filePath)
	{
		TIFF* tiff_ = TIFFOpen(filePath.c_str(), "r");
		if (tiff_ != nullptr)
		{
			if (TIFFIsTiled(tiff_) == 0)
			{
				TIFFClose(tiff_);
				return false;
			}
			else
			{
				TIFFClose(tiff_);
				return true;
			}
		}
		else { throw (std::runtime_error("Tile Loader ERROR: The file can not be opened.")); }
	}

} // namespace Nyxus
