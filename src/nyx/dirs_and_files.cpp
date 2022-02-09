//
// Helper functions for manipulating directories and files
//

#include <fstream>
#include <string>
#include <filesystem>
#include <vector>
#include <iostream>
#include <regex>
#include <sstream>
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
			std::string fullPath = entry.path().string(), 
				pureFname = entry.path().filename().string();	// The file name that should participate in the filepattern check
			if (std::regex_match(pureFname, re))
				files.push_back(fullPath);
			// else
			// 	std::cout << "Skipping file " << fp << " as not matching file pattern " << theEnvironment.file_pattern << "\n";
		}
	}

	int datasetDirsOK(const std::string& dirIntens, const std::string& dirLabels, const std::string& dirOut, bool mustCheckDirOut)
	{
		if (!directoryExists(dirIntens))
		{
			std::stringstream ss;
			ss << "Error: " << dirIntens << " is not a directory";
			#ifdef WITH_PYTHON_H
				throw ss.str().c_str();
			#endif
			std::cerr << ss.str() << std::endl;
			return 1;
		}

		if (!directoryExists(dirLabels))
		{
			std::stringstream ss;
			ss << "Error: " << dirLabels << " is not a directory";
			#ifdef WITH_PYTHON_H
				throw ss.str().c_str();
			#endif
			std::cerr << ss.str() << std::endl;
			return 2;
		}

		if (mustCheckDirOut && !directoryExists(dirOut))
		{
			std::stringstream ss;
			ss << "Error: " << dirOut << " is not a directory";
			#ifdef WITH_PYTHON_H
				throw ss.str().c_str();
			#endif
			std::cerr << ss.str() << std::endl;
			return 3;
		}
		return 0; // success
	}

	int read_dataset (
		// input
		const std::string& dirIntens, 
		const std::string& dirLabels, 
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

		if (!directoryExists(dirIntens))
		{
			std::stringstream ss;
			ss << "Error: nonexisting directory " << dirIntens;
			#ifdef WITH_PYTHON_H
				throw ss.str().c_str();
			#endif
			std::cerr << ss.str() << std::endl;
			return 1;
		}
		if (!directoryExists(dirLabels))
		{
			std::stringstream ss;
			ss << "Error: nonexisting directory " << dirLabels;
			#ifdef WITH_PYTHON_H
			throw ss.str().c_str();
			#endif
			std::cerr << ss.str() << std::endl;
			return 1;
		}
		if (!directoryExists(dirOut))
		{
			std::stringstream ss;
			ss << "Error: nonexisting directory " << dirOut;
			#ifdef WITH_PYTHON_H
				throw ss.str().c_str();
			#endif
			std::cerr << ss.str() << std::endl;
			return 1;
		}

		if (intLabMappingFile.empty())
		{
			// Common case - no ad hoc intensity-label file mapping, 1-to-1 correspondence instead
			readDirectoryFiles(dirIntens, intensFiles);
			readDirectoryFiles(dirLabels, labelFiles);

			// Check if the dataset is meaningful
			if (intensFiles.size() == 0 || labelFiles.size() == 0)
			{
				#ifdef WITH_PYTHON_H
					throw "Error: no intensity and/or label files to process";
				#endif
				std::cerr << "Error: no intensity and/or label files to process \n";
				return 2;
			}
			if (intensFiles.size() != labelFiles.size())
			{
				std::stringstream ss;
				ss << "Error: the number of intensity directory files (" << intensFiles.size() << ") should match the number of label directory files (" << labelFiles.size() << ")";
				#ifdef WITH_PYTHON_H
					throw ss.str().c_str();
				#endif
				std::cerr << ss.str() << std::endl;

				return 3;
			}

			// Sort the files to produce an intuitive sequence
			std::sort(intensFiles.begin(), intensFiles.end());
			std::sort(labelFiles.begin(), labelFiles.end());
		}
		else
		{
			// Special case - using intensity and label file pairs defined with the mapping file
			if (!directoryExists(intLabMappingDir))
			{
				std::stringstream ss;
				ss << "Error: nonexisting directory " << intLabMappingDir;
				#ifdef WITH_PYTHON_H
					throw ss.str().c_str();
				#endif
				std::cerr << ss.str() << std::endl;
				return 1;
			}

			std::string mapPath = intLabMappingDir + "/" + intLabMappingFile;
			if (!directoryExists(mapPath))
			{
				std::stringstream ss;
				ss << "Error: nonexisting file " << mapPath;
				#ifdef WITH_PYTHON_H
					throw ss.str().c_str();
				#endif
				std::cerr << ss.str() << std::endl;
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
					std::stringstream ss;
					ss << "Error: cannot recognize a file name pair in line #" << lineNo << " - " << ln;
					#ifdef WITH_PYTHON_H
						throw ss.str().c_str();
					#endif
					std::cerr << ss.str() << std::endl;
					return 1;
				}

				// We have a pair of file names. Let's check if they exist
				lineNo++;
				std::string intFpath = dirIntens + "/" + intFname;
				if (!directoryExists(intFpath))
				{
					std::stringstream ss;
					ss << "Error: nonexisting file " << intFpath;
					#ifdef WITH_PYTHON_H
						throw ss.str().c_str();
					#endif
					std::cerr << ss.str() << std::endl;
					return 1;
				}

				std::string segFpath = dirLabels + "/" + segFname;
				if (!directoryExists(intFpath))
				{
					std::stringstream ss;
					ss << "Error: nonexisting file " << intFpath;
					#ifdef WITH_PYTHON_H
						throw ss.str().c_str();
					#endif
					std::cerr << ss.str() << std::endl;
					return 1;
				}

				// Save the file pair
				intensFiles.push_back (intFpath);
				labelFiles.push_back (segFpath);
			}

			// Check if we have pairs to process
			if (intensFiles.size() == 0)
			{
				std::stringstream ss;
				ss << "Error: special mapping " << mapPath << " produced no intensity-label file pairs";
				#ifdef WITH_PYTHON_H
					throw ss.str().c_str();
				#endif
				std::cerr << ss.str() << std::endl;
				return 1;
			}

			// Inform the user
			std::cout << "Error: using special mapped intensity-label pairs:" << std::endl;
			for (int i = 0; i < intensFiles.size(); i++)
				std::cout << "\tintensity: " << intensFiles[i] << "\tlabels: " << labelFiles[i] << std::endl;
		}

		return 0; // success
	}

	std::string getPureFname(std::string fpath)
	{
		std::filesystem::path p(fpath);
		return p.filename().string();
	}

} // namespace Nyxus