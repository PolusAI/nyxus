#include <string>
#include <filesystem>
#include <vector>
#include <iostream>

bool directoryExists(const std::string & dir)
{
	std::filesystem::path p(dir);
	return std::filesystem::exists(p);
}

void readDirectoryFiles(const std::string & dir, std::vector<std::string> & files)
{
	for (auto& entry : std::filesystem::directory_iterator(dir))
		files.push_back (entry.path().string());
}

int datasetDirsOK (std::string & dirIntens, std::string & dirLabels, std::string & dirOut)
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

	if (!directoryExists(dirOut))
	{
		std::cout << "Error: " << dirOut << " is not a directory" << std::endl;
		return 3;
	}
	return 0; // success
}

int checkAndReadDataset(
	// input
	std::string& dirIntens, std::string& dirLabels, std::string& dirOut,
	// output
	std::vector <std::string>& intensFiles, std::vector <std::string>& labelFiles)
{
	// Check the directories
	if (datasetDirsOK(dirIntens, dirLabels, dirOut) != 0)
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




