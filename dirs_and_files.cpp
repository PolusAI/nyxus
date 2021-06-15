#include <string>
#include <filesystem>
#include <vector>
#include <iostream>

bool directoryExists(std::string dir)
{
	std::filesystem::path p(dir);
	return std::filesystem::exists(p);
}

void readDirectoryFiles(std::string dir, std::vector<std::string> & files)
{
	for (auto& entry : std::filesystem::directory_iterator(dir))
		files.push_back (entry.path().string());
}

