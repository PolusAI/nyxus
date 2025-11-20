//
// Helper functions for manipulating directories and files
//

#include <fstream>
#include <string>
#include "helpers/fsystem.h"

#include <vector>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <tiffio.h>
#include "dirs_and_files.h"
#include "strpat.h"
#include "helpers/helpers.h"

namespace Nyxus
{
	bool existsOnFilesystem(const std::string& dir)
	{
		fs::path p(dir);
		return fs::exists(p);
	}

	void readDirectoryFiles_2D (const std::string& dir, const std::string& file_pattern, std::vector<std::string>& fullpaths, std::vector<std::string>& purefnames)
	{
		std::regex re(file_pattern);

		for (auto& entry : fs::directory_iterator(dir))
		{
			// Skip hidden objects, e.g. directories '.DS_store' in OSX
			if (entry.path().filename().string()[0] == '.')
				continue; 

			std::string fullPath = entry.path().string(),
				pureFname = entry.path().filename().string();
			
			if (std::regex_match(pureFname, re))
			{
				fullpaths.push_back (fullPath);
				purefnames.push_back (pureFname);
			}
		}
	}

	std::optional<std::string> read_2D_dataset (
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
		// check the output directory
		if (!existsOnFilesystem(dirOut))
			return { "cannot access directory " + dirOut };

		// Check directories

		if (!existsOnFilesystem(dirIntens))
			return { "cannot access directory " + dirIntens };

		if (intLabMappingFile.empty())
		{
			std::vector<std::string> purefnames_i, purefnames_m; // we need these temp sets to check the 1:1 matching
			readDirectoryFiles_2D (dirIntens, filePatt, intensFiles, purefnames_i);

			// -- whole slide ?
			bool wholeslide = (dirIntens == dirLabels) || dirLabels.empty();

			if (wholeslide)
			{
				// populate with empty mask file names
				labelFiles.insert (labelFiles.begin(), intensFiles.size(), "");
			}
			else
			{
				if (!wholeslide && !existsOnFilesystem(dirLabels))
					return { "cannot access directory " + dirLabels };

				// read segmentation counterparts
				readDirectoryFiles_2D(dirLabels, filePatt, labelFiles, purefnames_m);

				// Check if the dataset is meaningful
				if (intensFiles.size() == 0 || labelFiles.size() == 0)
					return { "no intensity and/or label files to process, probably due to file pattern " + filePatt };
				if (intensFiles.size() != labelFiles.size())
					return { "mismatch: " + std::to_string(intensFiles.size()) + " intensity images vs " + std::to_string(labelFiles.size()) + " mask images" };

				// Sort the files to produce an intuitive sequence
				std::sort(intensFiles.begin(), intensFiles.end());
				std::sort(labelFiles.begin(), labelFiles.end());

				// Check if intensity and mask images are matching 
				auto nf = purefnames_i.size();
				std::string err;
				for (int i = 0; i < nf; i++)
				{
					auto& ifile = purefnames_i[i];
					if (std::find(purefnames_m.begin(), purefnames_m.end(), ifile) == purefnames_m.end())
						err += "cannot find the mask counterpart for " + ifile + "; ";
				}
				if (!err.empty())
					return { err };
			}
		}
		else
		{
			// Special case - using intensity and label file pairs defined with the mapping file
			if (!existsOnFilesystem(intLabMappingDir))
				return { "cannot access directory " + intLabMappingDir };

			std::string mapPath = intLabMappingDir + "/" + intLabMappingFile;
			if (!existsOnFilesystem(mapPath))
				return { "cannot access file " + mapPath };

			// Read the text file of intensity-mask matching
			std::ifstream file(mapPath);
			std::string ln, intFile, segFile;
			int lineNo = 1;
			while (std::getline(file, ln))
			{
				std::stringstream ss(ln);
				std::string intFname, segFname;
				bool pairOk = ss >> intFname && ss >> segFname;
				if (!pairOk)
					return { "cannot recognize a file name pair in line #" + std::to_string(lineNo) + " - " + ln };

				// We have a pair of file names. Let's check if they exist
				lineNo++;
				std::string intFpath = dirIntens + "/" + intFname;
				if (!existsOnFilesystem(intFpath))
					return { "cannot access file file " + intFpath };

				std::string segFpath = dirLabels + "/" + segFname;
				if (!existsOnFilesystem(intFpath))
					return { "cannot access file " + intFpath };

				// Save the file pair
				intensFiles.push_back (intFpath);
				labelFiles.push_back (segFpath);
			}

			// Check if we have pairs to process
			if (intensFiles.size() == 0)
				return { "special mapping " + mapPath + " produced no intensity-label file pairs" };
		}

		return { std::nullopt };
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
		else 
		{ 
			std::string erm = "\nError: cannot open file " + filePath + "\tDetails: " + __FILE__ + ":" + std::to_string(__LINE__) + "\n";
			std::cerr << erm;
			throw (std::runtime_error(erm)); 
		}
	}

	bool read_volumetric_filenames_as_25D (const std::string& dir, const StringPattern& filePatt, std::vector <Imgfile3D_layoutA>& files)
	{
		// grammar is OK to read data
		std::map<std::string, std::vector<std::string>> imgDirs;

		for (auto& fpath : fs::directory_iterator(dir))
		{
			// Skip hidden objects, e.g. directories '.DS_store' in OSX
			if (fpath.path().filename().string()[0] == '.')
				continue;

			std::string fullPath = fpath.path().string(),
				pureFname = fpath.path().filename().string();

			std::string ermsg;
			if (!filePatt.match(pureFname, imgDirs, ermsg))
			{
				std::cerr << "Error parsing file name " << fpath << " : " << ermsg << '\n';
				break;
			}

		} //- directory scan loop

		// copy the file info to the external container
		files.clear();
		size_t i = 0;
		for (const auto& x : imgDirs)
		{
			Nyxus::Imgfile3D_layoutA img3;
			img3.fname = x.first;	// image name
			img3.z_indices = x.second;	// z-indices as they are in corresponding file names
			files.push_back(img3);
		}

		return true;
	}

	bool read_volumetric_filenames (const std::string& dir, const StringPattern& filePatt, std::vector <Imgfile3D_layoutA>& files)
	{
		// read file names into a plain vector of strings
		std::vector<std::string> fn1, fn2;
		std::string p = filePatt.get_cached_pattern_string();
		readDirectoryFiles_2D (dir, p, fn1, fn2);

		// repackage it into a vector of extended file info (Imgfile3D_layoutA)
		files.clear();
		size_t i = 0;
		for (const auto& x : fn2)	// pure file names
		{
			Nyxus::Imgfile3D_layoutA img3;
			img3.fname = x;
			img3.fdir = dir;
			// intentionally leaving 'img3.z_indices' blank as we are not in 'layoutA' scenario
			files.push_back(img3);
		}

		return true;
	}

	bool readDirectoryFiles_3D (const std::string & dir, const StringPattern & fpatt, std::vector <Imgfile3D_layoutA> & fnames)
	{
		if (fpatt.is_25D())
			return read_volumetric_filenames_as_25D (dir, fpatt, fnames);
		else
			return read_volumetric_filenames (dir, fpatt, fnames);
	}

	std::optional<std::string> read_3D_dataset(
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
		std::vector <Imgfile3D_layoutA>& labelFiles)
	{
		if (!existsOnFilesystem(dirIntens))
			return { "Error: nonexisting directory " + dirIntens };
		if (!existsOnFilesystem(dirLabels))
			return { "Error: nonexisting directory " + dirLabels };
		if (!existsOnFilesystem(dirOut))
			return { "Error: nonexisting directory " + dirOut };

		// Common case - no ad hoc intensity-label file mapping, 1-to-1 correspondence instead
		if (!readDirectoryFiles_3D(dirIntens, filePatt, intensFiles))
			return { "Error reading directory " + dirIntens };
		if (!readDirectoryFiles_3D(dirLabels, filePatt, labelFiles))
			return { "Error reading directory " + dirLabels };

		// Check if the dataset isn't blank
		if (intensFiles.size() == 0 || labelFiles.size() == 0)
			return { "No intensity and/or label file pairs to process, probably due to file pattern " + filePatt.get_cached_pattern_string() };

		// There can be 2 layouts: 
		//		(1) 1-1 intensity-mask correspondence
		//		(2) 1-N intensity-mask correspondence

		// Shallow consistency check 
		// -- we check this only in layout (2)
		if (intensFiles.size() > 1)
			if (intensFiles.size() != labelFiles.size())
				return { "Mismatch: " + std::to_string(intensFiles.size()) + " intensity images vs " + std::to_string(labelFiles.size()) + " mask images" };

		// Deep consistency check 
		auto nf = labelFiles.size();
		if (intensFiles.size() > 1)
		{
			// -- layout #1: 1:1 correspondence
			for (auto i = 0; i < nf; i++)
			{
				auto& file_i = intensFiles[i],
					& file_m = labelFiles[i];

				// name mismatch ?
				if (file_i.fname != file_m.fname)
					return { "Mismatch: intensity " + file_i.fname + " mask " + file_m.fname };

				// z-stack size mismatch ?
				if (file_i.z_indices.size() != file_m.z_indices.size())
					return { "Z-stack size mismatch: intensity " + std::to_string(file_i.z_indices.size()) + " mask " + std::to_string(file_m.z_indices.size()) };

				// z-stack indices mismatch ?
				std::sort(file_i.z_indices.begin(), file_i.z_indices.end());
				std::sort(file_m.z_indices.begin(), file_m.z_indices.end());
				for (auto j = 0; j < file_i.z_indices.size(); j++)
					if (file_i.z_indices[j] != file_m.z_indices[j])
						return { "Mismatch in z-stack indices: " + file_i.fname + "[" + std::to_string(j) + "] != " + file_m.fname + "[" + std::to_string(j) + "]" };
			}
		}
		else
		{
			// layout #2: 1:N correspondence
			const auto ifile = intensFiles[0];
			intensFiles.clear();
			for (size_t i = 0; i < nf; i++)
				intensFiles.push_back(ifile);
		}

		// let each file know its directory
		for (auto i = 0; i < nf; i++)
		{
			auto& file_i = intensFiles[i],
				& file_m = labelFiles[i];
			file_i.fdir = dirIntens + '/';
			file_m.fdir = dirLabels + '/';
		}

		return { std::nullopt };
	}

	std::optional<std::string> read_3D_dataset_wholevolume (
		// input:
		const std::string& dirIntens,
		const StringPattern& filePatt,
		const std::string& dirOut,
		// output:
		std::vector <std::string>& intensFiles)
	{
		if (!existsOnFilesystem(dirIntens))
			return { "Error: nonexisting directory " + dirIntens };
		if (!existsOnFilesystem(dirOut))
			return { "Error: nonexisting directory " + dirOut };

		// Common case - no adhoc intensity-label file mapping, 1-to-1 correspondence instead

		// read file names into a plain vector of strings (non-wholevolume counterpart: readDirectoryFiles_3D() )
		std::vector<std::string> fnamesOnly;
		std::string p = filePatt.get_cached_pattern_string();
		readDirectoryFiles_2D (dirIntens, p, intensFiles, fnamesOnly);

		// Check if the dataset is blank
		if (intensFiles.size() == 0)
			return { "No intensity file pairs to process, probably due to file pattern " + filePatt.get_cached_pattern_string() };

		return std::nullopt;
	}

	Imgfile3D_layoutA::Imgfile3D_layoutA(const std::string& possibly_full_path)
	{
		auto p = fs::path(possibly_full_path);
		fname = p.filename().string();
		fdir = p.parent_path().string() + "/";
	}

	Imgfile3D_layoutA::Imgfile3D_layoutA()
	{
		fname = "";
		fdir = "";
	}


} // namespace Nyxus