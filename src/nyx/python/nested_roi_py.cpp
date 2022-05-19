#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#ifdef WITH_PYTHON_H
	#include <pybind11/pybind11.h>
#endif
#include "../environment.h"
#include "../globals.h"
#include "../image_loader1x.h"
#include "../results_cache.h"
#include "../roi_cache.h"

/// @brief Segment data cache for finding segment hierarchies 
class HieLR : public LR
{
public:
	std::vector<int> child_segs;
};

namespace Nyxus
{
	// Tables referring ROI labels to their cache per each parent-child image pair 
	std::unordered_set <int> uniqueLabels1, uniqueLabels2;
	std::unordered_map <int, HieLR> roiData1, roiData2;
	std::string theParFname, theChiFname;

	/// @brief ROI cache structure initializer for nested ROI functionality (Python class NyxusHie, nyxushie CLI)
	void init_label_record_2 (HieLR& r, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity, unsigned int tile_index)
	{
		// Cache the host tile's index
		r.host_tiles.insert(tile_index);

		// Initialize basic counters
		r.aux_area = 1;
		r.aux_min = r.aux_max = intensity;
		r.init_aabb(x, y);

		// Cache the ROI label
		r.label = label;

		// File names
		r.segFname = segFile;
		r.intFname = intFile;
	}

	/// @brief ROI cache structure updater for nested ROI functionality (Python class NyxusHie, nyxushie CLI)
	void update_label_record_2 (HieLR& lr, int x, int y, int label, PixIntens intensity, unsigned int tile_index)
	{
		// Cache the host tile's index
		lr.host_tiles.insert(tile_index);

		// Initialize basic counters
		lr.aux_area++;
		lr.aux_min = std::min(lr.aux_min, intensity);
		lr.aux_max = std::max(lr.aux_max, intensity);
		lr.update_aabb(x, y);
	}

	/// @brief Pixel feeder for nested ROI functionality (Python class NyxusHie, nyxushie CLI)
	void feed_pixel_2_metrics_H(
		std::unordered_set <int>& UL, // unique labels
		std::unordered_map <int, HieLR>& RD,	// ROI data
		int x, int y, int label, unsigned int tile_index)
	{
		if (UL.find(label) == UL.end())
		{
			// Remember this label
			UL.insert(label);

			// Initialize the ROI label record
			HieLR newData;
			init_label_record_2(newData, theParFname, "no2ndfile", x, y, label, 999/*dummy intensity*/, tile_index);
			RD[label] = newData;
		}
		else
		{
			// Update basic ROI info (info that doesn't require costly calculations)
			HieLR& existingData = RD[label];
			update_label_record_2(existingData, x, y, label, 999/*dummy intensity*/, tile_index);
		}
	}
}

/// @brief Gathers online morphological properties for nested ROI functionality (Python class NyxusHie, nyxushie CLI)
bool gatherRoisMetrics_H(const std::string& fpath, std::unordered_set <int>& uniqueLabels, std::unordered_map <int, HieLR>& roiData)
{
	theParFname = fpath;

	int lvl = 0, // Pyramid level
		lyr = 0; //	Layer

	// Open an image pair
	ImageLoader1x imlo;
	bool ok = imlo.open(fpath);
	if (!ok)
	{
		std::stringstream ss;
		ss << "Error opening file " << fpath;
#ifdef WITH_PYTHON_H
		throw ss.str();
#endif	
		std::cerr << ss.str() << "\n";
		return false;
	}

	// Read the tiff
	size_t nth = imlo.get_num_tiles_hor(),
		ntv = imlo.get_num_tiles_vert(),
		fw = imlo.get_tile_width(),
		th = imlo.get_tile_height(),
		tw = imlo.get_tile_width(),
		tileSize = imlo.get_tile_size();

	int cnt = 1;
	for (unsigned int row = 0; row < nth; row++)
		for (unsigned int col = 0; col < ntv; col++)
		{
			// Fetch the tile 
			ok = imlo.load_tile(row, col);
			if (!ok)
			{
				std::stringstream ss;
				ss << "Error fetching tile row=" << row << " col=" << col;
#ifdef WITH_PYTHON_H
				throw ss.str();
#endif	
				std::cerr << ss.str() << "\n";
				return false;
			}

			// Get ahold of tile's pixel buffer
			auto tileIdx = row * nth + col;
			auto data = imlo.get_tile_buffer();

			// 1st image -> uniqueLabels1, roiData1
			for (size_t i = 0; i < tileSize; i++)
			{
				// Skip non-mask pixels
				auto label = data[i];
				if (label != 0)
				{
					int y = row * th + i / tw,
						x = col * tw + i % tw;

					// Update pixel's ROI metrics
					feed_pixel_2_metrics_H(uniqueLabels, roiData, x, y, label, tileIdx); // Updates 'uniqueLabels' and 'roiData'
				}
			}

#ifdef WITH_PYTHON_H
			if (PyErr_CheckSignals() != 0)
				throw pybind11::error_already_set();
#endif

			// Show stayalive progress info
			if (cnt++ % 4 == 0)
				VERBOSLVL1(std::cout << "\t" << int((row * nth + col) * 100 / float(nth * ntv) * 100) / 100. << "%\t" << uniqueLabels.size() << " ROIs" << "\n");
		}

	imlo.close();
	return true;
}

/// @brief Scans a pair of mask image files 'par_fname' (parent segments) and 'chi_fname' (child segments) and searches segments in 'par_fname' enveloping segments in 'chi_fname'
/// @param P Output vector of segments in 'par_fname' enveloping at least 1 segment in 'chi_fname'
/// @param par_fname Mask image of parent segments e.g. cell membrane
/// @param chi_fname Mask image of child segments e.g. cell nuclei
/// @return Vector of parent segment labels 'P'
bool find_hierarchy (std::vector<int>& P, const std::string& par_fname, const std::string& chi_fname)
{
	VERBOSLVL1(std::cout << "\nUsing \n\t" << par_fname << " as container (parent) segment provider, \n\t" << chi_fname << " as child segment provider\n";)

	// Cache the file names to be picked up by labels to know their file origin
	std::filesystem::path parPath(par_fname), chiPath(chi_fname);

	// scan parent segments
	gatherRoisMetrics_H(parPath.string(), uniqueLabels1, roiData1);

	// scan child segments
	gatherRoisMetrics_H(chiPath.string(), uniqueLabels2, roiData2);

	size_t n_orphans = 0, n_non_orphans = 0;
	for (auto l_chi : Nyxus::uniqueLabels2)
	{
		HieLR& r_chi = Nyxus::roiData2[l_chi];
		const AABB& chiBB = r_chi.aabb;
		bool parFound = false;
		for (auto l2 : Nyxus::uniqueLabels1)
		{
			HieLR& r_par = Nyxus::roiData1[l2];
			const AABB& parBB = r_par.aabb;

			// Check the strict containment
			if (parBB.get_xmin() <= chiBB.get_xmin() &&
				parBB.get_xmax() >= chiBB.get_xmax() &&
				parBB.get_ymin() <= chiBB.get_ymin() &&
				parBB.get_ymax() >= chiBB.get_ymax())
			{
				r_par.child_segs.push_back(l_chi);
				n_non_orphans++;
				parFound = true;
			}
		}

		if (parFound == false)
			n_orphans++;
	}

	// Build the parents set
	VERBOSLVL1(std::cout << "Containers (parents):\n";)
	int ordn = 1;
	for (auto l_par : Nyxus::uniqueLabels1)
	{
		HieLR& r_par = Nyxus::roiData1[l_par];
		if (r_par.child_segs.size())
		{
			P.push_back(l_par);
			//--Diagnostic--	std::cout << "\t(" << ordn++ << ")\t" << l_par << " : " << r_par.child_segs.size() << " ch\n";
		}
	}

	// Optional - find the max # of child segments to know how many children columns we have across the image
	size_t max_n_children = 0;
	for (auto l_par : Nyxus::uniqueLabels1)
	{
		HieLR& r_par = Nyxus::roiData1[l_par];
		max_n_children = std::max(max_n_children, r_par.child_segs.size());
	}
	VERBOSLVL1(std::cout << "\n# explained = " << n_non_orphans << "\n# orphans = " << n_orphans << "\nmax # children per container = " << max_n_children << "\n";)

	return true;
}

bool 	output_roi_relational_table (
	const std::vector<int>& P, 
	ResultsCache& rescache)
{
	// Anything to do at all?
	if (P.size() == 0)
		return false;

	// Process parents
	for (auto l_par : P)
	{
		HieLR& r = Nyxus::roiData1[l_par];
		for (auto l_chi : r.child_segs)
		{
			rescache.add_string(r.segFname);	
			rescache.add_numeric(l_par); 
			rescache.add_numeric(l_chi); 
			rescache.inc_num_rows(); 
		}
	}

	return true;
}

/// @brief Finds related (nested) segments and sets global variables 'pyHeader', 'pyStrData', and 'pyNumData' consumed by Python binding function findrelations_imp()
int mine_segment_relations (const std::string& label_dir, const std::string& file_pattern, const std::string& channel_signature, const int parent_channel, const int child_channel)
{
	std::vector<std::string> segFiles;
	readDirectoryFiles(label_dir, file_pattern, segFiles);

	// Check if the dataset is meaningful
	if (segFiles.size() == 0)
	{
		throw std::runtime_error("No label files to process");
	}

	// Reverse the signature before using
	auto signRev = channel_signature;
	std::reverse (signRev.begin(), signRev.end());

	// Gather stems
	std::vector<std::pair<std::string, std::string>> Stems;		// parent stems and tails; children are expected to have the same stem and tail differing only in the channel number
	std::string parPath, ext;
	for (auto& segf : segFiles)
	{
		std::filesystem::path p(segf);

		// Store the extension once
		if (ext.empty())
		{
			auto pExt = p.extension();
			ext = pExt.string();		// store it
		}

		auto pParPath = p.parent_path();
		parPath = pParPath.string();	// store it

		auto pStem = p.stem();
		std::string stem = pStem.string();
		std::reverse(stem.begin(), stem.end());

		auto loc = stem.find (signRev, 0);
		std::string channel = stem.substr(0, loc);
		std::string stemNoChannel = stem.substr(loc, stem.size() - loc);

		// Tear off the non-numeric part
		std::reverse(channel.begin(), channel.end());	// we need it in the natural way (numeric part first) now
		int lenOfNumeric = -1;
		for (int i = 0; i < channel.size(); i++)
		{
			auto ch = channel[i];
			if (isdigit(ch))
				lenOfNumeric = i + 1;
			else
				break;
		}

		if (lenOfNumeric <= 0)
		{
			std::stringstream ss;
			ss << "Cannot find the numeric part in channel '" << channel;
			throw std::runtime_error(ss.str());
		}

		std::string numericChannel = channel.substr(0, lenOfNumeric);
		std::string tail = channel.substr(lenOfNumeric, channel.size());	

		// String to int
		int n_channel = std::atoi(numericChannel.c_str());

		// Store only parent channels
		if (n_channel == parent_channel)
		{
			// Flip the stem back to normal
			std::reverse(stemNoChannel.begin(), stemNoChannel.end());
			// Prepare a stem-tail pair
			std::pair<std::string, std::string> stemtail = {stemNoChannel, tail};
			// Store it
			Stems.push_back (stemtail);
		}
	}

	// Prepare the buffers. 
	// 'totalNumLabels', 'stringColBuf', and 'calcResultBuf' will be updated with every call of output_roi_relational_table()
	theResultsCache.clear();

	// Prepare the header
	theResultsCache.add_to_header({ "Image", "Parent_Label", "Child_Label" });

	// Mine parent-child relations 
	for (auto& parStemInfo : Stems)
	{
		auto stem = parStemInfo.first, 
			tail = parStemInfo.second;

		// Form parent and child file names
		std::stringstream ssParFname, ssChiFname;
		ssParFname << parPath << "/" << stem << parent_channel << tail << ext;
		ssChiFname << parPath << "/" << stem << child_channel << tail << ext;

		// Debug
		VERBOSLVL1(std::cout << stem << "\t" << parent_channel << ":" << child_channel << "\n";	)

		// Clear reference tables
		uniqueLabels1.clear();
		uniqueLabels2.clear();
		roiData1.clear();
		roiData2.clear();

		// Analyze geometric relationships and recognize the hierarchy
		std::vector<int> P;	// parents
		bool ok = find_hierarchy(P, ssParFname.str(), ssChiFname.str());
		if (!ok)
		{
			std::stringstream ss;
			ss << "Error finding hierarchy based on files " << ssParFname.str() << " as parent and " << ssChiFname.str() << " as children";
			throw std::runtime_error(ss.str());
		}

		// Output the relational table
		ok = output_roi_relational_table (
			P, 
			theResultsCache
			);
		if (!ok)
		{
			throw std::runtime_error("Cannot produce the output: somethig is wrong with data. Quitting");
		}
	}

	return 0;	// success
}