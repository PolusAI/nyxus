#include "cli_nested_roi_options.h"
#include "helpers/helpers.h"

#include <sstream>
#include "dirs_and_files.h"
#include "environment.h"
#include "globals.h"
#include "nested_roi.h"
#include "results_cache.h"
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem> 
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif

bool NestedRoiOptions::empty()
{
	return rawChannelSignature.empty() ||
		rawParentChannelNo.empty() ||
		rawChildChannelNo.empty() ||
		rawAggregationMethod.empty();
}

bool NestedRoiOptions::parse_input()
{
	if (rawChannelSignature.empty())
	{
		ermsg = "Error in " + rawChannelSignature + ": expecting a non-empty string";
		return false;
	}

	if (! rawParentChannelNo.empty())
	{
		// string -> integer
		if (!Nyxus::parse_as_int (rawParentChannelNo, parentChannelNo))
		{
			ermsg = "Error in " + rawParentChannelNo + ": expecting an integer value";
			return false;
		}
	}
	else
	{
		ermsg = "Error in " + rawParentChannelNo + ": expecting a non-empty string";
		return false;
	}

	if (! rawChildChannelNo.empty())
	{
		// string -> integer
		if (!Nyxus::parse_as_int (rawChildChannelNo, childChannelNo))
		{
			ermsg = "Error in " + rawChildChannelNo + ": expecting an integer value";
			return false;
		}
	}
	else
	{
		ermsg = "Error in " + rawChildChannelNo + ": expecting a non-empty string";
		return false;
	}

	// parse the aggregation method argument
	if (!rawAggregationMethod.empty())
	{
		if (rawAggregationMethod == "SUM")
			aggrMethod = Aggregations::aSUM;
		else
		if (rawAggregationMethod == "MEAN")
			aggrMethod = Aggregations::aMEAN;
		else
		if (rawAggregationMethod == "MIN")
			aggrMethod = Aggregations::aMIN;
		else
		if (rawAggregationMethod == "MAX")
			aggrMethod = Aggregations::aMAX;
		else
		if (rawAggregationMethod == "WMA")
			aggrMethod = Aggregations::aWMA;
		else
		if (rawAggregationMethod == "NONE")
			aggrMethod = Aggregations::aNONE;
		else
		{
			ermsg = "Unrecognized ROI aggregation method " + rawAggregationMethod + " Valid options are SUM, MEAN, MIN, MAX, WMA, NONE";
			return false;
		}
	}
	else
	{
		ermsg = "Error in " + rawAggregationMethod + ": expecting a non-empty string";
		return false;
	}

	defined_ = true;

	return true;
}

bool NestedRoiOptions::defined()
{
	return defined_;
}

std::string NestedRoiOptions::get_last_er_msg()
{
	return ermsg;
}

namespace Nyxus 
{
	std::unordered_map <int, std::vector<int>> parentsChildren;
	std::unordered_map <int, HieLR> roiDataP, roiDataC;

	void parse_csv_line2 (std::vector<std::string>& dst, std::istringstream& src)
	{
		dst.clear();
		std::string field;
		while (getline(src, field, ','))
			dst.push_back(field);
	}

	bool find_csv_record2 (std::string& csvLine, std::vector<std::string>& csvHeader, std::vector<std::string>& csvFields, const std::string& csvFP, int label)
	{
		std::ifstream f(csvFP);
		std::string line;
		std::istringstream ssLine;

		// just store the header
		std::getline(f, line);
		ssLine.str(line);
		parse_csv_line2(csvHeader, ssLine);

		while (std::getline(f, line))
		{
			std::istringstream ss(line); 
			parse_csv_line2(csvFields, ss);

			std::stringstream ssLab;
			ssLab << label;
			if (csvFields[2] == ssLab.str())
			{
				csvLine = line;
				return true;
			}
		}

		return false;
	}


/// @brief Save results of one set of parents to the results cache
	bool output_roi_relational_table_2_rescache(
		const std::vector<int>& P,
		ResultsCache& rescache)
	{
		// Anything to do at all?
		if (P.size() == 0)
			return false;

		// Process parents
		for (auto l_par : P)
		{
			HieLR& r = Nyxus::roiDataP[l_par];
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

	/// @brief Save results of one set of parents to a csv-file
	bool output_roi_relational_table_2_csv (const std::vector<int>& P, const std::string& outdir)
	{
		// Anything to do at all?
		if (P.size() == 0)
			return true;

		// Make the relational table file name
		auto& fullSegImgPath = Nyxus::roiDataP[P[0]].segFname;
		fs::path pSeg(fullSegImgPath);
		auto segImgFname = pSeg.stem().string();
		std::string fPath = outdir + "/" + segImgFname + "_nested_relations.csv";	// output file path

		// --diagnostic--
		std::cout << "\nWriting relational structure to file " << fPath << "\n";

		// Output <-- parent header
		std::ofstream ofile;
		ofile.open(fPath);
		ofile << "Image,Parent_Label,Child_Label\n";

		// Process parents
		for (auto l_par : P)
		{
			HieLR& r = Nyxus::roiDataP[l_par];
			for (auto l_chi : r.child_segs)
			{
				ofile << r.segFname << "," << l_par << "," << l_chi << "\n";
			}
		}

		ofile.close();
		std::cout << "\nCreated file " << fPath << "\n";

		return true;
	}

	void output_roi_relations_2_csv (const std::string parent_fname, const Nyxus::NestableRois& roiData, const std::string& outdir)
	{
		// Make the relational table file name
		std::string fPath = outdir + "/" + parent_fname + "_nested_relations.csv";	// output file path

		VERBOSLVL2(std::cout << "\nWriting relational structure to file " << fPath << "\n");	

		// Write a header
		std::ofstream ofile;
		ofile.open(fPath);
		ofile << "Image,Parent_Label,Child_Label\n";

		for (auto& p : roiData)
		{
			const NestedLR& r = p.second;

			// Write relations
			for (auto childLabel : r.children)
				ofile << r.segFname << "," << p.first << "," << childLabel << "\n";
		}

		ofile.close();	
	}

	bool find_hierarchy(std::vector<int>& P, const std::string& par_fname, const std::string& chi_fname, int verbosity_level)
	{
		if (verbosity_level >= 1)
			std::cout << "\nUsing \n\t" << par_fname << " as container (parent) segment provider, \n\t" << chi_fname << " as child segment provider\n";

		// Cache the file names to be picked up by labels to know their file origin
		std::string base_parFname = baseFname (par_fname), 
			base_chiFname = baseFname (chi_fname);

		// Initialize children lists of all potential parents
		parentsChildren.clear();

		for (auto lp : uniqueLabels)
		{
			// Check if 'lp' is parent
			LR& rp = roiData[lp];
			std::string baseFN = baseFname (rp.segFname);
			if (baseFN != base_parFname)
				continue;

			// Get ahold of this parent's children list
			std::vector<int> chlist;
			parentsChildren[lp] = chlist;
		}

		// Match children with parents
		size_t n_orphans = 0, 
			n_non_orphans = 0;
		for (auto lc : uniqueLabels)
		{
			// Check if 'lc' is child
			LR& rc = roiData[lc];
			std::string baseFN = baseFname (rc.segFname);
			if (baseFN != base_chiFname)
				continue;

			AABB& chiBB = rc.aabb;
			bool parFound = false;

			for (auto lp : uniqueLabels)
			{
				// Check if 'lp' is parent
				LR& rp = roiData[lp];
				if (rp.segFname != par_fname)
					continue;

				// Get ahold of this parent's children list
				auto& children = parentsChildren[lp];

				// We found a pair or ROIs belonging to images of specified channels. 
				// Now check if one is inside the other via a strict containment
				AABB& parBB = rp.aabb;
				if (parBB.get_xmin() <= chiBB.get_xmin() &&
					parBB.get_xmax() >= chiBB.get_xmax() &&
					parBB.get_ymin() <= chiBB.get_ymin() &&
					parBB.get_ymax() >= chiBB.get_ymax())
				{

					children.push_back (lc);
					n_non_orphans++;
					parFound = true;
				}

			}
			if (parFound == false)
				n_orphans++;
		}

		// Build the parents set
		if (verbosity_level >= 1)
			std::cout << "Containers (parents):\n";
		int ordn = 1;
		for (auto lp : uniqueLabels)
		{
			// Check if 'lp' is parent
			LR& rp = roiData[lp];
			if (rp.segFname != par_fname)
				continue;

			// Its children data
			auto& children = parentsChildren[lp];
			if (children.size())
				P.push_back (lp);
		}

		// Find the max number of child segments to know how many children columns we have across the image
		size_t max_n_children = 0;
		for (auto lp : uniqueLabels)
		{
			// Check if 'lp' is parent
			LR& rp = roiData[lp];
			if (rp.segFname != par_fname)
				continue;

			// Its children data
			auto& children = parentsChildren[lp];
			max_n_children = std::max (max_n_children, children.size());
		}
		if (verbosity_level >= 1)
			std::cout << "\n# explained = " << n_non_orphans << "\n# orphans = " << n_orphans << "\nmax # children per container = " << max_n_children << "\n";

		return true;
	}

	std::string aggr_get_method_string (NestedRoiOptions::Aggregations a)
	{
		switch (a)
		{
			case NestedRoiOptions::Aggregations::aNONE:	return "NONE";
			case NestedRoiOptions::Aggregations::aSUM:	return "SUM";
			case NestedRoiOptions::Aggregations::aMEAN:	return "MEAN";
			case NestedRoiOptions::Aggregations::aMIN:	return "MIN";
			case NestedRoiOptions::Aggregations::aMAX:	return "MAX";
			case NestedRoiOptions::Aggregations::aWMA:	return "WMA";
		}
		return "UNKNOWN";
	}

	/// @brief Scans the feature database and aggregates features of labels in parameter "P" 
	/// according to aggregation "aggrs". The result goes to directory "outdir" 
	bool aggregate_features2 (Nyxus::NestableRois& P, Nyxus::NestableRois& C, const std::string& outdir, const std::string& parentFname, const NestedRoiOptions::Aggregations& aggr)
	{
		// Anything to do at all?
		if (P.size() == 0)
		{
			VERBOSLVL2(std::cout << "Error: empty parent set\n");
			return true;
		}

		//=== Find the max # of child segments to know how many children columns we have across the image
		size_t max_n_children = 0;
		for (auto& p : P)
		{
			const NestedLR& r = p.second;
			max_n_children = std::max (max_n_children, r.children.size());
		}

		// number of child feature sets
		int n_childSets = aggr == NestedRoiOptions::Aggregations::aNONE ? max_n_children : 1; 

		// columns of a feature result record that need to be skipped
		int skipNonfeatureColumns = mandatory_output_columns.size();

		//=== Header cells

		// Make the output table file name
		std::string fPath = outdir + "/" + parentFname + "_nested_features.csv";	// output file path

		VERBOSLVL2(std::cout << "\nWriting aligned nested features to file " << fPath << "\n");

		std::ofstream ofile;
		ofile.open (fPath);

		// *** Columns of the parent
		// mandatory columns #1, #2, and #3
		for (const auto& s : mandatory_output_columns)
			ofile << s << ",";

		// User feature selection
		std::vector<std::tuple<std::string, AvailableFeatures>> F = theFeatureSet.getEnabledFeatures();
		for (auto& f : F)
		{
			auto fn = std::get<0>(f);	// feature name
			auto fc = std::get<1>(f);	// feature code
			ofile << fn << ",";
		}

		//--no line break now--	ofile << "\n";
		
		// *** Columns of the children or their aggregate
		if (aggr == NestedRoiOptions::Aggregations::aNONE)
		{
			// No aggregation scenario, so the output is the full set of selected features times the number of child segments
			for (int iCh = 1; iCh <= n_childSets; iCh++)
				for (auto& enabdF : F)
				{
					auto fn = std::get<0>(enabdF);	// feature name
					auto fc = std::get<1>(enabdF);	// feature code
					ofile << "child" << iCh << "_" << fn << ",";
				}
		}
		else
		{
			// Aggregation scenario
			for (auto& enabdF : F)
			{
				auto fn = std::get<0>(enabdF);	// feature name
				auto fc = std::get<1>(enabdF);	// feature code
				ofile << "aggr_" << fn << ",";
			}
		}

		// End of the header line
		ofile << "\n";

		//=== Done with the header, now deal with data cells

		for (auto& p : P)
		{
			const NestedLR& r = p.second;
			int lPar = p.first;
			auto nCh = r.children.size();

			// Search this parent's feature extraction result recod 
			std::string csvFP = get_feature_output_fname(r.intFname, r.segFname);
			std::string csvWholeline;
			std::vector<std::string> csvHeader, csvFields;
			bool ok = find_csv_record2 (csvWholeline, csvHeader, csvFields, csvFP, lPar);
			if (ok == false)
			{
				std::cerr << "Cannot find record for parent " << lPar << " in " << csvFP << "\n";

				// Write emergency CSV-code to zero-fill incomplete date of this parent 

				// --- zero-fill parent feature cells
				for (auto& f : F)
					ofile << "0.0,";

				// --- zero-fill child feature cells
				for (int i=0; i< n_childSets; i++)
					for (auto& f : F)
						ofile << "0.0,";

				// --- nNew line and proceed to the next parent
				ofile << "\n";
				continue;
			}

			// Write parent features
			for (auto& field : csvFields)
				ofile << field << ",";

			// Don't break the line here (ofile << "\n")! Children features may follow

			//====== Write child features, aggregated or raw 

			if (aggr == NestedRoiOptions::Aggregations::aNONE)
			{
				// Iterate children segments writing their raw features 
				// (that is, without aggregating those features)
				int iCh = 0;	// counter of actual digested children to know how many child cells need to zero-fill wrt max number of children
				for (auto lChi : r.children)
				{
					const auto& rChi = C[lChi];
					std::string fpath = get_feature_output_fname (rChi.intFname, rChi.segFname);

					// read child's feature CSV file
					std::string rawLine;
					bool ok = find_csv_record2 (rawLine, csvHeader, csvFields, fpath, lChi);
					if (ok == false)
					{
						std::cerr << "Cannot find record for child " << lChi << " in " << fpath << "\n";
						continue;
					}

					// write features 
					for (int i=0; i < F.size(); i++)
						ofile << csvFields [skipNonfeatureColumns + i] << ",";

					// don't break the line either! More children features may follow-- ofile << "\n";

					// actual children written
					iCh++;
				}
				// zero-fill unused cells
				if (iCh < max_n_children)
				{
					for (int i=iCh; i<max_n_children; i++)
						for (auto& f : F)
							ofile << "0,";	// blank cell for every enabled feature
				}
			} //- no aggregation
			else
			{
				// Step 1/2: iterate children accumulating their feature values in corresponding buffers, vector 'aggrBuf'
				std::vector<std::vector<double>> aggrBuf;

				int iCh = 1;	// counter of actual digested children
				for (auto lChi : r.children)
				{
					// File where child's features reside
					const auto& r_chi = C[lChi];
					std::string csvFP_chi = get_feature_output_fname (r_chi.intFname, r_chi.segFname);

					// Read child's features
					std::string csvWholeline_chi;
					bool ok = find_csv_record2 (csvWholeline_chi, csvHeader, csvFields, csvFP_chi, lChi);
					if (ok == false)
					{
						std::cerr << "Cannot find record for child " << lPar << " in " << csvFP << "\n";
						continue;
					}

					// Accumulate child's features
					std::vector<double> childRow;
					for (int i=0; i<F.size(); i++)
					{
						auto & f = F[i];
						auto fn = std::get<0>(f);	// feature name
						
						// Parse a feature value. (Nans, infs, etc. need to be handled.)
						float val = 0.0f;
						parse_as_float (csvFields[skipNonfeatureColumns+i], val);
						childRow.push_back (val);
					}
					aggrBuf.push_back (childRow);

					// actual children written
					iCh++;
				}
			
				// Handle a special case of no children: in that case, we need to zero-fill 
				// the common table's cells
				int n_chi = aggrBuf.size();
				VERBOSLVL2(std::cout << "Parent " << lPar << ": " << n_chi << " child ROIs\n");
				if (n_chi == 0)
				{
					// Zero-fill child cells
					for (auto& f : F)
						ofile << "0.0,";
					// New line to make the CSV file ready for the next parent
					ofile << "\n";
					// Proceed to the next parent
					continue;
				}

				// Iterate accumulated feature values, aggregate them, and write in the common table
				// --- aggregate
				std::vector<double> feaAggregates;
				for (int f=0; f < F.size(); f++)
				{
					double aggResult = 0.0;
					switch (aggr)
					{
					case NestedRoiOptions::Aggregations::aSUM:
						for (int child = 0; child < n_chi; child++)
							aggResult += aggrBuf[child][f];
						break;
					case NestedRoiOptions::Aggregations::aMEAN:
						for (int child = 0; child < n_chi; child++)
							aggResult += aggrBuf[child][f];
						aggResult /= n_chi;
						break;
					case NestedRoiOptions::Aggregations::aMIN:
						aggResult = aggrBuf[0][f];
						for (int child = 0; child < n_chi; child++)
							aggResult = std::min(aggrBuf[child][f], aggResult);
						break;
					case NestedRoiOptions::Aggregations::aMAX:
						aggResult = aggrBuf[0][f];
						for (int child = 0; child < n_chi; child++)
							aggResult = std::max(aggrBuf[child][f], aggResult);
						break;
					default: // aWMA
						for (int child = 0; child < n_chi; child++)
							aggResult += aggrBuf[child][f];
						aggResult /= n_chi;
						break;
					}
					feaAggregates.push_back(aggResult);
				}
				// --- write
				for (int f=0; f < F.size(); f++)
					ofile << feaAggregates[f] << ",";
			} //- aggregation

			// Finished writing data of this parent. Line break
			ofile << "\n";

		} //- foreach parent

		ofile.close();

		return true;
	}

/// @brief Finds related (nested) segments and sets global variables 'pyHeader', 'pyStrData', and 'pyNumData' consumed by Python binding function findrelations_imp()
bool mine_segment_relations2 (
	const std::vector <std::string>& seg_files,
	const std::string& file_pattern,
	const std::string& channel_signature,
	const int parent_channel,
	const int child_channel,
	const std::string& outdir,
	const NestedRoiOptions::Aggregations& aggr,
	int verbosity_level)
{
	// Check if the dataset is meaningful
	if (seg_files.size() == 0)
		throw std::runtime_error("No label files to process");

	// Infer a list of stems - common parts of file names without channel parts
	
	// Reverse the signature before using
	auto signRev = channel_signature;
	std::reverse(signRev.begin(), signRev.end());

	// Gather stems
	std::vector<std::pair<std::string, std::string>> Stems;	// parent stems and tails; children are expected to have the same stem and tail differing only in the channel number
	std::string parPath, ext;
	for (auto& segf : seg_files)
	{
		fs::path p(segf);

		// Store the extension once
		if (ext.empty())
		{
			auto pExt = p.extension();
			ext = pExt.string();
		}

		auto pParPath = p.parent_path();
		parPath = pParPath.string();

		auto pStem = p.stem();
		std::string stem = pStem.string();
		std::reverse(stem.begin(), stem.end());

		auto loc = stem.find(signRev, 0);
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
			std::pair<std::string, std::string> stemtail = { stemNoChannel, tail };
			// Store it
			Stems.push_back(stemtail);
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
		std::string parFname = stem + std::to_string(parent_channel) + tail + ext;
		std::string chiFname = stem + std::to_string(child_channel) + tail + ext;

		VERBOSLVL2(std::cout << stem << "analyzing parent provider " << parFname << " vs children provider " << chiFname << "\n");

		// Analyze geometric relationships and recognize the hierarchy
		std::vector<int> parCandidates, chiCandidates;
		auto& pData = nestedRoiData[parFname];
		auto & cData = nestedRoiData[chiFname];

		// Let all the parent ROIs recognize their childred. After this step we will know if a ROI is a parent by its 'second.children.size()' > 0
		for (auto& p : pData)
		{
			auto& parBB = p.second.aabb;
			for (auto& c : cData)
				if (parBB.contains(c.second.aabb))
					p.second.children.push_back(c.first);
		}

		// Output the relational table to object 'theResultsCache'
		std::string purePrntFname = stem + std::to_string(parent_channel);
		output_roi_relations_2_csv (purePrntFname, pData, outdir);

		// Aggregate features
		VERBOSLVL2(std::cout << "Aggregating nested ROI in " + parFname + " : " + std::to_string(pData.size()) + " parents\n");
		bool ok = aggregate_features2 (pData, cData, outdir, purePrntFname, aggr);
		if (!ok)
			throw std::runtime_error("Error aggregating features");
	} //- stems

	return true; // success
}

void save_nested_roi_info (std::unordered_map <std::string, NestableRois>& dst_nestedRoiData, const std::unordered_set<int> & src_labels, std::unordered_map <int, LR>& src_roiData)
{
	for (auto l : src_labels)
	{
		LR & r = src_roiData[l];
		const auto& fname = baseFname (r.segFname);

		auto cnt = dst_nestedRoiData.count(fname);
		if (!cnt)
		{
			NestableRois nlr;
			dst_nestedRoiData[fname] = nlr;
		}

		NestableRois& sameFileRois = dst_nestedRoiData[fname];

		NestedLR nr(r);

		sameFileRois[r.label] = nr;
	}
}

} // namespace Nyxus
