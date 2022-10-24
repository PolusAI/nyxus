#include <algorithm>
#include <fstream>
#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif
#include "version.h"
#include "dirs_and_files.h"
#include "environment_basic.h" 
#include "globals.h"
#include "image_loader1x.h"
#include "nested_roi.h"
#include "nested_feature_aggregation.h"

bool mine_segment_relations(bool output2python, const std::string& label_dir, const std::string& file_pattern, const std::string& channel_signature, const int parent_channel, const int child_channel, const std::string& outdir, const ChildFeatureAggregation& aggr, int verbosity_level);

class NyxusHieEnvironment : public BasicEnvironment
{
public:
	NyxusHieEnvironment() {}
} theEnvironment;


namespace Nyxus
{

bool 	output_relational_table (const std::vector<int>& P, const std::string& outdir)
{
	// Anything to do at all?
	if (P.size() == 0)
		return false;

	// Make the relational table file name
	auto & fullSegImgPath = Nyxus::roiData1[P[0]].segFname;
	fs::path pSeg(fullSegImgPath);
	auto segImgFname = pSeg.stem().string();
	std::string fPath = outdir + "/" + segImgFname + "_nested_relations.csv";	// output file path

	// Debug
	std::cout << "\nWriting relational structure to file " << fPath << "\n";

	// Output <-- parent header
	std::ofstream ofile;
	ofile.open(fPath);
	ofile << "Image,Parent_Label,Child_Label\n";

	// Process parents
	for (auto l_par : P)
	{
		HieLR& r = Nyxus::roiData1[l_par]; 
		for (auto l_chi : r.child_segs)
		{
			ofile << r.segFname << "," << l_par << "," << l_chi << "\n";
		}
	}

	ofile.close();
	std::cout << "\nCreated file " << fPath << "\n";

	return true;
}

bool parse_as_float(const std::string& raw, float& result)
{
	if (sscanf(raw.c_str(), "%f", &result) != 1)
		return false;
	else
		return true;
}

bool 	shape_all_parents (const std::vector<int> & P, const std::string & outdir, const ChildFeatureAggregation & aggr)
{
	// Anything to do at all?
	if (P.size() == 0)
		return false;

	// Find the max # of child segments to know how many children columns we have across the image
	size_t max_n_children = 0;
	for (auto l_par : Nyxus::uniqueLabels1)
	{
		HieLR& r_par = Nyxus::roiData1[l_par];
		max_n_children = std::max(max_n_children, r_par.child_segs.size());
	}

	// Header
	// -- Read any CSV file and extract its header, we'll need it multiple times
	int lab_temp = P[0];
	HieLR& r_temp = Nyxus::roiData1 [lab_temp];
	std::string csvFP = outdir + "/" + r_temp.get_output_csv_fname();	

	if (!existsOnFilesystem(csvFP))
	{
		std::cout << "Error: cannot access file " << csvFP << std::endl;
		return false;
	}

	std::string csvWholeline;
	std::vector<std::string> csvHeader, csvFields;
	bool ok = find_csv_record (csvWholeline, csvHeader, csvFields, csvFP, lab_temp);
	if (ok == false)
	{
		std::cout << "Cannot find record for parent " << lab_temp << " in " << csvFP << ". Quitting\n";
		return false;	// pointless to continue if the very 1st parent is unavailable
	}

	// Make the output table file name
	auto& fullSegImgPath = Nyxus::roiData1[P[0]].segFname;
	fs::path pSeg(fullSegImgPath);
	auto segImgFname = pSeg.stem().string();
	std::string fPath = outdir + "/" + segImgFname + "_nested_features.csv";	// output file path

	// --diagnostic--
	std::cout << "\nWriting aligned nested features to file " << fPath << "\n";

	// Output <-- parent header
	std::string csvNFP = fPath; //---outdir + "/nested_features.csv";	// output file path
	std::ofstream ofile;
	ofile.open(csvNFP);
	for (auto& field : csvHeader)
		ofile << field << ",";
	//--no line break now--	ofile << "\n";

	// Iterate children
	if (aggr.get_method() == aNONE)
	{
		// We are in the no-aggregation scenario
		for (int iCh = 1; iCh <= max_n_children; iCh++)
		{
			// Output <-- child's header
			for (auto& field : csvHeader)
				ofile << "child_" << iCh << "_" << field << ",";
		}
	}
	else
	{
		// We are in the AGGREGATION scenario
		// Output <-- child's header
		for (auto& field : csvHeader)
			ofile << "aggr_" << field << ",";
	}
	ofile << "\n";

	// Process parents
	for (auto l_par : P)
	{
		HieLR& r = Nyxus::roiData1[l_par];
		std::string csvFP = outdir + "/" + r.get_output_csv_fname();
		//std::string csvWholeline;
		//std::vector<std::string> csvHeader, csvFields;
		bool ok = find_csv_record(csvWholeline, csvHeader, csvFields, csvFP, l_par);
		if (ok == false)
		{
			std::cout << "Cannot find record for parent " << l_par << " in " << csvFP << "\n";
			continue;
		}

		// Output <-- parent features 
		for (auto& field : csvFields)
			ofile << field << ",";
		//-- don't break the line! children features will follow-- ofile << "\n";

		if (aggr.get_method() == aNONE)
		{
			// write features of all the children without aggregation
			int iCh = 1;
			for (auto l_chi : r.child_segs)
			{
				HieLR& r_chi = Nyxus::roiData2[l_chi];
				std::string csvFN_chi = r_chi.get_output_csv_fname();
				std::string csvFP_chi = outdir + "/" + csvFN_chi;
				std::string csvWholeline_chi;
				bool ok = find_csv_record(csvWholeline_chi, csvHeader, csvFields, csvFP_chi, l_chi);
				if (ok == false)
				{
					std::cout << "Cannot find record for child " << l_par << " in " << csvFP << "\n";
					continue;
				}

				// Output <-- child features 
				for (auto& field : csvFields)
					ofile << field << ",";
				//-- don't break the line either! more children features will follow-- ofile << "\n";

				// childrens' count
				iCh++;
			}
			// write empty cells if needed
			if (iCh < max_n_children)
			{
				for (int iCh2 = iCh; iCh2 <= max_n_children; iCh2++)
				{
					for (auto& field : csvFields)
						ofile << "0" << ",";	// blank cell
				}
			}
		} // no aggregation
		else
		{
			// read and aggregate
			std::vector<std::vector<double>> aggrBuf;

			int iCh = 1;
			for (auto l_chi : r.child_segs)
			{
				HieLR& r_chi = Nyxus::roiData2[l_chi];
				std::string csvFN_chi = r_chi.get_output_csv_fname();
				std::string csvFP_chi = outdir + "/" + csvFN_chi;
				std::string csvWholeline_chi;
				bool ok = find_csv_record(csvWholeline_chi, csvHeader, csvFields, csvFP_chi, l_chi);
				if (ok == false)
				{
					std::cout << "Cannot find record for child " << l_par << " in " << csvFP << "\n";
					continue;
				}

				// Output <-- child features 
				std::vector<double> childRow;
				for (auto& field : csvFields)
				{
					// Parse a table cell value. (Difficulty - nans, infs, etc.)
					float val = 0.0f;
					parse_as_float (field, val);
					childRow.push_back(val); //---  ofile << field << ","
				}
				aggrBuf.push_back(childRow);

				// childrens' count
				iCh++;
			}

			int n_chi = aggrBuf.size();

			// write aggregated
			//--first, aggregate
			std::vector<double> feaAggregates;
			for (int fea = 0; fea < csvFields.size(); fea++)
			{
				double aggResult = 0.0;
				switch (aggr.get_method())
				{
				case aSUM:
					for (int child = 0; child < n_chi; child++)
						aggResult += aggrBuf[child][fea];
					break;
				case aMEAN:
					for (int child = 0; child < n_chi; child++)
						aggResult += aggrBuf[child][fea];
					aggResult /= n_chi;
					break;
				case aMIN:
					aggResult = aggrBuf[0][fea];
					for (int child = 0; child < n_chi; child++)
						aggResult = std::min (aggrBuf[child][fea], aggResult);
					break;
				case aMAX:
					aggResult = aggrBuf[0][fea];
					for (int child = 0; child < n_chi; child++)
						aggResult = std::max(aggrBuf[child][fea], aggResult);
					break;
				default: // aWMA
					for (int child = 0; child < n_chi; child++)
						aggResult += aggrBuf[child][fea];
					aggResult /= n_chi;
					break;
				}
				feaAggregates.push_back(aggResult);
			}
			//--second, write
			for (int fea = 0; fea < csvFields.size(); fea++)
				ofile <<  feaAggregates[fea] << ",";	
		}

		// Output <-- line break
		ofile << "\n";
	}

	ofile.close();
	std::cout << "\nCreated file " << csvNFP << "\n";

	return true;
}

} // namespace Nyxus



#define OPTION_AGGREGATE "-aggregate"

int main (int argc, char** argv)
{
	std::cout << PROJECT_NAME << " /// " << PROJECT_VER << " /// (c) 2021-2022 Axle Informatics\t" << "Build of " << __TIMESTAMP__ << "\n";

	// Process the command line: check the command line (straightforward way - strictly positional)
	if (argc < 7)
	{
		std::cout << "nyxushie <mask 2D images directory> <file pattern> <channel signature> <parent channel> <child channel> <features dir> [" << OPTION_AGGREGATE << "=<aggregation method>]\n" 
			<< "\t<aggregation method> is " << ChildFeatureAggregation::get_valid_options() << "\n";
		std::cout << "Example: nyxushie /path/to/mask/2d-images/directory train_.*\\.tif _ch 1 0 /path/to/result/directory \n";
		return 1;
	}

	// Process the command line: consume the mandatory arguments
	std::string segCollectionDir = argv[1],
		filePattern = argv[2],
		channelSign = argv[3], 
		parentChannel = argv[4], 
		childChannel = argv[5], 
		resultFeaturesDir = argv[6];

	// -- file pattern
	if (!theEnvironment.check_file_pattern(filePattern))
	{
		std::cerr << "Filepattern provided is not valid\n";
		return 1;
	}

	// -- parent & child channel numbers
	int n_parent_channel;
	if (sscanf(parentChannel.c_str(), "%d", &n_parent_channel) != 1)
	{
		std::cerr << "Error parsing the parent channel number\n";
		return 1;
	}

	int n_child_channel;
	if (sscanf(childChannel.c_str(), "%d", &n_child_channel) != 1)
	{
		std::cerr << "Error parsing the child channel number\n";
		return 1;
	}

	// Process the command line: check the the aggregation option
	ChildFeatureAggregation aggr (OPTION_AGGREGATE);
	if (argc == 8)
	{
		auto rawAggrArg = argv[7];
		if (!aggr.parse(rawAggrArg))
		{
			std::cerr << "Error parsing the aggregation method argument " << rawAggrArg << " . Valid options are : " << OPTION_AGGREGATE << "=" << ChildFeatureAggregation::get_valid_options() << "\n";
			return 1;
		}
	}

	// Mine relations and leave the result in object 'theResultsCache'
	bool mineOK = mine_segment_relations(
		false,
		segCollectionDir,
		filePattern,
		channelSign,
		n_parent_channel,
		n_child_channel, 
		resultFeaturesDir,
		aggr, 
		theEnvironment.get_verbosity_level());
	if (! mineOK)
	{
		std::cerr << "Error minimg hierarchical relations\n";
		return 1;
	}

	return 0;
}

namespace Nyxus
{
	// Results cache serving Nyxus' CLI & Python API, NyxusHie's CLI & Python API
	ResultsCache theResultsCache;
}