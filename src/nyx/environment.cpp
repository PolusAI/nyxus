#define _CRT_SECURE_NO_WARNINGS

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <tuple>
#include <vector>
#include "environment.h"
#include "featureset.h"
#include "version.h"

bool directoryExists(const std::string&);

void parse_delimited_string_list(const std::string& rawString, std::vector<std::string>& result)
{

    result.clear();

    std::vector<std::string> S;

    std::string raw = rawString;    // a safe copy
    std::string delim = ",";
    size_t pos = 0;
    std::string token;
    while ((pos = raw.find(delim)) != std::string::npos)
    {
        token = raw.substr(0, pos);
        result.push_back(token); // std::cout << token << std::endl;
        raw.erase(0, pos + delim.length());
    }
    result.push_back(raw); // std::cout << raw << std::endl;
}

bool parse_as_float(std::string raw, float& result)
{
    if (sscanf(raw.c_str(), "%f", &result) != 1)
        return false;
    else
        return true;
}

bool parse_delimited_string_list_to_floats(const std::string& rawString, std::vector<float>& result)
{
    // It's legal to not have rotation angles specified
    if (rawString.length() == 0)
        return true;

    bool retval = true;
    std::vector<std::string> strings;
    parse_delimited_string_list (rawString, strings);
    result.clear();
    for (auto& s : strings)
    {
        float v;
        if (!parse_as_float(s, v))
        {
            retval = false;
            std::cout << "Error: in '" << rawString << "' expecting '" << s << "' to be a floating point number\n";
        }
        else
            result.push_back(v);
    }
    return retval;
}

std::string toupper (const std::string& s)
{
    auto s_uppr = s;
    for (auto& c : s_uppr)
        c = toupper(c);
    return s_uppr;
}

bool parse_delimited_string_list_to_features(const std::string& rawString, std::vector<std::string>& result)
{
    result.clear();

    if (rawString.length() == 0)
    {
        std::cout << "Warning: no features specified, defaulting to ALL\n";
        result.push_back(FEA_NICK_ALL);
        return true;
    }

    bool retval = true;
    std::vector<std::string> strings;
    parse_delimited_string_list(rawString, strings);

    // Check individual features
    for (const auto& s : strings)
    {
        auto s_uppr = toupper(s);
        if (s_uppr == FEA_NICK_ALL || 
            s_uppr == FEA_NICK_ALL_INTENSITY || 
            s_uppr == FEA_NICK_ALL_MORPHOLOGY || 
            s_uppr == FEA_NICK_BASIC_MORPHOLOGY || 
            s_uppr == FEA_NICK_ALL_GLCM || 
            s_uppr == FEA_NICK_ALL_GLRLM ||
            s_uppr == FEA_NICK_ALL_GLSZM ||
            s_uppr == FEA_NICK_ALL_GLDM ||
            s_uppr == FEA_NICK_ALL_NGTDM ||
            s_uppr == FEA_NICK_ALL_BUT_GABOR ||
            s_uppr == FEA_NICK_ALL_BUT_GLCM)
        {
            result.push_back(s_uppr);
            continue;
        }

        AvailableFeatures af;
        bool fnameExists = theFeatureSet.findFeatureByString(s_uppr, af);
        if (!fnameExists)
        {
            retval = false;
            std::cout << "Error: expecting '" << s << "' to be a proper feature name. \n";
        }
        else
            result.push_back(s_uppr);
    }

    // Show help on available features if necessary
    if (!retval)
        theFeatureSet.show_help();

    return retval;
}

void Environment::show_help()
{
    std::cout
        << PROJECT_NAME << " " << PROJECT_VER << "\nCopyright Axle Informatics 2021\n"
        << "Command line format:\n"
        << "\t" << PROJECT_NAME << " -h\tDisplay help info\n"
        << "\t" << PROJECT_NAME << " --help\tDisplay help info\n"
        << "\t" << PROJECT_NAME 
            << FILEPATTERN << " <fp> " 
            << OUTPUTTYPE << " <csv> " 
            << SEGDIR << " <sd> " 
            << INTDIR << " <id> " 
            << OUTDIR << " <od> "
            << " [" << FEATURES " <f>] "
            << " [" << EMBPIXSZ << " <eps>]\n"
            << " [" << LOADERTHREADS << " <lt>]\n"
            << " [" << PXLSCANTHREADS << " <st>]\n"
            << " [" << REDUCETHREADS << " <rt>]\n"
            << " [" << VERBOSITY << " <vl>]\n"
            << " [" << ROTATIONS << " <al>]\n"
            << " [" << VERBOSITY << " <verbo>]\n"

        << "Where\n"
        << "\t<fp> - file pattern regular expression e.g. .*, *.tif, etc [default = .*]\n"
        << "\t<csv> - 'separatecsv'[default] or 'singlecsv' \n"
        << "\t<sd> - directory of segmentation images \n"
        << "\t<id> - directory of intensity images \n"
        << "\t<od> - output directory \n"
        << "\t<f> - a specific feature or 'all' [default = 'all']\n"
        << "\t<eps> - [default = 0] \n"
        << "\t<lt> - number of image loader threads [default = 1] \n"
        << "\t<st> - number of pixel scanner threads within a TIFF tile [default = 1] \n"
        << "\t<rt> - number of feature reduction threads [default = 1] \n"
        << "\t<vl> - verbosity level [default = 0] \n"
        << "\t<al> - comma separated rotation angles [default = 0,45,90,135] \n"
        << "\t<verbo> - levels of verbosity 0 (silence), 2 (timing), 4 (roi diagnostics), 8 (granular diagnostics) [default = 0] \n"
        ;
}

void Environment::show_summary (const std::string & head, const std::string & tail)
{
    std::cout << head;
    std::cout << "Work plan:\n"
        << "\tlabels\t" << labels_dir << "\n"
        << "\tintensities\t" << intensity_dir << "\n"
        << "\toutput\t" << output_dir << "\n"
        << "\tfile pattern\t" << file_pattern << "\n"
        << "\tembedded pixel size\t" << embedded_pixel_size << "\n"
        << "\toutput type\t" << rawOutpType << "\n"
        << "\t# of image loader threads\t" << n_loader_threads << "\n"
        << "\t# of pixel scanner threads\t" << n_pixel_scan_threads << "\n"
        << "\t# of post-processing threads\t" << n_reduce_threads << "\n"
        << "\tverbosity level\t" << verbosity_level << "\n";

    // Features
    std::cout << "\tfeatures\t";
    for (auto f : desiredFeatures)
    {
        if (f != desiredFeatures[0])
            std::cout << ", ";
        std::cout << f;
    }
    std::cout << "\n";

    // Rotation angles
    std::cout << "\tangles of rotational features\t";
    for (auto ang : rotAngles)
    {
        if (ang != rotAngles[0])
            std::cout << ", ";
        std::cout << ang;
    }
    std::cout << "\n";

    std::cout << tail;
}

void Environment::show_memory(const std::string& head, const std::string& tail)
{
    std::cout << head << "Command line summary:\n";
    for (auto& m : memory)
    {
        std::cout << "\t" << std::get<0>(m) << " : " << std::get<1>(m) << "\n";
    }
    std::cout << tail;
}

bool Environment::find_string_argument (std::vector<std::string>::iterator& i, const char* arg, std::string& arg_value)
{
    std::string actualArgName = *i;

    // Syntax #1 <arg> <value>
    std::string a = arg;
    if (actualArgName == a)
    {
        arg_value = *++i;
        memory.push_back ({ a, arg_value });
        return true;
    }
    else
    {
        // Syntax #2 <arg>=<value>
        a.append("=");
        auto pos = actualArgName.find(a);
        if (pos != std::string::npos)
        {
            arg_value = actualArgName.substr(a.length());
            memory.push_back ({ a, arg_value });
            return true;
        }
    }
     
    // Argument was not recognized
    return false;
}

bool Environment::find_int_argument (std::vector<std::string>::iterator& i, const char* arg, int& arg_value)
{
    // Syntax #1
    std::string a = arg;
    if (*i == a)
    {
        std::string val = *++i;
        // string -> integer
        if (sscanf(val.c_str(), "%d", &arg_value) != 1)
            return true;
    }
    else
    {
        // Syntax #2
        a.append("=");
        auto pos = (*i).find(a);
        if (pos != std::string::npos)
        {
            std::string val = (*i).substr(a.length());
            // string -> integer
            if (sscanf(val.c_str(), "%d", &arg_value) != 1)
                return true;
        }
    }

    // Argument was not recognized
    return false;
}

bool Environment::check_file_pattern (const std::string & pat)
{
    try
    {
        std::regex re(pat);
    }
    catch(...)
    {
        return false;
    }

    return true;
}

// Examples:
// C:\WORK\AXLE\data\tissuenet\tissuenet-test-intensity C:\WORK\AXLE\data\tissuenet\tissuenet-test-labels C:\WORK\AXLE\data\output  
// "--features=kurtosis", "--filePattern=.*", "--csvfile=singlecsv", "--intDir=C:/WORK/AXLE/data/tissuenet/tissuenet-test-intensity", "--segDir=C:/WORK/AXLE/data/mesmer-untrained/labels", "--outDir=C:/WORK/AXLE/polus-feature-extraction-plugin/outputdir", "--embeddedpixelsize=true"
int Environment::parse_cmdline(int argc, char** argv)
{
    // Program being run without any flags and options?
    if (argc == 1)
        return 1;

    std::vector <std::string> args(argv + 1, argv + argc);
    std::vector <std::string> unrecognized;

    //==== Gather raw data
    for (auto i = args.begin(); i != args.end(); ++i)
    {
        if (*i == "-h" || *i == "--help")
        {
            show_help();
            return 1;
        }

        if (!(
            find_string_argument(i, INTDIR, intensity_dir) ||
            find_string_argument(i, SEGDIR, labels_dir) ||
            find_string_argument(i, OUTDIR, output_dir) ||
            find_string_argument(i, FEATURES, features) ||
            find_string_argument(i, FILEPATTERN, file_pattern) ||
            find_string_argument(i, OUTPUTTYPE, rawOutpType) ||
            find_string_argument(i, EMBPIXSZ, embedded_pixel_size) ||
            find_string_argument(i, LOADERTHREADS, loader_threads) ||
            find_string_argument(i, PXLSCANTHREADS, pixel_scan_threads) ||
            find_string_argument(i, REDUCETHREADS, reduce_threads) ||
            find_string_argument(i, ROTATIONS, rotations) ||
            find_string_argument(i, VERBOSITY, verbosity)
            ))
            unrecognized.push_back(*i);
    }

    //==== Report
    
    // --include the raw command line
    std::stringstream rawCL;
    rawCL << "\nRaw command line:\n" << argv[0] << " ";
    std::copy (args.begin(), args.end(), std::ostream_iterator<std::string> (rawCL, " "));  // vector of strings -> string
    rawCL << "\n\n";

    // --display how the command line was parsed
    show_memory (rawCL.str().c_str(), "\n");

    // --what's not recognized?
    if (unrecognized.size() > 0)
    {
        std::cout << "\nUnrecognized arguments:\n";
        for (auto& u : unrecognized)
            std::cout << "\t" << u << "\n";
    }
    std::cout << "\n";

    //==== Check mandatory parameters

    if (file_pattern == "")
    {
        std::cout << "Error: Missing argument " << FILEPATTERN << "\n";
        return 1;
    }
    if (check_file_pattern(file_pattern) == false)
    {
        std::cout << "Error: file pattern '" << file_pattern << "' is an invalid regular expression\n";
        return 1;
    }

    if (labels_dir == "")
    {
        std::cout << "Error: Missing argument " << SEGDIR << "\n";
        return 1;
    }
    if (intensity_dir == "")
    {
        std::cout << "Error: Missing argument " << INTDIR << "\n";
        return 1;
    }
    if (output_dir == "")
    {
        std::cout << "Error: Missing argument " << OUTDIR << "\n";
        return 1;
    }

    if (rawOutpType == "")
    {
        std::cout << "Error: Missing argument " << OUTPUTTYPE << "\n";
        return 1;
    }

    if (features == "")
    {
        std::cout << "Warning: " << FEATURES << "=<empty string>, defaulting to " << FEA_NICK_ALL << "\n";
        features = FEA_NICK_ALL;
    }

    //==== Single ROI?
    if (toupper(labels_dir) == toupper(intensity_dir))
    {
        singleROI = true;
        std::cout << 
            "+------------------------------+\n"
            "|                              |\n"
            "+  Activating single-ROI mode  +\n"
            "|                              |\n"
            "+------------------------------+\n";
    }

    //==== Output type
    auto rawOutpTypeUC = toupper(rawOutpType);
    if (rawOutpTypeUC != toupper(OT_SINGLECSV) && rawOutpTypeUC != toupper(OT_SEPCSV))
    {
            std::cout << "Error: valid values of " << OUTPUTTYPE << " are " << OT_SEPCSV << " or " << OT_SINGLECSV << "\n";
            return 1;    
    }
    separateCsv = rawOutpTypeUC == toupper(OT_SEPCSV);
        

    //==== Check numeric parameters
    if (! loader_threads.empty())
    {
        // string -> integer
        if (sscanf(loader_threads.c_str(), "%d", &n_loader_threads) != 1 || n_loader_threads <= 0)
        {
            std::cout << "Error: " << LOADERTHREADS << "=" << loader_threads << ": expecting a positive integer constant\n";
            return 1;
        }
    }

    if (! pixel_scan_threads.empty())
    {
        // string -> integer
        if (sscanf(pixel_scan_threads.c_str(), "%d", &n_pixel_scan_threads) != 1 || n_pixel_scan_threads <= 0)
        {
            std::cout << "Error: " << PXLSCANTHREADS << "=" << pixel_scan_threads << ": expecting a positive integer constant\n";
            return 1;
        }
    }

    if (! reduce_threads.empty())
    {
        // string -> integer
        if (sscanf(reduce_threads.c_str(), "%d", &n_reduce_threads) != 1 || n_reduce_threads <= 0)
        {
            std::cout << "Error: " << REDUCETHREADS << "=" << reduce_threads << ": expecting a positive integer constant\n";
            return 1;
        }
    }

    if (!verbosity.empty())
    {
        // string -> integer
        if (sscanf(verbosity.c_str(), "%d", &verbosity_level) != 1 || verbosity_level < 0)
        {
            std::cout << "Error: " << VERBOSITY << "=" << reduce_threads << ": expecting a positive integer constant\n";
            return 1;
        }
    }

    //==== Parse rotations
    if (!parse_delimited_string_list_to_floats (rotations, rotAngles))
    {
        return 1;
    }

    //==== Parse desired features
    
    // --Try to read a feature file
    if (features.length() >0 && directoryExists(features))
    {

        std::ifstream file(features);
        std::string ln, fileText;
        while (std::getline(file, ln))
        {
            if (fileText.length() > 0)
                fileText += ",";
            fileText += ln;
        }

        // Modify the input string
        features = fileText;

        std::cout << "Using features [" << fileText << "] from file " << features << "\n";
    }

    // --Make sure all the feature names are legal and cast to uppercase (class FeatureSet understands uppercase names)
    if (!parse_delimited_string_list_to_features (features, desiredFeatures)) 
    {
        return 1;
    }

    // --Feature names are ok, set the flags
    theFeatureSet.enableAll(false); // First, disable all
    for (auto& s : desiredFeatures) // Second, iterate uppercased feature names
    {
        // Check if features are requested via a group nickname
        if (s == FEA_NICK_ALL)
        {
            theFeatureSet.enableAll();
            break;  // No need to bother of others
        }
        if (s == FEA_NICK_ALL_BUT_GABOR)
        {
            theFeatureSet.enableAll();
            auto F = {GABOR};
            theFeatureSet.disableFeatures(F);
            break;  // No need to bother of others
        }
        if (s == FEA_NICK_ALL_BUT_GLCM)
        {
            theFeatureSet.enableAll();
            auto F = {
                TEXTURE_ANGULAR2NDMOMENT,
                TEXTURE_CONTRAST,
                TEXTURE_CORRELATION,
                TEXTURE_VARIANCE,
                TEXTURE_INVERSEDIFFERENCEMOMENT,
                TEXTURE_SUMAVERAGE,
                TEXTURE_SUMVARIANCE,
                TEXTURE_SUMENTROPY,
                TEXTURE_ENTROPY,
                TEXTURE_DIFFERENCEVARIANCE,
                TEXTURE_DIFFERENCEENTROPY,
                TEXTURE_INFOMEAS1,
                TEXTURE_INFOMEAS2 };
            theFeatureSet.disableFeatures(F);
            break;  // No need to bother of others
        }
        if (s == FEA_NICK_ALL_INTENSITY)
        {
            auto F = {
                MEAN,
                MEDIAN,
                MIN,
                MAX,
                RANGE,
                STANDARD_DEVIATION,
                SKEWNESS,
                KURTOSIS,
                MEAN_ABSOLUTE_DEVIATION,
                ENERGY,
                ROOT_MEAN_SQUARED,
                ENTROPY,
                MODE,
                UNIFORMITY,
                P10, P25, P75, P90,
                INTERQUARTILE_RANGE,
                ROBUST_MEAN_ABSOLUTE_DEVIATION,
                WEIGHTED_CENTROID_Y,
                WEIGHTED_CENTROID_X
            };
            theFeatureSet.enableFeatures(F);
            continue;
        }
        if (s == FEA_NICK_ALL_MORPHOLOGY)
        {
            auto F = {
                AREA_PIXELS_COUNT,
                CENTROID_X,
                CENTROID_Y,
                BBOX_YMIN,
                BBOX_XMIN,
                BBOX_HEIGHT,
                BBOX_WIDTH,
                MAJOR_AXIS_LENGTH,
                MINOR_AXIS_LENGTH,
                ECCENTRICITY,
                ORIENTATION,
                NUM_NEIGHBORS,
                EXTENT,
                ASPECT_RATIO,
                EQUIVALENT_DIAMETER,
                CONVEX_HULL_AREA,
                SOLIDITY,
                PERIMETER,
                CIRCULARITY
            };
            theFeatureSet.enableFeatures(F);
            continue;
        }
        if (s == FEA_NICK_BASIC_MORPHOLOGY)
        {
            auto F = {
                AREA_PIXELS_COUNT,
                CENTROID_X,
                CENTROID_Y,
                BBOX_YMIN,
                BBOX_XMIN,
                BBOX_HEIGHT,
                BBOX_WIDTH
            };
            theFeatureSet.enableFeatures(F);
            continue;
        }
        if (s == FEA_NICK_ALL_GLCM)
        {
            auto F = {
                TEXTURE_ANGULAR2NDMOMENT,
                TEXTURE_CONTRAST,
                TEXTURE_CORRELATION,
                TEXTURE_VARIANCE,
                TEXTURE_INVERSEDIFFERENCEMOMENT,
                TEXTURE_SUMAVERAGE,
                TEXTURE_SUMVARIANCE,
                TEXTURE_SUMENTROPY,
                TEXTURE_ENTROPY,
                TEXTURE_DIFFERENCEVARIANCE,
                TEXTURE_DIFFERENCEENTROPY,
                TEXTURE_INFOMEAS1,
                TEXTURE_INFOMEAS2 
            };
            theFeatureSet.enableFeatures (F);
            continue;
        }
        if (s == FEA_NICK_ALL_GLRLM)
        {
            auto F = {
                GLRLM_SRE,
                GLRLM_LRE,
                GLRLM_GLN,
                GLRLM_GLNN,
                GLRLM_RLN,
                GLRLM_RLNN,
                GLRLM_RP,
                GLRLM_GLV,
                GLRLM_RV,
                GLRLM_RE,
                GLRLM_LGLRE,
                GLRLM_HGLRE,
                GLRLM_SRLGLE,
                GLRLM_SRHGLE,
                GLRLM_LRLGLE,
                GLRLM_LRHGLE
            };
            theFeatureSet.enableFeatures(F);
            continue;
        }
        if (s == FEA_NICK_ALL_GLSZM)
        {
            auto F = {
                GLSZM_SAE,
                GLSZM_LAE,
                GLSZM_GLN,
                GLSZM_GLNN,
                GLSZM_SZN,
                GLSZM_SZNN,
                GLSZM_ZP,
                GLSZM_GLV,
                GLSZM_ZV,
                GLSZM_ZE,
                GLSZM_LGLZE,
                GLSZM_HGLZE,
                GLSZM_SALGLE,
                GLSZM_SAHGLE,
                GLSZM_LALGLE,
                GLSZM_LAHGLE 
            };
            theFeatureSet.enableFeatures (F);
            continue;
        }
        if (s == FEA_NICK_ALL_GLDM)
        {
            auto F = {
                GLDM_SDE,
                GLDM_LDE,
                GLDM_GLN,
                GLDM_DN,
                GLDM_DNN,
                GLDM_GLV,
                GLDM_DV,
                GLDM_DE,
                GLDM_LGLE,
                GLDM_HGLE,
                GLDM_SDLGLE,
                GLDM_SDHGLE,
                GLDM_LDLGLE,
                GLDM_LDHGLE
            };
            theFeatureSet.enableFeatures(F);
            continue;
        }
        if (s == FEA_NICK_ALL_NGTDM)
        {
            auto F = {
                NGTDM_COARSENESS,
                NGTDM_CONTRAST,
                NGTDM_BUSYNESS,
                NGTDM_COMPLEXITY,
                NGTDM_STRENGTH
            };
            theFeatureSet.enableFeatures(F);
            continue;
        }

        // Process features individually
        AvailableFeatures af;
        if (!theFeatureSet.findFeatureByString(s, af))
        {
            std::cout << "Error: expecting '" << s << "' to be a proper feature name. \n";
            return 1;
        }

        theFeatureSet.enableFeature(af);
    }

    // Success
    return 0;
}




