#define _CRT_SECURE_NO_WARNINGS

#include <fstream>
#include <iostream>
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

bool parse_delimited_string_list_to_features (const std::string& rawString, std::vector<std::string>& result)
{
    result.clear();

    if (rawString.length() == 0)
    {
        std::cout << "Warning: no features specified, defaulting to ALL\n";
        result.push_back("ALL");
        return true;
    }

    bool retval = true;
    std::vector<std::string> strings;
    parse_delimited_string_list (rawString, strings);

    // Check individual features
    for (const auto &s : strings)
    {
        auto s_uppr = toupper(s);
        if (s_uppr == "ALL")
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
            << CSVFILE << " <csv> " 
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
        << "\t<fp> - file pattern e.g. *, *.tif, etc [default = *]\n"
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
        << "\tcsv file\t" << csv_file << "\n"
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

void Environment::find_string_argument (std::vector<std::string>::iterator& i, const char* arg, std::string& arg_value)
{
    // Syntax #1
    std::string a = arg;
    if (*i == a)
    {
        arg_value = *++i;
        memory.push_back({ a, arg_value });
    }
    else
    {
        // Syntax #2
        a.append("=");
        auto pos = (*i).find(a);
        if (pos != std::string::npos)
        {
            arg_value = (*i).substr(a.length());
            memory.push_back({ a, arg_value });
        }
    }
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
            return false;
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
                return false;
        }
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

    //==== Gather raw data
    for (auto i = args.begin(); i != args.end(); ++i)
    {
        if (*i == "-h" || *i == "--help")
        {
            show_help();
            return 1;
        }

        find_string_argument (i, INTDIR, intensity_dir);
        find_string_argument (i, SEGDIR, labels_dir);
        find_string_argument (i, OUTDIR, output_dir);
        find_string_argument (i, FEATURES, features);
        find_string_argument (i, FILEPATTERN, file_pattern);
        find_string_argument (i, CSVFILE, csv_file);
        find_string_argument (i, EMBPIXSZ, embedded_pixel_size);
        find_string_argument (i, LOADERTHREADS, loader_threads);
        find_string_argument (i, PXLSCANTHREADS, pixel_scan_threads);
        find_string_argument (i, REDUCETHREADS, reduce_threads);
        find_string_argument (i, ROTATIONS, rotations);
        find_string_argument (i, VERBOSITY, verbosity);
    }

    //==== Report
    show_memory("\n","\n");

    //==== Check mandatory parameters
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
    if (file_pattern == "")
    {
        std::cout << "Error: Missing argument " << FILEPATTERN << "\n";
        return 1;
    }
    if (csv_file == "")
    {
        std::cout << "Error: Missing argument " << CSVFILE << "\n";
        return 1;
    }
    if (features == "")
    {
        std::cout << "Error: Missing argument " << FEATURES << "\n";
        return 1;
    }

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
        // Check if all the features are requested
        if (s == "ALL")
        {
            theFeatureSet.enableAll();
            break;  // No need to bother of others
        }

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




