#include <algorithm>
#include <exception>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../version.h"
#include "../environment.h"
#include "../feature_mgr.h"
#include "../dirs_and_files.h"  
#include "../globals.h"
#include "../nested_feature_aggregation.h"

namespace py = pybind11;
using namespace Nyxus;

// Defined in nested.cpp
bool mine_segment_relations (
	bool output2python, 
	const std::string& label_dir,
	const std::string& parent_file_pattern,
	const std::string& child_file_pattern,
	const std::string& outdir, 
	const ChildFeatureAggregation& aggr, 
	int verbosity_level);

template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence &&seq)
{
    auto size = seq.size();
    auto data = seq.data();
    std::unique_ptr<Sequence> seq_ptr = std::make_unique<Sequence>(std::move(seq));
    auto capsule = py::capsule(seq_ptr.get(), [](void *p)
                               { std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p)); });
    seq_ptr.release();
    return py::array(size, data, capsule);
}

void initialize_environment(
    const std::vector<std::string> &features,
    float neighbor_distance,
    float pixels_per_micron,
    uint32_t coarse_gray_depth, 
    uint32_t n_reduce_threads,
    uint32_t n_loader_threads,
    int using_gpu)
{
    theEnvironment.desiredFeatures = features;
    theEnvironment.set_pixel_distance(static_cast<int>(neighbor_distance));
    theEnvironment.set_verbosity_level (0);
    theEnvironment.xyRes = theEnvironment.pixelSizeUm = pixels_per_micron;
    theEnvironment.set_coarse_gray_depth(coarse_gray_depth);
    theEnvironment.n_reduce_threads = n_reduce_threads;
    theEnvironment.n_loader_threads = n_loader_threads;

    // Throws exception if invalid feature is supplied.
    theEnvironment.process_feature_list();
    theFeatureMgr.compile();
    theFeatureMgr.apply_user_selection();

    #ifdef USE_GPU
        if(using_gpu == -1) {
            theEnvironment.set_use_gpu(false);
        } else {
            theEnvironment.set_gpu_device_id(using_gpu);
        }
    #else 
        if (using_gpu != -1) {
            std::cout << "No gpu available." << std::endl;
        }
    #endif
}

py::tuple featurize_directory_imp_fast (
    const std::string &intensity_dir,
    const std::string &labels_dir,
    const std::string &file_pattern,
    bool use_fastloop)
{
    theEnvironment.intensity_dir = intensity_dir;
    theEnvironment.labels_dir = labels_dir;
    theEnvironment.set_file_pattern (file_pattern);

    if (!theEnvironment.check_file_pattern(file_pattern))
        throw std::invalid_argument("Filepattern provided is not valid.");

    std::vector<std::string> intensFiles, labelFiles;
    int errorCode = Nyxus::read_dataset(
        intensity_dir,
        labels_dir,
        theEnvironment.get_file_pattern(), 
        "./",   // output directory
        theEnvironment.intSegMapDir,
        theEnvironment.intSegMapFile,
        true,
        intensFiles, labelFiles);

    if (errorCode)
        throw std::runtime_error("Dataset structure error.");

    init_feature_buffers();

    theResultsCache.clear();

    // Process the image sdata
    int min_online_roi_size = 0;
    errorCode = processDataset(
        intensFiles,
        labelFiles,
        theEnvironment.n_loader_threads,
        theEnvironment.n_pixel_scan_threads,
        theEnvironment.n_reduce_threads,
        min_online_roi_size,
        false, // 'true' to save to csv
        theEnvironment.output_dir,
        use_fastloop);

    if (errorCode)
        throw std::runtime_error("Error occurred during dataset processing.");

    auto pyHeader = py::array(py::cast(theResultsCache.get_headerBuf()));
    auto pyStrData = py::array(py::cast(theResultsCache.get_stringColBuf()));
    auto pyNumData = as_pyarray(std::move(theResultsCache.get_calcResultBuf()));
    auto nRows = theResultsCache.get_num_rows();
    pyStrData = pyStrData.reshape({nRows, pyStrData.size() / nRows});
    pyNumData = pyNumData.reshape({ nRows, pyNumData.size() / nRows });

    return py::make_tuple(pyHeader, pyStrData, pyNumData);
}

py::tuple featurize_directory_imp (
    const std::string &intensity_dir,
    const std::string &labels_dir,
    const std::string &file_pattern)
{
    return featurize_directory_imp_fast (
    intensity_dir,
    labels_dir,
    file_pattern,
    false);
}

py::tuple featurize_fname_lists_imp_fast (const py::list& int_fnames, const py::list & seg_fnames, bool use_fastloop)
{
    std::vector<std::string> intensFiles, labelFiles;
    for (auto it = int_fnames.begin(); it != int_fnames.end(); ++it)
    {
        std::string fn = it->cast<std::string>();
        intensFiles.push_back(fn);
    }
    for (auto it = seg_fnames.begin(); it != seg_fnames.end(); ++it)
    {
        std::string fn = it->cast<std::string>();
        labelFiles.push_back(fn);
    }

    // Check the file names 
    if (intensFiles.size() == 0)
        throw std::runtime_error("Intensity file list is blank");
    if (labelFiles.size() == 0)
        throw std::runtime_error("Segmentation file list is blank");
    if (intensFiles.size() != labelFiles.size())
        throw std::runtime_error("Imbalanced intensity and segmentation file lists");
    for (auto i = 0; i < intensFiles.size(); i++)
    {
        const std::string& i_fname = intensFiles[i];
        const std::string& s_fname = labelFiles[i];

        if (!existsOnFilesystem(i_fname))
        {
            auto msg = "File does not exist: " + i_fname;
            throw std::runtime_error(msg);
        }
        if (!existsOnFilesystem(s_fname))
        {
            auto msg = "File does not exist: " + s_fname;
            throw std::runtime_error(msg);
        }
    }

    init_feature_buffers();

    theResultsCache.clear();

    // Process the image sdata
    int min_online_roi_size = 0;
    int errorCode = processDataset(
        intensFiles,
        labelFiles,
        theEnvironment.n_loader_threads,
        theEnvironment.n_pixel_scan_threads,
        theEnvironment.n_reduce_threads,
        min_online_roi_size,
        false, // 'true' to save to csv
        theEnvironment.output_dir,
        use_fastloop);
    if (errorCode)
        throw std::runtime_error("Error occurred during dataset processing.");

    auto pyHeader = py::array(py::cast(theResultsCache.get_headerBuf()));
    auto pyStrData = py::array(py::cast(theResultsCache.get_stringColBuf()));
    auto pyNumData = as_pyarray(std::move(theResultsCache.get_calcResultBuf()));
    auto nRows = theResultsCache.get_num_rows();
    pyStrData = pyStrData.reshape({nRows, pyStrData.size() / nRows});
    pyNumData = pyNumData.reshape({ nRows, pyNumData.size() / nRows });

    return py::make_tuple(pyHeader, pyStrData, pyNumData);
}


py::tuple featurize_fname_lists_imp (const py::list& int_fnames, const py::list & seg_fnames)
{
    return featurize_fname_lists_imp_fast (int_fnames, seg_fnames, false);
}

py::tuple findrelations_imp(
        std::string& label_dir,
        std::string& parent_file_pattern,
        std::string& child_file_pattern
    )
{
    if (! theEnvironment.check_file_pattern(parent_file_pattern) || ! theEnvironment.check_file_pattern(child_file_pattern))
        throw std::invalid_argument("Filepattern provided is not valid.");

    theResultsCache.clear();

    // Result -> headerBuf, stringColBuf, calcResultBuf
    ChildFeatureAggregation aggr;
    bool mineOK = mine_segment_relations (true, label_dir, parent_file_pattern, child_file_pattern, ".", aggr, theEnvironment.get_verbosity_level());  // the 'outdir' parameter is not used if 'output2python' is true

    if (! mineOK)
        throw std::runtime_error("Error occurred during dataset processing: mine_segment_relations() returned false");
    
    auto pyHeader = py::array(py::cast(theResultsCache.get_headerBuf())); // Column names
    auto pyStrData = py::array(py::cast(theResultsCache.get_stringColBuf())); // String cells of first n columns
    auto pyNumData = as_pyarray(std::move(theResultsCache.get_calcResultBuf()));  // Numeric data
    auto nRows = theResultsCache.get_num_rows();
    pyStrData = pyStrData.reshape({ nRows, pyStrData.size() / nRows });
    pyNumData = pyNumData.reshape({ nRows, pyNumData.size() / nRows });

    return py::make_tuple(pyHeader, pyStrData, pyNumData);
}


/**
 * @brief Set whether to use the gpu for available gpu features
 * 
 * @param yes True to use gpu
 */
void use_gpu(bool yes){
    #ifdef USE_GPU
        theEnvironment.set_use_gpu(yes);
    #else 
        std::cout << "GPU is not available." << std::endl;
    #endif
}

/**
 * @brief Get the gpu properties. If gpu is not available, return an empty vector
 * 
 * @return std::vector<std::map<std::string, std::string>> Properties of gpu
 */
static std::vector<std::map<std::string, std::string>> get_gpu_properties() {
    #ifdef USE_GPU
        return theEnvironment.get_gpu_properties();
    #else 
        std::vector<std::map<std::string, std::string>> empty;
        return empty;
    #endif
}


PYBIND11_MODULE(backend, m)
{
    m.doc() = "Nyxus";

    m.def("initialize_environment", &initialize_environment, "Environment initialization");
    m.def("featurize_directory_imp", &featurize_directory_imp, "Calculate features of images defined by intensity and mask image collection directories");
    m.def("featurize_fname_lists_imp", &featurize_fname_lists_imp, "Calculate features of intensity-mask image pairs defined by lists of image file names");
    m.def("featurize_directory_imp_fast", &featurize_directory_imp_fast, "Calculate features of images defined by intensity and mask image collection directories");
    m.def("featurize_fname_lists_imp_fast", &featurize_fname_lists_imp_fast, "Calculate features of intensity-mask image pairs defined by lists of image file names");
    m.def("findrelations_imp", &findrelations_imp, "Find relations in segmentation images");
    m.def("gpu_available", &Environment::gpu_is_available, "Check if CUDA gpu is available");
    m.def("use_gpu", &use_gpu, "Enable/disable GPU features");
    m.def("get_gpu_props", &get_gpu_properties, "Get properties of CUDA gpu");
}

///
/// The following code block is a quick & simple manual test of the Python interface 
/// invokable from from the command line. It lets you bypass building and installing the Python library.
/// To use it, 
///     #define TESTING_PY_INTERFACE, 
///     exclude file main_nyxus.cpp from build, and 
///     rebuild the CLI target.
/// 
#ifdef TESTING_PY_INTERFACE
//
// Testing Python interface
//
void initialize_environment(
    const std::vector<std::string>& features,
    float neighbor_distance,
    float pixels_per_micron,
    uint32_t coarse_gray_depth,
    uint32_t n_reduce_threads,
    uint32_t n_loader_threads);

py::tuple featurize_directory_imp(
    const std::string& intensity_dir,
    const std::string& labels_dir,
    const std::string& file_pattern);

int main(int argc, char** argv)
{
    std::cout << "main() \n";

    // Test feature extraction
    
    //  initialize_environment({ "*ALL*" }, 5, 120, 1, 1);
    //
    //  py::tuple result = featurize_directory_imp(
    //      "C:\\WORK\\AXLE\\data\\mini\\int", // intensity_dir,
    //      "C:\\WORK\\AXLE\\data\\mini\\seg", // const std::string & labels_dir,
    //      "p0_y1_r1_c0\\.ome\\.tif"); // const std::string & file_pattern

    // Test nested segments functionality

    py::tuple result = findrelations_imp(
        "C:\\WORK\\AXLE\\data\\mini\\seg",  // label_dir, 
        ".*", // file_pattern,
        "_c", // channel_signature, 
        "1", // parent_channel, 
        "0"); // child_channel

    std::cout << "finishing \n";
}

#endif

