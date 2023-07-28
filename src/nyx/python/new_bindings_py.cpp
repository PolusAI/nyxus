#include <algorithm>
#include <exception>
#include <variant>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../version.h"
#include "../environment.h"
#include "../feature_mgr.h"
#include "../dirs_and_files.h"   
#include "../globals.h"
#include "../nested_feature_aggregation.h"
#include "../features/gabor.h"

#ifdef USE_ARROW
    #include "../output_writers.h" 

    #include "../arrow_output.h"

    #include <arrow/python/pyarrow.h>
    #include <arrow/table.h>

    #include <arrow/python/platform.h>

    #include <arrow/python/datetime.h>
    #include <arrow/python/init.h>
    #include <arrow/python/pyarrow.h>

    #include "table_caster.h"
#endif

namespace py = pybind11;
using namespace Nyxus;

using ParameterTypes = std::variant<int, float, double, unsigned int, std::vector<double>, std::vector<std::string>>;

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
    int neighbor_distance,
    float pixels_per_micron,
    uint32_t coarse_gray_depth, 
    uint32_t n_reduce_threads,
    uint32_t n_loader_threads,
    int using_gpu,
    bool ibsi)
{
    theEnvironment.recognizedFeatureNames = features;
    theEnvironment.set_pixel_distance(neighbor_distance);
    theEnvironment.set_verbosity_level (0);
    theEnvironment.xyRes = theEnvironment.pixelSizeUm = pixels_per_micron;
    theEnvironment.set_coarse_gray_depth(coarse_gray_depth);
    theEnvironment.n_reduce_threads = n_reduce_threads;
    theEnvironment.n_loader_threads = n_loader_threads;
    theEnvironment.ibsi_compliance = ibsi;

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

void set_if_ibsi_imp(bool ibsi) {
    theEnvironment.set_ibsi_compliance(ibsi);
}

void set_environment_params_imp (
    const std::vector<std::string> &features = {},
    int neighbor_distance = -1,
    float pixels_per_micron = -1,
    uint32_t coarse_gray_depth = 0, 
    uint32_t n_reduce_threads = 0,
    uint32_t n_loader_threads = 0,
    int using_gpu = -2
) {
    if (features.size() > 0) {
        theEnvironment.recognizedFeatureNames = features;
    }
     
    if (neighbor_distance > -1) {
        theEnvironment.set_pixel_distance(neighbor_distance);
    }

    if (pixels_per_micron > -1) {
        theEnvironment.xyRes = theEnvironment.pixelSizeUm = pixels_per_micron;
    }

    if (coarse_gray_depth != 0) {
        theEnvironment.set_coarse_gray_depth(coarse_gray_depth);
    }

    if (n_reduce_threads != 0) {
        theEnvironment.n_reduce_threads = n_reduce_threads;
    }
    
    if (n_loader_threads != 0) {
        theEnvironment.n_loader_threads = n_loader_threads;
    }
}

py::tuple featurize_directory_imp (
    const std::string &intensity_dir,
    const std::string &labels_dir,
    const std::string &file_pattern,
    bool pandas_output=true)
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
        theEnvironment.output_dir);

    if (errorCode)
        throw std::runtime_error("Error occurred during dataset processing.");

    if (pandas_output) {

    #ifdef USE_ARROW
        // Get by value to preserve buffers for writing to arrow
        auto pyHeader = py::array(py::cast(theResultsCache.get_headerBufByVal()));
        auto pyStrData = py::array(py::cast(theResultsCache.get_stringColBufByVal()));
        auto pyNumData = as_pyarray(std::move(theResultsCache.get_calcResultBufByVal()));
    #else 
        auto pyHeader = py::array(py::cast(theResultsCache.get_headerBuf()));
        auto pyStrData = py::array(py::cast(theResultsCache.get_stringColBuf()));
        auto pyNumData = as_pyarray(std::move(theResultsCache.get_calcResultBuf()));
    #endif
        auto nRows = theResultsCache.get_num_rows();
        pyStrData = pyStrData.reshape({nRows, pyStrData.size() / nRows});
        pyNumData = pyNumData.reshape({ nRows, pyNumData.size() / nRows });

        return py::make_tuple(pyHeader, pyStrData, pyNumData);
    
    } 

    // Return "nothing" when output will be an Arrow format
    return py::make_tuple();
}

py::tuple featurize_montage_imp (
    const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intensity_images,
    const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_images,
    const std::vector<std::string>& intensity_names,
    const std::vector<std::string>& label_names,
    bool pandas_output=true)
{  
    
    auto intens_buffer = intensity_images.request();
    auto label_buffer = label_images.request();

    auto width = intens_buffer.shape[1];
    auto height = intens_buffer.shape[2];

    auto nf = intens_buffer.shape[0];

    auto label_width = intens_buffer.shape[1];
    auto label_height = intens_buffer.shape[2];

    auto label_nf = intens_buffer.shape[0];


    if(nf != label_nf) {
         throw std::invalid_argument("The number of intensity (" + std::to_string(nf) + ") and label (" + std::to_string(label_nf) + ") images must be the same.");
    }

    if(width != label_width || height != label_height) {
        throw std::invalid_argument("Intensity (width " + std::to_string(width) + ", height " + std::to_string(height) + ") and label (width " + std::to_string(label_width) + ", height " + std::to_string(label_height) + ") image size mismatch");
    }

    theEnvironment.intensity_dir = "__NONE__";
    theEnvironment.labels_dir = "__NONE__";

    // One-time initialization
    init_feature_buffers();

    theResultsCache.clear();

    // Process the image sdata
    std::string error_message = "";
    int errorCode = processMontage(
        intensity_images,
        label_images,
        theEnvironment.n_reduce_threads,
        intensity_names,
        label_names,
        error_message);

    if (errorCode)
        throw std::runtime_error("Error #" + std::to_string(errorCode) + " " + error_message + " occurred during dataset processing.");

    if (pandas_output) {

        #ifdef USE_ARROW
            // Get by value to preserve buffers for writing to arrow
            auto pyHeader = py::array(py::cast(theResultsCache.get_headerBufByVal()));
            auto pyStrData = py::array(py::cast(theResultsCache.get_stringColBufByVal()));
            auto pyNumData = as_pyarray(std::move(theResultsCache.get_calcResultBufByVal()));
        #else 
            auto pyHeader = py::array(py::cast(theResultsCache.get_headerBuf()));
            auto pyStrData = py::array(py::cast(theResultsCache.get_stringColBuf()));
            auto pyNumData = as_pyarray(std::move(theResultsCache.get_calcResultBuf()));
        #endif
            auto nRows = theResultsCache.get_num_rows();
            pyStrData = pyStrData.reshape({nRows, pyStrData.size() / nRows});
            pyNumData = pyNumData.reshape({ nRows, pyNumData.size() / nRows });

        return py::make_tuple(pyHeader, pyStrData, pyNumData, error_message);
    
    } 

    // Return "nothing" when output will be an Arrow format
    return py::make_tuple(error_message);
}

py::tuple featurize_fname_lists_imp (const py::list& int_fnames, const py::list & seg_fnames, bool pandas_output=true)
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
        theEnvironment.output_dir);
    if (errorCode)
        throw std::runtime_error("Error occurred during dataset processing.");

    if (pandas_output) {

        #ifdef USE_ARROW
            // Get by value to preserve buffers for writing to arrow
            auto pyHeader = py::array(py::cast(theResultsCache.get_headerBufByVal()));
            auto pyStrData = py::array(py::cast(theResultsCache.get_stringColBufByVal()));
            auto pyNumData = as_pyarray(std::move(theResultsCache.get_calcResultBufByVal()));
        #else 
            auto pyHeader = py::array(py::cast(theResultsCache.get_headerBuf()));
            auto pyStrData = py::array(py::cast(theResultsCache.get_stringColBuf()));
            auto pyNumData = as_pyarray(std::move(theResultsCache.get_calcResultBuf()));
        #endif
            auto nRows = theResultsCache.get_num_rows();
            pyStrData = pyStrData.reshape({nRows, pyStrData.size() / nRows});
            pyNumData = pyNumData.reshape({ nRows, pyNumData.size() / nRows });

            return py::make_tuple(pyHeader, pyStrData, pyNumData);
        
    } 

    // Return "nothing" when output will be an Arrow format
    return py::make_tuple();
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
    
#ifdef USE_ARROW
    // Get by value to preserve buffers for writing to arrow
    auto pyHeader = py::array(py::cast(theResultsCache.get_headerBufByVal()));
    auto pyStrData = py::array(py::cast(theResultsCache.get_stringColBufByVal()));
    auto pyNumData = as_pyarray(std::move(theResultsCache.get_calcResultBufByVal()));
#else 
    auto pyHeader = py::array(py::cast(theResultsCache.get_headerBuf()));
    auto pyStrData = py::array(py::cast(theResultsCache.get_stringColBuf()));
    auto pyNumData = as_pyarray(std::move(theResultsCache.get_calcResultBuf()));
#endif
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

void blacklist_roi_imp (std::string raw_blacklist)
{
    // After successfully parsing the blacklist, Nyxus runtime becomes able 
    // to skip blacklisted ROIs until the cached blacklist is cleared 
    // with Environment::clear_roi_blacklist()

    std::string lastError;
    if (! theEnvironment.parse_roi_blacklist_raw_string (raw_blacklist, lastError))
    {
        std::string ermsg = "Error parsing ROI blacklist definition: " + lastError;
        throw std::runtime_error(ermsg);
    }
}

void clear_roi_blacklist_imp()
{
    theEnvironment.clear_roi_blacklist();
}

py::str roi_blacklist_get_summary_imp()
{
    std::string response;
    theEnvironment.get_roi_blacklist_summary(response);
    return py::str(response);
}

void customize_gabor_feature_imp(
    const std::string& kersize,
    const std::string& gamma,
    const std::string& sig2lam,
    const std::string& f0,
    const std::string& theta,
    const std::string& thold,
    const std::string& freqs)
{

    // Step 1 - set raw strings of parameter values
    theEnvironment.gaborOptions.rawKerSize = kersize;
    theEnvironment.gaborOptions.rawGamma = gamma;
    theEnvironment.gaborOptions.rawSig2lam = sig2lam;
    theEnvironment.gaborOptions.rawF0 = f0;
    theEnvironment.gaborOptions.rawTheta = theta;
    theEnvironment.gaborOptions.rawGrayThreshold = thold;
    theEnvironment.gaborOptions.rawFreqs = freqs;

    // Step 2 - validate them and consume if all are valid
    std::string ermsg;
    if (!theEnvironment.parse_gabor_options_raw_inputs(ermsg))
        throw std::invalid_argument("Invalid GABOR parameter value: " + ermsg);
}

std::map<std::string, ParameterTypes> get_params_imp(const std::vector<std::string>& vars ) {
    std::map<std::string, ParameterTypes> params;

    params["features"] = theEnvironment.recognizedFeatureNames;
    params["neighbor_distance"] = theEnvironment.n_pixel_distance;
    params["pixels_per_micron"] = theEnvironment.xyRes;
    params["coarse_gray_depth"] = theEnvironment.get_coarse_gray_depth();
    params["n_feature_calc_threads"] = theEnvironment.n_reduce_threads;
    params["n_loader_threads"] = theEnvironment.n_loader_threads;
    params["ibsi"] = theEnvironment.ibsi_compliance;

    params["gabor_kersize"] = GaborFeature::n;
    params["gabor_gamma"] = GaborFeature::gamma;
    params["gabor_sig2lam"] = GaborFeature::sig2lam;
    params["gabor_f0"] = GaborFeature::f0LP;
    params["gabor_theta"] = GaborFeature::get_theta_in_degrees(); // convert theta back from radians
    params["gabor_thold"] = GaborFeature::GRAYthr;
    params["gabor_freqs"] = GaborFeature::f0;

    if (vars.size() == 0) 
        return params;

    std::map<std::string, ParameterTypes> params_subset;

    for (const auto& var: vars) {

        auto it = params.find(var);

        if (it != params.end()) {
            params_subset.insert(*it);
        }
    }

    return params_subset;

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
    int neighbor_distance,
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


void create_arrow_file_imp(const std::string& arrow_file_path="") {

#ifdef USE_ARROW

    return theEnvironment.arrow_output.create_arrow_file(theResultsCache.get_headerBuf(),
                                          theResultsCache.get_stringColBuf(),
                                          theResultsCache.get_calcResultBuf(),
                                          theResultsCache.get_num_rows(),
                                          arrow_file_path);

#else
    
    throw std::runtime_error("Arrow functionality is not available. Rebuild Nyxus with Arrow enabled.");

#endif

}

std::string get_arrow_file_imp() {
#ifdef USE_ARROW

    return theEnvironment.arrow_output.get_arrow_file();

#else
    
    throw std::runtime_error("Arrow functionality is not available. Rebuild Nyxus with Arrow enabled.");

#endif
}

void create_parquet_file_imp(std::string& parquet_file_path) {

#ifdef USE_ARROW

    return theEnvironment.arrow_output.create_parquet_file(theResultsCache.get_headerBuf(),
                                            theResultsCache.get_stringColBuf(),
                                            theResultsCache.get_calcResultBuf(),
                                            theResultsCache.get_num_rows(),
                                            parquet_file_path);

#else
    
    throw std::runtime_error("Arrow functionality is not available. Rebuild Nyxus with Arrow enabled.");

#endif
}

std::string get_parquet_file_imp() {

#ifdef USE_ARROW

    return theEnvironment.arrow_output.get_parquet_file();

#else
    
    throw std::runtime_error("Arrow functionality is not available. Rebuild Nyxus with Arrow enabled.");

#endif
}

#ifdef USEARROW

std::shared_ptr<arrow::Table> get_arrow_table_imp() {

    return theEnvironment.arrow_output.get_arrow_table(theResultsCache.get_headerBuf(),
                                                       theResultsCache.get_stringColBuf(),
                                                       theResultsCache.get_calcResultBuf(),
                                                       theResultsCache.get_num_rows());
}

#else

void get_arrow_table_imp() {
    throw std::runtime_error("Arrow functionality is not available. Rebuild Nyxus with Arrow enabled.");
}

#endif

bool arrow_is_enabled_imp() {
    return theEnvironment.arrow_is_enabled();
}

PYBIND11_MODULE(backend, m)
{

#ifdef USE_ARROW
    Py_Initialize();

    int success = arrow::py::import_pyarrow();

    if (success != 0) {
        throw std::runtime_error("Error initializing pyarrow.");
    } 
#endif
    m.doc() = "Nyxus";
    
    m.def("initialize_environment", &initialize_environment, "Environment initialization");
    m.def("featurize_directory_imp", &featurize_directory_imp, "Calculate features of images defined by intensity and mask image collection directories");
    m.def("featurize_montage_imp", &featurize_montage_imp, "Calculate features of images defined by intensity and mask image collection directories");
    m.def("featurize_fname_lists_imp", &featurize_fname_lists_imp, "Calculate features of intensity-mask image pairs defined by lists of image file names");
    m.def("findrelations_imp", &findrelations_imp, "Find relations in segmentation images");
    m.def("gpu_available", &Environment::gpu_is_available, "Check if CUDA gpu is available");
    m.def("use_gpu", &use_gpu, "Enable/disable GPU features");
    m.def("get_gpu_props", &get_gpu_properties, "Get properties of CUDA gpu");
    m.def("blacklist_roi_imp", &blacklist_roi_imp, "Set up a global or per-mask file blacklist definition");
    m.def("clear_roi_blacklist_imp", &clear_roi_blacklist_imp, "Clear the ROI black list");
    m.def("roi_blacklist_get_summary_imp", &roi_blacklist_get_summary_imp, "Returns a summary of the ROI blacklist");
    m.def("customize_gabor_feature_imp", &customize_gabor_feature_imp, "Sets custom GABOR feature's parameters");
    m.def("set_if_ibsi_imp", &set_if_ibsi_imp, "Set if the features will be ibsi compliant");
    m.def("set_environment_params_imp", &set_environment_params_imp, "Set the environment variables of Nyxus");
    m.def("get_params_imp", &get_params_imp, "Get parameters of Nyxus");
    m.def("create_arrow_file_imp", &create_arrow_file_imp, "Creates an arrow file for the feature calculations");
    m.def("get_arrow_file_imp", &get_arrow_file_imp, "Get path to arrow file");
    m.def("get_parquet_file_imp", &get_parquet_file_imp, "Returns path to parquet file");
    m.def("create_parquet_file_imp", &create_parquet_file_imp, "Create parquet file for the features calculations");
    m.def("get_arrow_table_imp", &get_arrow_table_imp, py::call_guard<py::gil_scoped_release>());
    m.def("arrow_is_enabled_imp", &arrow_is_enabled_imp, "Check if arrow is enabled.");
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
    int neighbor_distance,
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

