#include <algorithm>
#include <exception>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../version.h"
#include "../environment.h"
#include "../feature_mgr.h"
#include "../globals.h"

namespace py = pybind11;
using namespace Nyxus;

// Defined in nested.cpp
int mine_segment_relations (const std::string& label_dir, const std::string& file_pattern, const std::string& channel_signature, const int parent_channel, const int child_channel);

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
    uint32_t n_reduce_threads,
    uint32_t n_loader_threads)
{
    theEnvironment.desiredFeatures = features;
    theEnvironment.set_pixel_distance(static_cast<int>(neighbor_distance));
    theEnvironment.verbosity_level = 0;
    theEnvironment.xyRes = theEnvironment.pixelSizeUm = pixels_per_micron;
    theEnvironment.n_reduce_threads = n_reduce_threads;
    theEnvironment.n_loader_threads = n_loader_threads;

    // Throws exception if invalid feature is supplied.
    theEnvironment.process_feature_list();
    theFeatureMgr.compile();
    theFeatureMgr.apply_user_selection();
}

py::tuple process_data(
    const std::string &intensity_dir,
    const std::string &labels_dir,
    const std::string &file_pattern)
{
    theEnvironment.intensity_dir = intensity_dir;
    theEnvironment.labels_dir = labels_dir;
    theEnvironment.file_pattern = file_pattern;

    if (!theEnvironment.check_file_pattern(file_pattern))
        throw std::invalid_argument("Filepattern provided is not valid.");

    std::vector<std::string> intensFiles, labelFiles;
    int errorCode = Nyxus::read_dataset(
        intensity_dir,
        labels_dir,
        "./",
        theEnvironment.intSegMapDir,
        theEnvironment.intSegMapFile,
        true,
        intensFiles, labelFiles);

    if (errorCode)
        throw std::runtime_error("Dataset structure error.");

    init_feature_buffers();

    totalNumFeatures = 0;
    totalNumLabels = 0;
    headerBuf.clear();
    stringColBuf.clear();
    calcResultBuf.clear();

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

    auto pyHeader = py::array(py::cast(headerBuf));
    auto pyStrData = py::array(py::cast(stringColBuf));
    auto pyNumData = as_pyarray(std::move(calcResultBuf));

    pyStrData = pyStrData.reshape({totalNumLabels, pyStrData.size() / totalNumLabels});
    pyNumData = pyNumData.reshape({totalNumLabels, pyNumData.size() / totalNumLabels});

    return py::make_tuple(pyHeader, pyStrData, pyNumData);
}

py::tuple findrelations_imp(
    const std::string& label_dir, 
    const std::string& file_pattern, 
    const std::string& channel_signature, 
    const std::string& parent_channel, 
    const std::string& child_channel)
{
    if (! theEnvironment.check_file_pattern(file_pattern))
        throw std::invalid_argument("Filepattern provided is not valid.");

    // check channel numbers
    int n_parent_channel;
    if (sscanf(parent_channel.c_str(), "%d", &n_parent_channel) != 1)
        throw std::runtime_error("Error parsing the parent channel number");

    int n_child_channel;
    if (sscanf(child_channel.c_str(), "%d", &n_child_channel) != 1)
        throw std::runtime_error("Error parsing the child channel number");

    headerBuf.clear();
    stringColBuf.clear();
    calcResultBuf.clear();
    totalNumLabels = 0;

    // Result -> headerBuf, stringColBuf, calcResultBuf
    int errorCode = mine_segment_relations (label_dir, file_pattern, channel_signature, n_parent_channel, n_child_channel);

    if (errorCode)
        throw std::runtime_error("Error occurred during dataset processing.");

    auto pyHeader = py::array(py::cast(headerBuf)); // Column names
    auto pyStrData = py::array(py::cast(stringColBuf)); // String cells of first n columns
    auto pyNumData = as_pyarray(std::move(calcResultBuf));  // Numeric data

    pyStrData = pyStrData.reshape({ totalNumLabels, pyStrData.size() / totalNumLabels });
    pyNumData = pyNumData.reshape({ totalNumLabels, pyNumData.size() / totalNumLabels });

    return py::make_tuple(pyHeader, pyStrData, pyNumData);
}

PYBIND11_MODULE(backend, m)
{
    m.doc() = "Nyxus";

    m.def("initialize_environment", &initialize_environment, "Environment initialization");
    m.def("process_data", &process_data, "Process images, calculate features");
    m.def("findrelations_imp", &findrelations_imp, "Find relations in segmentation images");
}

#if TESTING_PY_INTERFACE
//
// Testing Python interface
//
void initialize_environment(
    const std::vector<std::string>& features,
    float neighbor_distance,
    float pixels_per_micron,
    uint32_t n_reduce_threads,
    uint32_t n_loader_threads);

py::tuple process_data(
    const std::string& intensity_dir,
    const std::string& labels_dir,
    const std::string& file_pattern);

int main(int argc, char** argv)
{
    std::cout << "main() \n";

    // Test feature extraction
    
    //  initialize_environment({ "*ALL*" }, 5, 120, 1, 1);
    //
    //  py::tuple result = process_data(
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

