#include <algorithm>
#include <exception>
#include <variant>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../version.h"
#include "../dirs_and_files.h"   
#include "../environment.h"
#include "../feature_mgr.h"
#include "../globals.h"
#include "../nested_feature_aggregation.h"
#include "../features/gabor.h"
#include "../output_writers.h" 
#include "../arrow_output_stream.h"
#include "../strpat.h"

namespace py = pybind11;
using namespace Nyxus;

namespace Nyxus {

    std::unordered_set<uint64_t> unique_pynyxus_ids;
    std::unordered_map<uint64_t, Environment> pynyxus_cache;

    Environment & findenv (uint64_t instid)
    {
        // create if missing
        auto [it, inserted] = Nyxus::pynyxus_cache.try_emplace(instid);
        // find
        Environment& env = Nyxus::pynyxus_cache [instid];
        return env;
    }

};

using ParameterTypes = std::variant<int, float, double, unsigned int, std::string, std::vector<double>, std::vector<std::string>>;

// Defined in nested.cpp
bool mine_segment_relations (
    ResultsCache & res_cache,
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
    // Copy data into a numpy-owned array rather than wrapping the C++ memory
    // with a pybind11 capsule. A capsule's destructor is compiled code inside
    // this .so — if Python's GC frees the array after the .so is unloaded
    // (e.g. during interpreter shutdown), the destructor call jumps to
    // unmapped memory and segfaults. Copying avoids that by letting numpy
    // manage the memory entirely with no reference back to this module.
    using T = typename Sequence::value_type;
    py::array_t<T> arr(seq.size());
    std::copy(seq.begin(), seq.end(), arr.mutable_data());
    return arr;
}

/// @brief Converts accumulated FmapArrayResult entries into a Python list of dicts.
/// Each dict contains: parent_roi_label, intensity_image, mask_image, origin_x, origin_y,
/// and a 'features' dict mapping feature names to numpy arrays.
/// For 2D: arrays are (map_h, map_w). For 3D: arrays are (map_d, map_h, map_w).
py::list fmap_results_to_python(ResultsCache & rescache)
{
    py::list result;
    for (auto & fr : rescache.get_fmapArrayResults())
    {
        py::dict d;
        d["parent_roi_label"] = fr.parent_label;
        d["intensity_image"] = fr.intens_name;
        d["mask_image"] = fr.seg_name;
        d["origin_x"] = fr.origin_x;
        d["origin_y"] = fr.origin_y;

        bool is_3d = fr.map_d > 1;
        if (is_3d)
            d["origin_z"] = fr.origin_z;

        size_t map_size = (size_t)fr.map_d * fr.map_h * fr.map_w;
        size_t n_features = fr.feature_names.size();

        py::dict features;
        for (size_t fi = 0; fi < n_features; fi++)
        {
            py::array_t<double> arr;
            if (is_3d)
                arr = py::array_t<double>({fr.map_d, fr.map_h, fr.map_w});
            else
                arr = py::array_t<double>({fr.map_h, fr.map_w});
            auto ptr = arr.mutable_data();
            std::copy(
                fr.feature_data.begin() + fi * map_size,
                fr.feature_data.begin() + (fi + 1) * map_size,
                ptr);
            features[py::str(fr.feature_names[fi])] = arr;
        }
        d["features"] = features;
        result.append(d);
    }
    return result;
}

void initialize_environment(
    uint64_t instid,
    int n_dim,
    const std::vector<std::string> &features,
    int neighbor_distance,
    float pixels_per_micron,
    int coarse_gray_depth,
    uint32_t n_reduce_threads,
    int using_gpu,
    bool ibsi,
    float dynamic_range,
    float min_intensity,
    float max_intensity,
    bool is_imq,
    int ram_limit_mb,
    int verb_lvl,
    float aniso_x,
    float aniso_y,
    float aniso_z,
    bool fmaps = false,
    int fmaps_radius = 2)
{
    Environment & theEnvironment = Nyxus::findenv (instid);

    theEnvironment.set_imq(is_imq);
    theEnvironment.set_dim(n_dim);
    theEnvironment.recognizedFeatureNames = features;
    theEnvironment.set_pixel_distance(neighbor_distance);
    theEnvironment.set_verbosity_level (verb_lvl);
    theEnvironment.xyRes = theEnvironment.pixelSizeUm = pixels_per_micron;
    theEnvironment.set_coarse_gray_depth(coarse_gray_depth);
    theEnvironment.n_reduce_threads = n_reduce_threads;
    theEnvironment.ibsi_compliance = ibsi;

    // feature maps
    theEnvironment.fmaps_mode = fmaps;
    if (fmaps_radius >= 1)
        theEnvironment.fmaps_kernel_radius = fmaps_radius;

    // Throws exception if invalid feature is passed
    theEnvironment.expand_featuregroups();
    if (!theEnvironment.theFeatureMgr.compile())
        throw std::runtime_error ("Error: compiling feature methods");
    theEnvironment.theFeatureMgr.apply_user_selection (theEnvironment.theFeatureSet);

    // prepare feature settings
    theEnvironment.compile_feature_settings();

    // real-valued range hints
    theEnvironment.fpimageOptions.set_target_dyn_range(dynamic_range);
    theEnvironment.fpimageOptions.set_min_intensity(min_intensity);
    theEnvironment.fpimageOptions.set_max_intensity(max_intensity);

    // RAM limit controlling trivial-nontrivial featurization
    if (ram_limit_mb >= 0)
        theEnvironment.set_ram_limit(ram_limit_mb);

    // anisotropy
    theEnvironment.anisoOptions.set_aniso_x (aniso_x);
    theEnvironment.anisoOptions.set_aniso_y (aniso_y);
    theEnvironment.anisoOptions.set_aniso_z (aniso_z);

    // GPU related
    #ifdef USE_GPU
        if(using_gpu == -1)
        {
            theEnvironment.set_using_gpu(false);
        }
        else
        {
            theEnvironment.set_gpu_device_id(using_gpu);
            theEnvironment.set_using_gpu(true);
        }
    #else
        if (using_gpu != -1)
        {
            throw std::runtime_error ("this Nyxus backend was built without the GPU support");
        }
    #endif

}

void set_if_ibsi_imp (uint64_t instid, bool ibsi)
{
    Environment & env = Nyxus::findenv (instid);
    env.set_ibsi_compliance (ibsi);
}

void set_fmaps_imp (uint64_t instid, int set_mode, int radius)
{
    Environment & env = Nyxus::findenv (instid);
    // Always validate radius when explicitly provided (not sentinel -1),
    // even if fmaps_mode is false, to prevent stale invalid values
    if (radius != -1)
    {
        if (radius >= 1)
            env.fmaps_kernel_radius = radius;
        else
            throw std::invalid_argument("fmaps_radius must be an integer >= 1");
    }
    // set_mode: 0=disable, 1=enable, -1=leave unchanged (radius-only update)
    if (set_mode >= 0)
        env.fmaps_mode = (set_mode != 0);
}

void set_environment_params_imp (
    uint64_t instid,
    const std::vector<std::string> &features = {},
    int neighbor_distance = -1,
    float pixels_per_micron = -1,
    int coarse_gray_depth = 0,
    uint32_t n_reduce_threads = 0,
    int using_gpu = -2,
    float dynamic_range = -1,
    float min_intensity = -1,
    float max_intensity = -1,
    int ram_limit_mb = -1,
    int verb_level = 0)
{
    Environment & theEnvironment = Nyxus::findenv (instid);

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
    
    if (dynamic_range >= 0) {
        theEnvironment.fpimageOptions.set_target_dyn_range(dynamic_range);
    }

    if (min_intensity >= 0) {
        theEnvironment.fpimageOptions.set_min_intensity(min_intensity);
    }

    if (max_intensity >= 0) {
        theEnvironment.fpimageOptions.set_max_intensity(max_intensity);
    }

    if (verb_level >= 0) 
    {
        theEnvironment.set_verbosity_level (verb_level);
    } 
    else 
    {
        std::cerr << "Error: verbosity (" + std::to_string(verb_level) + ") should be a non-negative value" << std::endl;
    }

    if (ram_limit_mb >= 0) {
        auto success = theEnvironment.set_ram_limit(ram_limit_mb);
    }
}

py::tuple featurize_directory_imp(
    uint64_t instid,
    const std::string& intensity_dir,
    const std::string& labels_dir,
    const std::string& file_pattern,
    const std::string& output_type,
    const std::string& output_path = "")
{
    Environment& env = Nyxus::findenv (instid);

    // user's input
    env.separateCsv = false;

    // (file pattern)
    if (! env.check_2d_file_pattern(file_pattern))
        throw std::invalid_argument ("Invalid filepattern " + file_pattern);
    env.set_file_pattern (file_pattern);

    // (directories)
    env.intensity_dir = intensity_dir;
    env.labels_dir = labels_dir;

    // (whole-slide flag)
    env.singleROI = (intensity_dir == labels_dir);

    // read the dataset directory (file names, file pairs)
    std::vector<std::string> intensFiles, labelFiles;
    std::optional<std::string> mayBerror = Nyxus::read_2D_dataset(
        intensity_dir,
        labels_dir,
        env.get_file_pattern(),
        "./",   // output directory
        env.intSegMapDir,
        env.intSegMapFile,
        true,
        intensFiles,
        labelFiles);

    if (mayBerror.has_value())
        throw std::runtime_error("Error reading the dataset: " + *mayBerror);

    // prepare the output objects
    env.theResultsCache.clear();

    env.saveOption = [&output_type]()
    {
        if (output_type == "arrowipc")
        {
            return SaveOption::saveArrowIPC;
        }
        else
            if (output_type == "parquet")
            {
                return SaveOption::saveParquet;
            }
            else
            {
                return SaveOption::saveBuffer;
            }
    }();

    // run the workflow
    int ercode = 0;
    if (env.fmaps_prevents_arrow())
        throw std::runtime_error("Arrow/Parquet output is not supported in feature maps (fmaps) mode. Use CSV or buffer output instead.");

    if (env.fmaps_mode)
        ercode = processDataset_2D_fmaps(
            env,
            intensFiles,
            labelFiles,
            env.n_reduce_threads,
            env.saveOption,
            output_path);
    else if (env.singleROI)
        ercode = processDataset_2D_wholeslide(
            env,
            intensFiles,
            labelFiles,
            env.n_reduce_threads,
            env.saveOption,
            output_path);
    else
        ercode = processDataset_2D_segmented(
            env,
            intensFiles,
            labelFiles,
            env.n_reduce_threads,
            env.saveOption,
            output_path);

    if (ercode)
        throw std::runtime_error("Error: " + std::to_string(ercode));

    // Fmaps mode returns spatial arrays, not a DataFrame
    if (env.fmaps_mode && env.saveOption == Nyxus::SaveOption::saveBuffer)
    {
        auto fmaps = fmap_results_to_python(env.theResultsCache);
        return py::make_tuple(fmaps);
    }

    // shape the resulting data frame

    if (env.saveOption == Nyxus::SaveOption::saveBuffer)
    {
        // has the backend produced any result ?
        auto nRows = env.theResultsCache.get_num_rows();
        if (nRows == 0)
        {
            VERBOSLVL2 (env.get_verbosity_level(), std::cerr << "\nfeaturize_directory_imp(): returning a blank tuple\n");

            // return a blank dataframe
            std::vector<std::string> h({ "column1", "column2", "column3", "column4" });
            std::vector<std::string> s({ "blank", "blank" });
            std::vector<double> n({ 0, 0 });

            pybind11::array pyH = py::array(py::cast(h));
            pybind11::array pySD = py::array(py::cast(s));
            pybind11::array pyND = as_pyarray(std::move(n));

            size_t nr = 1;
            pySD = pySD.reshape({ nr, pySD.size() / nr });
            pyND = pyND.reshape({ nr, pyND.size() / nr });
            return py::make_tuple(pyH, pySD, pyND);
        }

        // we have informative result, package it
        auto pyHeader = py::array(py::cast(env.theResultsCache.get_headerBuf()));
        auto pyStrData = py::array(py::cast(env.theResultsCache.get_stringColBuf()));
        auto pyNumData = as_pyarray(std::move(env.theResultsCache.get_calcResultBuf()));

        // - shape the user-facing dataframe
        pyStrData = pyStrData.reshape({ nRows, pyStrData.size() / nRows });
        pyNumData = pyNumData.reshape({ nRows, pyNumData.size() / nRows });
        return py::make_tuple(pyHeader, pyStrData, pyNumData);
    }

    return py::make_tuple();
    
    //
    //






    // Process the image sdata




    // Output the result

}

py::tuple featurize_directory_imq_imp (
    uint64_t instid,
    const std::string &intensity_dir,
    const std::string &labels_dir,
    const std::string &file_pattern,
    const std::string &output_type,
    const std::string &output_path="")
{
    Environment& theEnvironment = Nyxus::findenv (instid);

    // Check and cache the file pattern
    if (! theEnvironment.check_2d_file_pattern(file_pattern))
        throw std::invalid_argument ("Invalid filepattern " + file_pattern);
    theEnvironment.set_file_pattern(file_pattern);

    // Cache the directories
    theEnvironment.intensity_dir = intensity_dir;
    theEnvironment.labels_dir = labels_dir;

    // Set the whole-slide/multi-ROI flag
    theEnvironment.singleROI = intensity_dir == labels_dir;

    // Read image pairs from the intensity and label directories applying the filepattern
    std::vector<std::string> intensFiles, labelFiles;
    std::string ermsg;
    auto mayBerror = Nyxus::read_2D_dataset(
        intensity_dir,
        labels_dir,
        theEnvironment.get_file_pattern(), 
        "./",   // output directory
        theEnvironment.intSegMapDir,
        theEnvironment.intSegMapFile,
        true,
        intensFiles, labelFiles);

    if (mayBerror.has_value())
       throw std::runtime_error ("Error traversing the dataset: " + *mayBerror);

    // We're good to extract features. Reset the feature results cache
    theEnvironment.theResultsCache.clear();

    theEnvironment.separateCsv = false;

    // Process the image sdata
    int min_online_roi_size = 0;

    theEnvironment.saveOption = [&output_type](){
        if (output_type == "arrowipc") {
            return SaveOption::saveArrowIPC;
	    } else if (output_type == "parquet") {
            return SaveOption::saveParquet;
        } else {return SaveOption::saveBuffer;}
	}();

    int errorCode = processDataset_2D_segmented (
        theEnvironment,
        intensFiles,
        labelFiles,
        theEnvironment.n_reduce_threads,
        theEnvironment.saveOption,
        output_path);

    if (errorCode)
        throw std::runtime_error("Error " + std::to_string(errorCode) + " occurred during dataset processing");

    // Output the result
    if (theEnvironment.saveOption == Nyxus::SaveOption::saveBuffer)
    {
        auto pyHeader = py::array(py::cast(theEnvironment.theResultsCache.get_headerBuf()));
        auto pyStrData = py::array(py::cast(theEnvironment.theResultsCache.get_stringColBuf()));
        auto pyNumData = as_pyarray(std::move(theEnvironment.theResultsCache.get_calcResultBuf()));

        // Shape the user-facing dataframe
        auto nRows = theEnvironment.theResultsCache.get_num_rows();
        pyStrData = pyStrData.reshape ({nRows, pyStrData.size() / nRows});
        pyNumData = pyNumData.reshape ({ nRows, pyNumData.size() / nRows });

        return py::make_tuple (pyHeader, pyStrData, pyNumData);
    } 

    return py::make_tuple();
}

py::tuple featurize_directory_3D_imp(
    uint64_t instid,
    const std::string& intensity_dir,
    const std::string& labels_dir,
    const std::string& file_pattern,
    const std::string& output_type,
    const std::string& output_path = "")
{
    Environment& theEnvironment = Nyxus::findenv (instid);

    // Set dimensionality =3 to let all the modules know the context
    theEnvironment.set_dim(3);
    
    // Check and cache the file pattern
    std::string ermsg;
    if (!theEnvironment.check_3d_file_pattern(file_pattern))
        throw std::invalid_argument("Invalid file pattern " + file_pattern + " : " + theEnvironment.file_pattern_3D.get_ermsg());

    // No need to set the raw file pattern separately for 3D
    //      theEnvironment.set_file_pattern(file_pattern);

    // Cache the directories
    theEnvironment.intensity_dir = intensity_dir;
    theEnvironment.labels_dir = labels_dir;

    // Set the whole-slide/multi-ROI flag
    theEnvironment.singleROI = intensity_dir == labels_dir;

    // We're good to extract features. Reset the feature results cache
    theEnvironment.theResultsCache.clear();

    // Enforce flag 'separateCsv' to be FALSE to prevent flushing intermediate results after each input image. We're going to return the result in a buffer, not leave file(s)
    theEnvironment.separateCsv = false;

    // Process the image sdata

    theEnvironment.saveOption = [&output_type]() {
        if (output_type == "arrowipc") {
            return SaveOption::saveArrowIPC;
        }
        else if (output_type == "parquet") {
            return SaveOption::saveParquet;
        }
        else { return SaveOption::saveBuffer; }
    }();

    if (theEnvironment.singleROI)
    {
        std::vector<std::string> ifiles;

        std::optional<std::string> mayBerror = Nyxus::read_3D_dataset_wholevolume(
            theEnvironment.intensity_dir,
            theEnvironment.file_pattern_3D,
            "./", // theEnvironment.output_dir,
            ifiles);
        if (mayBerror.has_value())
            throw std::runtime_error ("error reading whole volume dataset " + theEnvironment.intensity_dir + " , error " + *mayBerror);

        auto [ok, erm] = processDataset_3D_wholevolume(
            theEnvironment,
            ifiles,
            theEnvironment.n_reduce_threads,
            theEnvironment.saveOption,
            output_path);
        if (!ok)
            throw std::runtime_error (*erm);
    }
    else
    {
        // Read image pairs from the intensity and label directories applying the filepattern
        std::vector <Imgfile3D_layoutA> intensFiles, labelFiles;
        std::optional<std::string> mayBerror= Nyxus::read_3D_dataset(
            intensity_dir,
            labels_dir,
            theEnvironment.file_pattern_3D,
            "./",   // output directory
            theEnvironment.intSegMapDir,
            theEnvironment.intSegMapFile,
            true,
            intensFiles, labelFiles);

        if (mayBerror.has_value())
            throw std::runtime_error ("Error traversing dataset: " + *mayBerror);

        int errorCode = 0;
        if (theEnvironment.fmaps_mode)
        {
            errorCode = processDataset_3D_fmaps(
                theEnvironment,
                intensFiles,
                labelFiles,
                theEnvironment.n_reduce_threads,
                theEnvironment.saveOption,
                output_path);
        }
        else
        {
            errorCode = processDataset_3D_segmented(
                theEnvironment,
                intensFiles,
                labelFiles,
                theEnvironment.n_reduce_threads,
                theEnvironment.saveOption,
                output_path);
        }

        if (errorCode)
            throw std::runtime_error ("Error " + std::to_string(errorCode) + " occurred during dataset processing");
    }

    // Output the result
    if (theEnvironment.fmaps_mode && theEnvironment.saveOption == Nyxus::SaveOption::saveBuffer)
    {
        py::list fmaps = fmap_results_to_python(theEnvironment.theResultsCache);
        return py::make_tuple(fmaps);
    }

    if (theEnvironment.saveOption == Nyxus::SaveOption::saveBuffer)
    {
        auto pyHeader = py::array(py::cast(theEnvironment.theResultsCache.get_headerBuf()));
        auto pyStrData = py::array(py::cast(theEnvironment.theResultsCache.get_stringColBuf()));
        auto pyNumData = as_pyarray(std::move(theEnvironment.theResultsCache.get_calcResultBuf()));

        // Shape the user-facing dataframe
        auto nRows = theEnvironment.theResultsCache.get_num_rows();
        pyStrData = pyStrData.reshape({ nRows, pyStrData.size() / nRows });
        pyNumData = pyNumData.reshape({ nRows, pyNumData.size() / nRows });
        return py::make_tuple(pyHeader, pyStrData, pyNumData);
    }

    return py::make_tuple();
}

py::tuple featurize_montage_imp (
    uint64_t instid,
    const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& intensity_images,
    const py::array_t<unsigned int, py::array::c_style | py::array::forcecast>& label_images,
    const std::vector<std::string>& intensity_names,
    const std::vector<std::string>& label_names,
    const std::string output_type="",
    const std::string output_path="")
{  
    Environment& theEnvironment = Nyxus::findenv (instid);

    // Set the whole-slide/multi-ROI flag
    theEnvironment.singleROI = false;

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
    init_slide_rois (theEnvironment.uniqueLabels, theEnvironment.roiData);

    theEnvironment.theResultsCache.clear();

    // Process the image sdata
    std::string error_message = "";

    theEnvironment.saveOption = [&output_type](){
        if (output_type == "arrowipc") {
            return SaveOption::saveArrowIPC;
	    } else if (output_type == "parquet") {
			return SaveOption::saveParquet;
		} else {return SaveOption::saveBuffer;}
	}();

    std::optional<std::string> mayBerror = processMontage(
        theEnvironment,
        intensity_images,
        label_images,
        theEnvironment.n_reduce_threads,
        intensity_names,
        label_names,
        theEnvironment.saveOption,
        output_path);

    if (mayBerror.has_value())
        throw std::runtime_error ("Error occurred during dataset processing: " + *mayBerror);

    if (theEnvironment.fmaps_mode)
    {
        auto fmaps = fmap_results_to_python(theEnvironment.theResultsCache);
        return py::make_tuple(fmaps, error_message);
    }

    if (theEnvironment.saveOption == Nyxus::SaveOption::saveBuffer) {

        auto pyHeader = py::array(py::cast(theEnvironment.theResultsCache.get_headerBuf()));
        auto pyStrData = py::array(py::cast(theEnvironment.theResultsCache.get_stringColBuf()));
        auto pyNumData = as_pyarray(std::move(theEnvironment.theResultsCache.get_calcResultBuf()));

        auto nRows = theEnvironment.theResultsCache.get_num_rows();
        pyStrData = pyStrData.reshape({nRows, pyStrData.size() / nRows});
        pyNumData = pyNumData.reshape({ nRows, pyNumData.size() / nRows });

        return py::make_tuple(pyHeader, pyStrData, pyNumData, error_message);
    }

    return py::make_tuple(error_message);
}

py::tuple featurize_fname_lists_imp (uint64_t instid, const py::list& int_fnames, const py::list & seg_fnames, bool single_roi, const std::string& output_type, const std::string& output_path)
{
    Environment & theEnvironment = Nyxus::findenv(instid);

    // Set the whole-slide/multi-ROI flag
    theEnvironment.singleROI = single_roi;

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
        throw std::runtime_error("Segmentation mask file list is blank");
    if (intensFiles.size() != labelFiles.size())
        throw std::runtime_error("Imbalanced intensity and segmentation mask file lists");
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

    theEnvironment.theResultsCache.clear();

    // Process the image sdata
    int min_online_roi_size = 0;
    int errorCode;

    theEnvironment.saveOption = [&output_type](){
        if (output_type == "arrowipc") {
            return SaveOption::saveArrowIPC;
	    } else if (output_type == "parquet") {
            return SaveOption::saveParquet;
		} else {return SaveOption::saveBuffer;}
	}();

    if (theEnvironment.fmaps_prevents_arrow())
        throw std::runtime_error("Arrow/Parquet output is not supported in feature maps (fmaps) mode. Use CSV or buffer output instead.");

    if (theEnvironment.fmaps_mode)
        errorCode = processDataset_2D_fmaps (
            theEnvironment,
            intensFiles,
            labelFiles,
            theEnvironment.n_reduce_threads,
            theEnvironment.saveOption,
            output_path);
    else
        errorCode = processDataset_2D_segmented (
            theEnvironment,
            intensFiles,
            labelFiles,
            theEnvironment.n_reduce_threads,
            theEnvironment.saveOption,
            output_path);

    if (errorCode)
        throw std::runtime_error("Error occurred during dataset processing.");

    if (theEnvironment.fmaps_mode && theEnvironment.saveOption == Nyxus::SaveOption::saveBuffer)
    {
        auto fmaps = fmap_results_to_python(theEnvironment.theResultsCache);
        return py::make_tuple(fmaps);
    }

    if (theEnvironment.saveOption == Nyxus::SaveOption::saveBuffer) {

            auto pyHeader = py::array(py::cast(theEnvironment.theResultsCache.get_headerBuf()));
            auto pyStrData = py::array(py::cast(theEnvironment.theResultsCache.get_stringColBuf()));
            auto pyNumData = as_pyarray(std::move(theEnvironment.theResultsCache.get_calcResultBuf()));

            auto nRows = theEnvironment.theResultsCache.get_num_rows();
            pyStrData = pyStrData.reshape({nRows, pyStrData.size() / nRows});
            pyNumData = pyNumData.reshape({ nRows, pyNumData.size() / nRows });

            return py::make_tuple(pyHeader, pyStrData, pyNumData);
    }

    // Return "nothing" when output will be an Arrow format
    return py::make_tuple();
}

py::tuple featurize_fname_lists_3D_imp (
    uint64_t instid,
    const py::list& pyside_int_fnames, 
    const py::list& pyside_seg_fnames, 
    bool single_roi, 
    const std::string& output_type, 
    const std::string& output_path)
{
    Environment& theEnvironment = Nyxus::findenv(instid);

    // set the whole-slide/multi-ROI flag
    theEnvironment.singleROI = single_roi;

    // python-side file name lists to c++ vectors 
    std::vector <Imgfile3D_layoutA> ifiles, mfiles;
    for (auto it = pyside_int_fnames.begin(); it != pyside_int_fnames.end(); ++it)
    {
        std::string fn = it->cast<std::string>();
        Imgfile3D_layoutA f(fn);
        ifiles.push_back(f);
    }
    for (auto it = pyside_seg_fnames.begin(); it != pyside_seg_fnames.end(); ++it)
    {
        std::string fn = it->cast<std::string>();
        Imgfile3D_layoutA f(fn);
        mfiles.push_back(f);
    }

    // check the file name sets 
    if (ifiles.size() == 0)
        throw std::runtime_error("Intensity file list is blank");
    if (mfiles.size() == 0)
        throw std::runtime_error("Mask file list is blank");
    if (ifiles.size() != mfiles.size())
        throw std::runtime_error("Imbalanced intensity and segmentation mask file lists");
    for (auto i = 0; i < ifiles.size(); i++)
    {
        const std::string& i_fname = ifiles[i].fdir + ifiles[i].fname;		// .fdir ends with /
        const std::string& s_fname = mfiles[i].fdir + mfiles[i].fname;

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

    theEnvironment.theResultsCache.clear();

    // featurize
    int min_online_roi_size = 0;
    int errorCode;

    theEnvironment.saveOption = [&output_type]()
    {
        if (output_type == "arrowipc")
        {
            return SaveOption::saveArrowIPC;
        }
        else
            if (output_type == "parquet")
            {
                return SaveOption::saveParquet;
            }
            else
            {
                return SaveOption::saveBuffer;
            }
    }();

    if (theEnvironment.fmaps_mode)
    {
        errorCode = processDataset_3D_fmaps(
            theEnvironment,
            ifiles,
            mfiles,
            theEnvironment.n_reduce_threads,
            theEnvironment.saveOption,
            output_path);
    }
    else
    {
        errorCode = processDataset_3D_segmented(
            theEnvironment,
            ifiles,
            mfiles,
            theEnvironment.n_reduce_threads,
            theEnvironment.saveOption,
            output_path);
    }

    if (errorCode)
        throw std::runtime_error("Error occurred during dataset processing");

    // save the result
    if (theEnvironment.fmaps_mode && theEnvironment.saveOption == Nyxus::SaveOption::saveBuffer)
    {
        py::list fmaps = fmap_results_to_python(theEnvironment.theResultsCache);
        return py::make_tuple(fmaps);
    }

    if (theEnvironment.saveOption == Nyxus::SaveOption::saveBuffer)
    {
        auto pyHeader = py::array(py::cast(theEnvironment.theResultsCache.get_headerBuf()));
        auto pyStrData = py::array(py::cast(theEnvironment.theResultsCache.get_stringColBuf()));
        auto pyNumData = as_pyarray(std::move(theEnvironment.theResultsCache.get_calcResultBuf()));

        auto nRows = theEnvironment.theResultsCache.get_num_rows();
        pyStrData = pyStrData.reshape({ nRows, pyStrData.size() / nRows });
        pyNumData = pyNumData.reshape({ nRows, pyNumData.size() / nRows });

        return py::make_tuple(pyHeader, pyStrData, pyNumData);
    }

    // return "nothing" when output will be an Arrow format
    return py::make_tuple();
}

py::tuple findrelations_imp(
    uint64_t instid,
    std::string& label_dir,
    std::string& parent_file_pattern,
    std::string& child_file_pattern)
{
    Environment& env = Nyxus::findenv (instid);

    if (! env.check_2d_file_pattern(parent_file_pattern) || ! env.check_2d_file_pattern(child_file_pattern))
        throw std::invalid_argument("Filepattern provided is not valid.");

    env.theResultsCache.clear();

    // Result -> headerBuf, stringColBuf, calcResultBuf
    ChildFeatureAggregation aggr;
    bool mineOK = mine_segment_relations (env.theResultsCache, true, label_dir, parent_file_pattern, child_file_pattern, ".", aggr, env.get_verbosity_level());  // the 'outdir' parameter is not used if 'output2python' is true

    if (! mineOK)
        throw std::runtime_error("Error occurred during dataset processing: mine_segment_relations() returned false");
    
    auto pyHeader = py::array(py::cast(env.theResultsCache.get_headerBuf()));
    auto pyStrData = py::array(py::cast(env.theResultsCache.get_stringColBuf()));
    auto pyNumData = as_pyarray(std::move(env.theResultsCache.get_calcResultBuf()));
    auto nRows = env.theResultsCache.get_num_rows();
    pyStrData = pyStrData.reshape({ nRows, pyStrData.size() / nRows });
    pyNumData = pyNumData.reshape({ nRows, pyNumData.size() / nRows });

    return py::make_tuple(pyHeader, pyStrData, pyNumData);
}

bool gpu_available_imp (uint64_t instid)
{
    Environment & env = Nyxus::findenv(instid);
    return env.gpu_is_available();
}

/**
 * @brief Set whether to use the gpu for available gpu features
 * 
 * @param yes True to use gpu
 */
void use_gpu (uint64_t instid, bool yes)
{
    #ifdef USE_GPU
    Environment & env = Nyxus::findenv(instid);
    env.set_using_gpu(yes);
    #else 
        throw std::runtime_error("this Nyxus backend was built without the GPU support");
    #endif
}

/**
 * @brief Get the gpu properties. If gpu is not available, return an empty vector
 * 
 * @return std::vector<std::map<std::string, std::string>> Properties of gpu
 */
static std::vector<std::map<std::string, std::string>> get_gpu_properties(uint64_t instid) 
{
    #ifdef USE_GPU
    Environment& env = Nyxus::findenv(instid);
    return env.get_gpu_properties();
    #else 
        std::vector<std::map<std::string, std::string>> empty;
        return empty;
    #endif
}

void blacklist_roi_imp (uint64_t instid, std::string raw_blacklist)
{
    // After successfully parsing the blacklist, Nyxus runtime becomes able 
    // to skip blacklisted ROIs until the cached blacklist is cleared 
    // with Environment::clear_roi_blacklist()

    Environment & env = Nyxus::findenv (instid);

    std::string lastError;
    if (! env.parse_roi_blacklist_raw_string (raw_blacklist, lastError))
    {
        std::string ermsg = "Error parsing ROI blacklist definition: " + lastError;
        throw std::runtime_error(ermsg);
    }
}

void clear_roi_blacklist_imp (uint64_t instid)
{
    Environment & env = Nyxus::findenv (instid);
    env.clear_roi_blacklist();
}

py::str roi_blacklist_get_summary_imp (uint64_t instid)
{
    Environment & env = Nyxus::findenv (instid);
    std::string response;
    env.get_roi_blacklist_summary(response);
    return py::str(response);
}

void customize_gabor_feature_imp(
    uint64_t instid,
    const std::string& kersize,
    const std::string& gamma,
    const std::string& sig2lam,
    const std::string& f0,
    const std::string& theta,
    const std::string& thold,
    const std::string& freqs)
{
    Environment & env = Nyxus::findenv (instid);

    // Step 1 - set raw strings of parameter values
    env.gaborOptions.rawKerSize = kersize;
    env.gaborOptions.rawGamma = gamma;
    env.gaborOptions.rawSig2lam = sig2lam;
    env.gaborOptions.rawF0 = f0;
    env.gaborOptions.rawTheta = theta;
    env.gaborOptions.rawGrayThreshold = thold;
    env.gaborOptions.rawFreqs = freqs;

    // Step 2 - validate them and consume if all are valid
    std::string ermsg;
    if (! env.parse_gabor_options_raw_inputs(ermsg))
        throw std::invalid_argument("Invalid GABOR parameter value: " + ermsg);
}

std::map<std::string, ParameterTypes> get_params_imp (uint64_t instid, const std::vector<std::string>& vars )
{
    Environment & theEnvironment = Nyxus::findenv (instid);

    std::map<std::string, ParameterTypes> params;

    params["features"] = theEnvironment.recognizedFeatureNames;
    params["neighbor_distance"] = theEnvironment.n_pixel_distance;
    params["pixels_per_micron"] = theEnvironment.xyRes;
    int cgd = theEnvironment.get_coarse_gray_depth();
    params["coarse_gray_depth"] = std::abs(cgd);
    params["binning_origin"] = std::string(cgd < 0 ? "min" : "zero");
    params["n_feature_calc_threads"] = theEnvironment.n_reduce_threads;
    params["ibsi"] = theEnvironment.ibsi_compliance;

    params["gabor_kersize"] = GaborFeature::n;
    params["gabor_gamma"] = GaborFeature::gamma;
    params["gabor_sig2lam"] = GaborFeature::sig2lam;
    params["gabor_f0"] = GaborFeature::f0LP;
    params["gabor_thold"] = GaborFeature::GRAYthr;

    std::vector<double> f, t;
    for (auto p : GaborFeature::f0_theta_pairs)
    {
        f.push_back (p.first);
        t.push_back (Nyxus::rad2deg(p.second));
    }
    params["gabor_freqs"] = f;
    params["gabor_thetas"] = t;

    params["dynamic_range"] = theEnvironment.fpimageOptions.target_dyn_range();
    params["min_intensity"] = theEnvironment.fpimageOptions.min_intensity();
    params["max_intensity"] = theEnvironment.fpimageOptions.max_intensity();
    params["ram_limit"] = (int)(theEnvironment.get_ram_limit()/1048576); // convert from bytes to megabytes

    params["fmaps"] = theEnvironment.fmaps_mode;
    params["fmaps_radius"] = theEnvironment.fmaps_kernel_radius;

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

std::string get_arrow_file_imp (uint64_t instid)
{
#ifdef USE_ARROW

    Environment & env = Nyxus::findenv (instid);
    return env.arrow_stream.get_arrow_path();

#else
    
    throw std::runtime_error("Arrow functionality is not available. Rebuild Nyxus with Arrow enabled.");

#endif
}

std::string get_parquet_file_imp (uint64_t instid)
{
#ifdef USE_ARROW

    Environment & env = Nyxus::findenv (instid);
    return env.arrow_stream.get_arrow_path();

#else
    
    throw std::runtime_error("Arrow functionality is not available. Rebuild Nyxus with Arrow enabled.");

#endif
}

bool arrow_is_enabled_imp (uint64_t instid)
{
    Environment & env = Nyxus::findenv (instid);
    return env.arrow_is_enabled();
}

py::tuple set_metaparam_imp (uint64_t instid, const std::string p_val)
{
    Environment & env = Nyxus::findenv (instid);
    std::optional<std::string> mayBerror = env.set_metaparam (p_val);
    if (mayBerror.has_value())
        return py::make_tuple (false, mayBerror.value());
    else
        return py::make_tuple (true, "success");
}

py::tuple get_metaparam_imp (uint64_t instid, const std::string p_name)
{
    Environment& env = Nyxus::findenv(instid);
    double p_val;
    std::optional<std::string> mayBerror = env.get_metaparam (p_val, p_name);
    if (mayBerror.has_value())
        return py::make_tuple (false, mayBerror.value());
    else
        return py::make_tuple (p_val, "");
}



PYBIND11_MODULE(backend, m)
{
    m.doc() = "Nyxus";

    // Register an atexit handler to destroy all Environment objects while
    // the Python interpreter is still alive.  Without this, the global
    // pynyxus_cache is destroyed during C++ static destruction — after
    // Python has already torn down modules — causing segfaults from
    // stale references in LR/ImageMatrix/ResultsCache destructors.
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
        Nyxus::pynyxus_cache.clear();
        Nyxus::unique_pynyxus_ids.clear();
    }));

    m.def("initialize_environment",     &initialize_environment,    "Environment initialization");
    m.def("featurize_directory_imp",    &featurize_directory_imp,   "Calculate features of images defined by intensity and mask image collection directories");
    m.def("featurize_directory_3D_imp", &featurize_directory_3D_imp,    "Calculate 3D features of images defined by intensity and mask image collection directories");
    m.def("featurize_montage_imp",      &featurize_montage_imp,     "Calculate features of images defined by intensity and mask image collection directories");
    m.def("featurize_fname_lists_imp",  &featurize_fname_lists_imp, "Calculate features of intensity-mask image pairs defined by lists of image file names");
    m.def("featurize_fname_lists_3D_imp",   &featurize_fname_lists_3D_imp,  "Calculate 3D features of intensity-mask volume pairs defined by lists of file names");
    m.def("findrelations_imp",          &findrelations_imp,         "Find relations in segmentation mask images");
    m.def("gpu_available_imp",          &gpu_available_imp,         "Check if CUDA gpu is available");
    m.def("use_gpu",                    &use_gpu,                   "Enable/disable GPU features");
    m.def("get_gpu_props",              &get_gpu_properties,        "Get properties of the active CUDA gpu device");
    m.def("blacklist_roi_imp",          &blacklist_roi_imp,         "Set up a global or per-mask file blacklist definition");
    m.def("clear_roi_blacklist_imp",    &clear_roi_blacklist_imp,   "Clear the ROI black list");
    m.def("roi_blacklist_get_summary_imp",  &roi_blacklist_get_summary_imp, "Returns a summary of the ROI blacklist");
    m.def("customize_gabor_feature_imp",    &customize_gabor_feature_imp,   "Sets custom GABOR feature's parameters");
    m.def("set_if_ibsi_imp",            &set_if_ibsi_imp,           "Set if the features will be ibsi compliant");
    m.def("set_fmaps_imp",              &set_fmaps_imp,             "Enable/disable feature maps mode and set kernel radius");
    m.def("set_environment_params_imp", &set_environment_params_imp,    "Set the environment variables of Nyxus");
    m.def("get_params_imp",             &get_params_imp,            "Get parameters of Nyxus");
    m.def("arrow_is_enabled_imp",       &arrow_is_enabled_imp,      "Check if arrow is enabled.");
    m.def("get_arrow_file_imp",         &get_arrow_file_imp,        "Get path to arrow file");
    m.def("get_parquet_file_imp",       &get_parquet_file_imp,      "Returns path to parquet file");
    m.def("set_metaparam_imp", &set_metaparam_imp, "Setting a common or feature-specific metaparameter");
    m.def("get_metaparam_imp", &get_metaparam_imp, "Getting a common or feature-specific metaparameter value");
}



