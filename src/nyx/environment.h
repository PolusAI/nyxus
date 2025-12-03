#pragma once

#include <map>
#include <string>
#include <vector>

#include "arrow_output_stream.h"
#include "environment_basic.h"
#include "cache.h"
#include "cli_anisotropy_options.h"
#include "cli_fpimage_options.h"
#include "cli_gabor_options.h"
#include "cli_glcm_options.h"
#include "cli_nested_roi_options.h"
#include "cli_option_constants.h"
#include "cli_result_options.h"
#include "dataset.h"
#include "feature_mgr.h"
#include "feature_settings.h"
#include "results_cache.h"
#include "roi_blacklist.h"
#include "save_option.h"

#ifdef USE_GPU
	#include <cuda_runtime.h>
	#include "cli_gpu_options.h"
#endif

/// @brief Class encapsulating the the feature extraction environment - command line option values, default values, etc. Use it to add a parseable command line parameter.
class Environment: public BasicEnvironment
{
public:

	Environment();
	~Environment();
	bool parse_cmdline(int argc, char **argv);
	void show_cmdline_help();
	void show_featureset_help(); 
	void show_summary();

	std::string labels_dir, //= ""
		intensity_dir, //= ""
		output_dir, //=""
		intSegMapDir, //= "",
		intSegMapFile, //= "";
		nyxus_result_fname; //= "NyxusFeatures";	// Default file name without extension ".csv", ".arrow", etc

	// Returns the expected dataset dimensionality based on the command line options
	int dim() { return dim_; }
	void set_dim(int d) { dim_ = d; }
	bool is_imq() {return is_imq_;};
	void set_imq(bool is_imq) {is_imq_ = is_imq;}

	bool singleROI; //= false; // Applies to dim()==2: singleROI is set to 'true' parse_cmdline() if labels_dir==intensity_dir

	Nyxus::ArrowOutputStream arrow_stream;

	std::string embedded_pixel_size; //= "";

	std::string rawFeatures;
	std::vector<std::string> recognizedFeatureNames;

	std::string reduce_threads; //= "";
	int n_reduce_threads; //= 4;

	std::string pixel_distance; //= "";
	int n_pixel_distance; //= 5;

	std::string rawVerbosity; //= "";	// 'verbosity_level' is inherited from BasicEnvironment

	std::string rawOnlineStatsThresh; //= "";
	int onlineStatsTreshold; //= 0;

	std::string rawOutpType; //= ""; // Valid values: "separatecsv", "singlecsv", "arrow", "parquet"
	bool separateCsv; //= true;

	Nyxus::SaveOption saveOption;

	// x- and y- resolution in pixels per centimeter
	std::string rawXYRes; //= "";
	float xyRes; //= 0.0,
	float pixelSizeUm; //= 0.0;

	int get_pixel_distance();
	void set_pixel_distance(int pixelDistance);
	size_t get_ram_limit();
	void expand_featuregroups();

	void expand_IMQ_featuregroups();

	bool gpu_is_available();

	// IBSI mode switch
	bool ibsi_compliance;
	std::string raw_ibsi_compliance;
	void set_ibsi_compliance(bool skip);

#ifdef USE_GPU
	GpuOptions gpuOptions;
	bool parse_gpu_options_raw_string (const std::string& raw_params_string, std::string& error_message);

	// these are called from Python API's side
	int get_gpu_device_choice();
	void set_gpu_device_id (int id);

	void set_using_gpu (bool yes);
	static std::vector<std::map<std::string, std::string>> get_gpu_properties();
#endif

	bool using_gpu();	

	int get_floating_point_precision();

	int get_coarse_gray_depth();
	void set_coarse_gray_depth(unsigned int new_depth);

	// implementation of SKIPROI
	bool roi_is_blacklisted (const std::string& fname, int roi_label);
	bool parse_roi_blacklist_raw_string (const std::string& raw_blacklist_string, std::string& error_message);
	void clear_roi_blacklist ();
	void get_roi_blacklist_summary(std::string& response);

	bool set_ram_limit(size_t bytes);

	// implementation of Gabor feature options
	bool parse_gabor_options_raw_inputs (std::string& error_message);
	GaborOptions gaborOptions;

	// implementation of GLCM feature options
	bool parse_glcm_options_raw_inputs (std::string& error_message);
	GLCMoptions glcmOptions;

	// implementation of nested ROI options
	bool parse_nested_options_raw_inputs (std::string& error_message);
	NestedRoiOptions nestedOptions;
  
	// implementation of floating point image options
	bool parse_fpimage_options_raw_inputs (std::string& error_message);
	FpImageOptions fpimageOptions;

	std::tuple<bool, std::optional<std::string>> parse_aniso_options_raw_inputs ();
	AnisotropyOptions anisoOptions;

	// implementation of Apache options
	bool arrow_is_enabled();

	// feature result options (yes/no to annotation columns, yes/no to aggregate by slide, NAN substitute, etc)
	ResultOptions resultOptions;
	std::tuple<bool, std::optional<std::string>> parse_result_options_4cli ();

	// feature settings
	Fsettings fsett_PixelIntensity,
		fsett_BasicMorphology,
		fsett_Neighbors,
		fsett_Contour,
		fsett_ConvexHull,
		fsett_EllipseFitting,
		fsett_Extrema,
		fsett_EulerNumber,
		fsett_CaliperFeret,
		fsett_CaliperMartin,
		fsett_CaliperNassenstein,
		fsett_Chords,
		fsett_HexagonalityPolygonality,
		fsett_EnclosingInscribingCircumscribingCircle,
		fsett_GeodeticLengthThickness,
		fsett_RoiRadius,
		fsett_ErosionPixels,
		fsett_FractalDimension,
		fsett_GLCM,
		fsett_GLRLM,
		fsett_GLDZM,
		fsett_GLSZM,
		fsett_GLDM,
		fsett_NGLDM,
		fsett_NGTDM,
		fsett_Imoms2D,
		fsett_Smoms2D,
		fsett_Gabor,
		fsett_Zernike,
		fsett_RadialDistribution,
		// 3D
		fsett_D3_VoxelIntensity,
		fsett_D3_Surface,
		fsett_D3_GLCM,
		fsett_D3_GLDM,
		fsett_D3_GLDZM,
		fsett_D3_NGLDM,
		fsett_D3_NGTDM,
		fsett_D3_GLSZM,
		fsett_D3_GLRLM,
		// 2D image quality
		fsett_FocusScore,
		fsett_PowerSpectrum,
		fsett_Saturation,
		fsett_Sharpness;

	std::vector<std::reference_wrapper<Fsettings>> f_settings_;
	std::map<size_t, int> feature2settings_;
	void compile_feature_settings();
	const Fsettings & get_feature_settings (const std::type_info& ftype);

	// Features
	FeatureSet theFeatureSet;
	FeatureManager theFeatureMgr;

	// image/volume loading
	ImageLoader theImLoader;

	// dataset (properties of each slide/volume)
	Dataset dataset;

	// ROI data
	std::unordered_set<int> uniqueLabels;
	std::unordered_map <int, LR> roiData;

	// Results cache serving Nyxus' CLI & Python API, NyxusHie's CLI & Python API
	ResultsCache theResultsCache;		// global feature extraction results table

	// Runtime cache
	CpusideCache hostCache;

#ifdef USE_GPU
	GpusideCache devCache;
#endif

private:

	std::vector<std::tuple<std::string, std::string>> recognizedArgs;	// Accepted command line arguments

	bool find_string_argument (std::vector<std::string>::iterator &i, const char *arg, std::string &arg_value);
	bool find_int_argument(std::vector<std::string>::iterator &i, const char *arg, int &arg_value);
	bool spellcheck_raw_featurelist (const std::string & comma_separated_fnames, std::vector<std::string> & fnames);

	std::string rawTempDirPath; //= "";

	int floating_point_precision; //= 10;

	int coarse_grayscale_depth; //= 64;
	std::string raw_coarse_grayscale_depth; //= "";

	// data members implementing RAMLIMIT
	std::string rawRamLimit; //= "";
	size_t ramLimit; //= 0;

	// data members implementing TEMPDIR
	std::string rawTempDir; //= "";

	// implementation of SKIPROI
	std::string rawBlacklistedRois; //= "";
	RoiBlacklist roiBlacklist;

	// Dataset's dimensionality. Valid values: 2 and 3
	int dim_; //= 2;
	std::string raw_dim; //= "";
	bool expand_2D_featuregroup (const std::string& name);
	bool expand_3D_featuregroup (const std::string& name);
	bool expand_IMQ_featuregroup (const std::string & s);

	bool is_imq_; //= false;

	// data members implementing exclusive-inclusive timing switch
	#ifdef CHECKTIMING
		std::string rawExclusiveTiming; //= "";
	#endif

};

#define VERBOSLVL1(lvl, stmt) if(lvl>=1){stmt;}
#define VERBOSLVL2(lvl, stmt) if(lvl>=2){stmt;}
#define VERBOSLVL3(envobject, stmt) if(envobject.get_verbosity_level()>=3){stmt;}
#define VERBOSLVL4(lvl, stmt) if(lvl>=4){stmt;}	
#define VERBOSLVL5(lvl, stmt) if(lvl>=5){stmt;}	
