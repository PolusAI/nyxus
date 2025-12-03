#pragma once

// Command line options and arguments
#define clo_SEGDIR "--segDir"						// Environment :: labels_dir
#define clo_INTDIR "--intDir"						// Environment :: intensity_dir
#define clo_OUTDIR "--outDir"						// Environment :: output_dir
#define clo_INTSEGMAPDIR "--intSegMapDir"			// get_int_seg_map_dir()
#define clo_INTSEGMAPFILE "--intSegMapFile"			// get_int_seg_map_file()
#define clo_FEATURES "--features"					// Environment :: features	-- Example: (1) --features=area,kurtosis,neighbors (2) --features=featurefile.txt
#define clo_FILEPATTERN "--filePattern"				// Environment :: file_pattern
#define clo_OUTPUTTYPE "--outputType"				// Environment :: Output type for feature values (speratecsv, singlecsv, arrow, parquet)
#define clo_EMBPIXSZ "--embeddedpixelsize"			// Environment :: embedded_pixel_size
#define clo_REDUCETHREADS "--reduceThreads"			// Environment :: n_reduce_threads
#define clo_VERBOSITY "--verbose"					// Environment :: verbosity_level	-- Example: --verbosity=3
#define clo_ONLINESTATSTHRESH "--onlineStatsThresh" // Environment :: onlineStatsThreshold	-- Example: --onlineStatsThresh=150
#define clo_XYRESOLUTION "--pixelsPerCentimeter"	// pixels per centimeter
#define clo_PXLDIST "--pixelDistance"				// used in neighbor features
#define clo_COARSEGRAYDEPTH "--coarseGrayDepth"		// Environment :: raw_coarse_grayscale_depth
#define clo_RAMLIMIT "--ramLimit"					// Optional. Limit for treating ROIs as non-trivial and for setting the batch size of trivial ROIs. Default - amount of available system RAM
#define clo_TEMPDIR "--tempDir"						// Optional. Used in processing non-trivial features. Default - system temp directory
#define clo_IBSICOMPLIANCE "--ibsi" // skip binning for grey level and grey tone features
#define clo_SKIPROI "--skiproi"		// Optional. Skip ROIs having specified labels. Sybtax: --skiproi <label[,label,label,...]>
#define clo_RESULTFNAME "--resultFname"				// Environment :: nyxus_result_fname
#define clo_CLI_DIM "--dim"							// Environment :: raw_dim

#ifdef CHECKTIMING
	#define clo_EXCLUSIVETIMING "--exclusivetiming"
#endif

#ifdef USE_GPU
	#define clo_USEGPU "--useGpu"					// Environment::rawUseGpu, "true" or "false"
	#define clo_GPUDEVICEID "--gpuDeviceID"		// Environment::rawGpuDeviceID
#endif

// Gabor feature CLI arguments
#define clo_GABOR_FREQS "--gaborfreqs"		// Example: "2,4,8,72". Frequencies should atch thetas: --gaborfreqs=1,2,3,4,5 --gabortheta=30,30,45,90,90
#define clo_GABOR_GAMMA "--gaborgamma"		// Example: "0.1"
#define clo_GABOR_SIG2LAM "--gaborsig2lam"	// Example: "0.8"
#define clo_GABOR_KERSIZE "--gaborkersize"	// Example: "20"
#define clo_GABOR_F0 "--gaborf0"			// Example: "0.1"
#define clo_GABOR_THETA "--gabortheta"		// Example: "60,45,90"
#define clo_GABOR_THRESHOLD "--gaborthold"	// Example: "0.025"

// GLCM feature
#define clo_GLCMANGLES "--glcmAngles"				// Environment :: rotAngles
#define clo_GLCMOFFSET "--glcmOff"					// Environment :: raw_glcm_

// Nested ROI functionality
#define clo_NESTEDROI_CHNL_SIGNATURE "--hsig"		// Channel signature Example: "_c" in "p0_y1_r1_c1.ome.tiff"
#define clo_NESTEDROI_PARENT_CHNL "--hpar"			// Channel number that should be used as a provider of parent segments. Example: --hpar=1
#define clo_NESTEDROI_CHILD_CHNL "--hchi"			// Channel number that should be used as a provider of child segments. Example: --hchi=0
#define clo_NESTEDROI_AGGREGATION_METHOD "--hag"	// How to aggregate features of segments recognized as children of same parent segment. See class NestedRoiOptions for options.

// Floating-point voxel image options (served by class FpImageOptions)
#define clo_FPIMAGE_TARGET_DYNRANGE "--fpimgdr"		// Desired dynamic range of the integer voxel intensities converted from floating-point intensities
#define clo_FPIMAGE_MIN "--fpimgmin"				// Expected voxel min intensity
#define clo_FPIMAGE_MAX "--fpimgmax"				// Expected voxel max intensity

// Anisotropy
#define clo_ANISO_X "--anisox"
#define clo_ANISO_Y "--anisoy"
#define clo_ANISO_Z "--anisoz"

// Result options
#define clo_NOVAL "--noval"						// -> raw_noval
#define clo_TINYVAL "--tinyval"					// -> raw_tiny
#define clo_AGGREGATE "--aggr"				// -> raw_aggregate
#define clo_ANNOTATE "--annot"				// -> raw_annotate
#define clo_ANNOT_SEP "--annotsep"		// -> raw_anno_separator

// Valid values of 'OUTPUTTYPE'
#define clo_OT_SEPCSV "separatecsv"
#define clo_OT_SINGLECSV "singlecsv"
#define clo_OT_ARROWIPC "arrowipc"
#define clo_OT_PARQUET "parquet"

// Verbosity levels (combinable via binary and)
#define VERBOSITY_TIMING 2
#define VERBOSITY_ROI_INFO 4
#define VERBOSITY_DETAILED 8
