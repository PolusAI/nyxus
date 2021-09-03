#pragma once

//---	#define SINGLE_ROI_TEST

#include <climits>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include "featureset.h"
#include "histogram.h"
#include "pixel.h"
#include "aabb.h"
#include "image_matrix.h"

#define INF 10E200	// Cautious infinity

bool datasetDirsOK (const std::string & dirIntens, const std::string & dirLab, const std::string & dirOut, bool mustCheckDirOut);
bool directoryExists (const std::string & dir);
void readDirectoryFiles (const std::string & dir, std::vector<std::string> & files);
bool scanViaFastloader (const std::string & fpath, int num_threads);
bool scanFilePair (const std::string& intens_fpath, const std::string& label_fpath, int num_fastloader_threads, int num_sensemaker_threads);
bool scanFilePairParallel (const std::string& intens_fpath, const std::string& label_fpath, int num_fastloader_threads, int num_sensemaker_threads);
bool TraverseViaFastloader1 (const std::string& fpath, int num_threads);
std::string getPureFname(std::string fpath);
int processDataset (std::vector<std::string> & intensFiles, std::vector<std::string> & labelFiles, int numFastloaderThreads, int numSensemakerThreads, int min_online_roi_size, bool save2csv, std::string csvOutputDir);

// 2 scenarios of saving a result of feature calculation of a label-intensity file pair: saving to a CSV-file and saving to a matrix to be later consumed by a Python endpoint
bool save_features_2_csv (std::string inputFpath, std::string outputDir);
bool save_features_2_buffer (std::vector<double> & resultMatrix);

void showCmdlineHelp();
int checkAndReadDataset(
	// input
	const std::string& dirIntens, const std::string& dirLabels, const std::string& dirOut, bool mustCheckDirOut, 
	// output
	std::vector<std::string>& intensFiles, std::vector<std::string>& labelFiles);

using Histo = OnlineHistogram<PixIntens>;

void init_feature_buffers();
void update_label_stats (int x, int y, int label, PixIntens intensity);
void update_label_stats_parallel (int x, int y, int label, PixIntens intensity);
void print_label_stats();
void print_by_label(const char* featureName, std::unordered_map<int, StatsInt> L, int numColumns = 8); 
void print_by_label(const char* featureName, std::unordered_map<int, StatsReal> L, int numColumns = 4);
void clearLabelStats();
void reduce_all_labels(int min_online_roi_size);

// Inherited from WNDCHRM, used for Feret and Martin statistics calculation
struct Statistics 
{
	double min, max, mode;
	double mean, median, stdev;
};

Statistics ComputeCommonStatistics2 (std::vector<double>& Data);

class Contour
{
public:
	Contour() {}
	void calculate(std::vector<Pixel2> rawPixels);
	std::vector<Pixel2> theContour;
	StatsInt get_roi_perimeter();
	StatsReal get_diameter_equal_perimeter();
protected:
};

class ConvexHull
{
public:
	ConvexHull() {}
	void calculate (std::vector<Pixel2> & rawPixels);
	double getSolidity();
	double getArea();
	std::vector<Pixel2> CH;
};

class Hexagonality_and_Polygonality
{
public:
	Hexagonality_and_Polygonality() {}
	std::tuple<double, double, double> calculate(int num_neighbors, int roi_area, int roi_perimeter, double convhull_area, double min_feret_diam, double max_feret_diam);

};

// Longest chord, Feret, Martin, Nassenstein diameters
class ParticleMetrics
{
public:
	ParticleMetrics(std::vector<Pixel2>& _convex_hull);

	void calc_ferret(
		// output:
		double& minFeretDiameter,
		double& minFeretAngle,
		double& maxFeretDiameter,
		double& maxFeretAngle,
		std::vector<double>& all_D);
	void calc_martin (std::vector<double>& D);
	void calc_nassenstein (std::vector<double>& D);
	const int NY = 10;
	const int rot_angle_increment = 10;	// degrees
protected:
	std::tuple<StatsInt, StatsInt, StatsInt, StatsInt> get_pixelcloud_bounds (std::vector<Pixel2> & pixels);	// Returns minX, minY, maxX, maxY
	void rotate_pixels(
		// in 
		float angle_deg,
		std::vector<Pixel2> & P, 
		// out
		std::vector<Pixel2>& P_rot);

	std::vector<Pixel2>& convex_hull;
};

class EulerNumber
{
public:
	long euler_number;
	EulerNumber (std::vector<Pixel2>& P, StatsInt min_x, StatsInt  min_y, StatsInt max_x, StatsInt max_y, int mode);
protected:
	long calculate (std::vector<unsigned char>& I, int height, int width, int mode);
};

class MinEnclosingCircle
{
public:
	MinEnclosingCircle() {}
	double calculate_diam (std::vector<Pixel2>& Contour);
	const float EPS = 1.0e-4f;

protected:
	void minEnclosingCircle (
		// in:
		std::vector<Pixel2> & Contour, 
		// out:
		Point2f &center, 
		float &radius);
	void findMinEnclosingCircle (const std::vector<Pixel2>& pts, int count, Point2f& center, float& radius);
	void findSecondPoint (const std::vector<Pixel2>& pts, int i, Point2f& center, float& radius);
	void findThirdPoint (const std::vector<Pixel2>& pts, int i, int j, Point2f& center, float& radius);
	void findCircle3pts (const std::vector<Pixel2>& pts, Point2f& center, float& radius);
};

class InscribingCircumscribingCircle
{
public:
	std::tuple <double, double> calculateInsCir (std::vector<Pixel2>& contours, double xCentroid, double yCentroid);
};

class GeodeticLength_and_Thickness
{
public:
	GeodeticLength_and_Thickness() {}
	std::tuple<double, double> calculate(StatsInt roiArea, StatsInt roiPerimeter);

protected:
};

void haralick2D(
	// in
	std::vector <Pixel2>& nonzero_intensity_pixels,
	AABB & aabb,
	double distance,
	// out
	std::vector<double>& Texture_Feature_Angles,
	std::vector<double>& Texture_AngularSecondMoments,
	std::vector<double>& Texture_Contrast,
	std::vector<double>& Texture_Correlation,
	std::vector<double>& Texture_Variance,
	std::vector<double>& Texture_InverseDifferenceMoment,
	std::vector<double>& Texture_SumAverage,
	std::vector<double>& Texture_SumVariance,
	std::vector<double>& Texture_SumEntropy,
	std::vector<double>& Texture_Entropy,
	std::vector<double>& Texture_DifferenceVariance,
	std::vector<double>& Texture_DifferenceEntropy,
	std::vector<double>& Texture_InfoMeas1,
	std::vector<double>& Texture_InfoMeas2);

void zernike2D(
	// in
	std::vector <Pixel2>& nonzero_intensity_pixels,
	AABB& aabb,
	int order,
	// out
	std::vector<double>& Z_values);

extern FeatureSet featureSet;

// Label record - structure aggregating label's cached data and calculated features
#define DFLT0 -0.0	// default unassigned value
#define DFLT0i -0	// default unassigned value
struct LR
{
	// Helper objects
	std::vector <Pixel2> raw_pixels;	
	AABB aabb;	
	ConvexHull convHull;
	Contour cntr;

	//==== Pixel intensity statistics

	StatsInt 
		pixelCountRoiArea,	
		median,
		min,
		max,
		massEnergy;
	StatsReal 
		mean,
		variance,
		stddev,
		centroid_x,
		centroid_y,
		aux_M2,
		aux_M3,
		aux_M4,
		skewness,
		kurtosis,
		MAD,
		RMS,		// Root Mean Squared (RMS) is the square-root of the mean of all the squared intensity values. It is another measure of the magnitude of the image values.
		p10,
		p25,
		p75,
		p90,
		IQR,
		entropy,
		mode,
		uniformity,
		RMAD;

	std::shared_ptr<Histo> aux_Histogram;
	StatsInt 
		aux_PrevCount,
		aux_PrevIntens;

	//==== Morphology

	void init_aabb (StatsInt x, StatsInt y);
	void update_aabb (StatsInt x, StatsInt y);
	int num_neighbors		= DFLT0i;
	StatsReal extent		= DFLT0;	// Calculated in reduce(), not accumulated online

	// --ellipse related
	StatsReal 
		major_axis_length	= DFLT0,
		minor_axis_length	= DFLT0,
		eccentricity		= DFLT0,
		orientation			= DFLT0;

	// --contour and convex hull related
	StatsInt roiPerimeter	= DFLT0i;
	StatsReal 
		aspectRatio			= DFLT0,
		equivDiam			= DFLT0,
		circularity			= DFLT0,
		convHullArea		= DFLT0,
		solidity			= DFLT0;

	// --extrema
	StatsReal	
		extremaP1y			= DFLT0, 
		extremaP1x			= DFLT0,
		extremaP2y			= DFLT0, 
		extremaP2x			= DFLT0,
		extremaP3y			= DFLT0, 
		extremaP3x			= DFLT0,
		extremaP4y			= DFLT0, 
		extremaP4x			= DFLT0,
		extremaP5y			= DFLT0, 
		extremaP5x			= DFLT0,
		extremaP6y			= DFLT0, 
		extremaP6x			= DFLT0,
		extremaP7y			= DFLT0, 
		extremaP7x			= DFLT0,
		extremaP8y			= DFLT0, 
		extremaP8x			= DFLT0;

	// --Feret
	StatsReal	
		maxFeretDiameter	= DFLT0,
		maxFeretAngle		= DFLT0,
		minFeretDiameter	= DFLT0,
		minFeretAngle		= DFLT0,
		feretStats_minD		= DFLT0,
		feretStats_maxD		= DFLT0,
		feretStats_meanD	= DFLT0,
		feretStats_medianD	= DFLT0,
		feretStats_stddevD	= DFLT0,
		feretStats_modeD	= DFLT0;

	// --Martin
	StatsReal	
		martinStats_minD	= DFLT0,
		martinStats_maxD	= DFLT0,
		martinStats_meanD	= DFLT0,
		martinStats_medianD	= DFLT0,
		martinStats_stddevD	= DFLT0,
		martinStats_modeD	= DFLT0;

	// --Nassenstein
	StatsReal
		nassStats_minD		= DFLT0,
		nassStats_maxD		= DFLT0,
		nassStats_meanD		= DFLT0,
		nassStats_medianD	= DFLT0,
		nassStats_stddevD	= DFLT0,
		nassStats_modeD		= DFLT0;

	// --Euler
	long euler_number		= DFLT0;

	// --hexagonality & polygonality
	StatsReal 
		polygonality_ave	= DFLT0, 
		hexagonality_ave	= DFLT0, 
		hexagonality_stddev	= DFLT0;

	// --Circle fitting
	StatsReal
		diameter_min_enclosing_circle	= DFLT0,
		diameter_circumscribing_circle	= DFLT0,
		diameter_inscribing_circle		= DFLT0;

	StatsReal 
		geodeticLength		= DFLT0,
		thickness			= DFLT0;

	//==== Texture

	// --Haralick 2D aka CellProfiler_*
	std::vector<double> 
		texture_Feature_Angles,	// (auxiliary field) angles e.g. 0, 45, 90, 135, etc.
		texture_AngularSecondMoments, // Angular Second Moment
		texture_Contrast, // Contrast
		texture_Correlation, // Correlation
		texture_Variance, // Variance
		texture_InverseDifferenceMoment, // Inverse Diffenence Moment
		texture_SumAverage, // Sum Average
		texture_SumVariance, // Sum Variance
		texture_SumEntropy, // Sum Entropy
		texture_Entropy, // Entropy
		texture_DifferenceVariance, // Difference Variance
		texture_DifferenceEntropy, // Diffenence Entropy
		texture_InfoMeas1, // Measure of Correlation 1
		texture_InfoMeas2; // Measure of Correlation 2
	
	// Zernike calculator may put an arbitrary number of Z_a^b terms 
	// but we output only 'NUM_ZERNIKE_COEFFS_2_OUTPUT' of them 
	static const short aux_ZERNIKE2D_ORDER = 9, aux_ZERNIKE2D_NUM_COEFS = 30;	// z00, z11, z20, z22, z31, z33, z40, z42, z44, ... ,z97, z99 - 30 items altogether 
	std::vector<double> Zernike2D;	

	double getValue (AvailableFeatures f);
};

void init_label_record(LR& lr, int x, int y, int label, PixIntens intensity);
void update_label_record(LR& lr, int x, int y, int label, PixIntens intensity);
void reduce_neighbors (int labels_collision_radius);

// Timing
extern double totalTileLoadTime, totalPixStatsCalcTime;
double test_containers1();
double test_containers2();
bool test_histogram();

// Spatial hashing
inline bool aabbNoOverlap (
	StatsInt xmin1, StatsInt xmax1, StatsInt ymin1, StatsInt ymax1, 
	StatsInt xmin2, StatsInt xmax2, StatsInt ymin2, StatsInt ymax2,
	int R)
{
	bool retval = xmin2 - R > xmax1 + R || xmax2 + R < xmin1 - R
		|| ymin2 - R > ymax1 + R || ymax2 + R < ymin1 - R;
	return retval;
}

inline bool aabbNoOverlap (LR & r1, LR & r2, int radius)
{
	bool retval = aabbNoOverlap(r1.aabb.get_xmin(), r1.aabb.get_xmax(), r1.aabb.get_ymin(), r1.aabb.get_ymax(),
		r2.aabb.get_xmin(), r2.aabb.get_xmax(), r2.aabb.get_ymin(), r2.aabb.get_ymax(), radius);
	return retval;
}

inline unsigned long spat_hash_2d (StatsInt x, StatsInt y, int m)
{
	unsigned long h = x * 73856093;
	h = h ^ y * 19349663;
	// hash   hash  z × 83492791	// For the future
	// hash   hash  l × 67867979
	unsigned long retval = h % m;
	return retval;
}

// Label data
extern std::unordered_set<int> uniqueLabels;
extern std::unordered_map <int, LR> labelData;
extern std::vector<double> calcResultBuf;	// [# of labels X # of features]
extern std::unordered_map <int, std::shared_ptr<std::mutex>> labelMutexes;
