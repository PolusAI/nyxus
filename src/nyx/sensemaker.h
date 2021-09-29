#pragma once

#include <climits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
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
int processDataset (const std::vector<std::string> & intensFiles, const std::vector<std::string> & labelFiles, int numFastloaderThreads, int numSensemakerThreads, int numReduceThreads, int min_online_roi_size, bool save2csv, const std::string & csvOutputDir);

// 2 scenarios of saving a result of feature calculation of a label-intensity file pair: saving to a CSV-file and saving to a matrix to be later consumed by a Python endpoint
bool save_features_2_csv (std::string inputFpath, std::string outputDir);
bool save_features_2_buffer (std::vector<double> & resultMatrix);

void showCmdlineHelp();
int checkAndReadDataset(
	// input
	const std::string& dirIntens, const std::string& dirLabels, const std::string& dirOut, bool mustCheckDirOut, 
	// output
	std::vector<std::string>& intensFiles, std::vector<std::string>& labelFiles);

using Histo = OnlineHistogram;

void init_feature_buffers();
void update_label (int x, int y, int label, PixIntens intensity);
void update_label_parallel (int x, int y, int label, PixIntens intensity);
void print_label_stats();
void print_by_label(const char* featureName, std::unordered_map<int, StatsInt> L, int numColumns = 8); 
void print_by_label(const char* featureName, std::unordered_map<int, StatsReal> L, int numColumns = 4);
void clearLabelStats();
void reduce (int nThr, int min_online_roi_size);

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
	Contour() 
	{ 
		contour_pixels.reserve(100);
	}
	//void calculate (const std::vector<Pixel2> & rawPixels);
	void calculate (const ImageMatrix& im);	// Leaves result in 'contour_pixels'
	std::vector<Pixel2> contour_pixels;
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
	std::vector<Pixel2>& convex_hull;
};

class EulerNumber
{
public:
	EulerNumber (std::vector<Pixel2>& P, StatsInt min_x, StatsInt  min_y, StatsInt max_x, StatsInt max_y, int mode);
	long euler_number;	// Calculated via calculate()
protected:
	long calculate (std::vector<unsigned char>& I, int height, int width, int mode);
	static constexpr unsigned char Px[10] = { //MM: 0 or 1 in the left side of << represent binary pixel values
	// P1 - single pixel  8/4/2/1
	(1 << 3) | (0 << 2) |
	(0 << 1) | (0 << 0),
	(0 << 3) | (1 << 2) |
	(0 << 1) | (0 << 0),
	(0 << 3) | (0 << 2) |
	(1 << 1) | (0 << 0),
	(0 << 3) | (0 << 2) |
	(0 << 1) | (1 << 0),
		// P3 - 3-pixel   7/11/13/14
		(0 << 3) | (1 << 2) |
		(1 << 1) | (1 << 0),
		(1 << 3) | (0 << 2) |
		(1 << 1) | (1 << 0),
		(1 << 3) | (1 << 2) |
		(0 << 1) | (1 << 0),
		(1 << 3) | (1 << 2) |
		(1 << 1) | (0 << 0),
		// Pd - diagonals  9/6
		(1 << 3) | (0 << 2) |
		(0 << 1) | (1 << 0),
		(0 << 3) | (1 << 2) |
		(1 << 1) | (0 << 0)
	};
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

// Label record - structure aggregating label's cached data and calculated features
#define DFLT0 -0.0	// default unassigned value
#define DFLT0i -0	// default unassigned value
struct LR
{
	int label;
	std::string segFname, intFname;

	bool roi_disabled = false;

	// Helper objects
	std::vector <Pixel2> raw_pixels;	
	AABB aabb;	
	ConvexHull convHull;
	Contour contour;
	std::shared_ptr<Histo> aux_Histogram;
	StatsInt
		aux_PrevCount,
		aux_PrevIntens;

	// Zernike calculator may put an arbitrary number of Z_a^b terms 
	// but we output only 'NUM_ZERNIKE_COEFFS_2_OUTPUT' of them 
	static const short aux_ZERNIKE2D_ORDER = 9, aux_ZERNIKE2D_NUM_COEFS = 30;	// z00, z11, z20, z22, z31, z33, z40, z42, z44, ... ,z97, z99 - 30 items altogether 
	std::vector<std::vector<StatsReal>> fvals;
	std::vector<StatsReal> getFeatureValues (AvailableFeatures af) { return fvals [af]; }

	StatsInt pixelCountRoiArea;
	StatsReal aux_M2,
		aux_M3,
		aux_M4, 
		aux_variance;

	void init_aabb (StatsInt x, StatsInt y);
	void update_aabb (StatsInt x, StatsInt y);

	double getValue (AvailableFeatures f);
	void reduce_pixel_intensity_features();
	bool intensitiesAllZero();
	void reduce_edge_intensity_features();
};

void init_label_record (LR& lr, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity);
void update_label_record (LR& lr, int x, int y, int label, PixIntens intensity);
void reduce_neighbors (int labels_collision_radius);

// Timing
extern double totalTileLoadTime, totalFeatureReduceTime;
double test_containers1();
double test_containers2();
bool test_histogram();

// Label data
extern std::string theSegFname, theIntFname;	// Cached file names while iterating a dataset
extern std::unordered_set<int> uniqueLabels;
extern std::vector<int> sortedUniqueLabels;	// Populated in reduce()
extern std::unordered_map <int, LR> labelData;
extern std::vector<double> calcResultBuf;	// [# of labels X # of features]
extern std::unordered_map <int, std::shared_ptr<std::mutex>> labelMutexes;
