#pragma once

//---	#define SINGLE_ROI_TEST

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include "histogram.h"

bool datasetDirsOK (std::string & dirIntens, std::string & dirLab, std::string & dirOut);
bool directoryExists (const std::string & dir);
void readDirectoryFiles (const std::string & dir, std::vector<std::string> & files);
bool scanViaFastloader (const std::string & fpath, int num_threads);
bool scanFilePair (const std::string& intens_fpath, const std::string& label_fpath, int num_fastloader_threads, int num_sensemaker_threads);
bool scanFilePairParallel (const std::string& intens_fpath, const std::string& label_fpath, int num_fastloader_threads, int num_sensemaker_threads);
bool TraverseViaFastloader1 (const std::string& fpath, int num_threads);
std::string getPureFname(std::string fpath);
int ingestDataset (std::vector<std::string> & intensFiles, std::vector<std::string> & labelFiles, int numFastloaderThreads, int numSensemakerThreads, int min_online_roi_size, std::string outputDir);
bool save_features (std::string inputFpath, std::string outputDir);
void showCmdlineHelp();
int checkAndReadDataset(
	// input
	std::string& dirIntens, std::string& dirLabels, std::string& dirOut,
	// output
	std::vector<std::string>& intensFiles, std::vector<std::string>& labelFiles);

using PixIntens = unsigned int;
using StatsInt = long;
using StatsReal = double;
using Histo = OnlineHistogram<PixIntens>;

void init_feature_buffers();
void update_label_stats (int x, int y, int label, PixIntens intensity);
void update_label_stats_parallel (int x, int y, int label, PixIntens intensity);
void print_label_stats();
void print_by_label(const char* featureName, std::unordered_map<int, StatsInt> L, int numColumns = 8); 
void print_by_label(const char* featureName, std::unordered_map<int, StatsReal> L, int numColumns = 4);
void clearLabelStats();
void reduce_all_labels(int min_online_roi_size);

template <typename T>
struct Point2
{
	T x, y;
	Point2 (T x_, T y_): x(x_), y(y_) {}
	Point2(): x(0), y(0) {}

	double normL2() const { return sqrt(x*x+y*y); }

	Point2 operator - ()
	{
		Point2 p2(-(this->x), -(this->y));
		return p2;
	}
	Point2 operator - (const Point2& v)
	{
		Point2 p2(this->x - v.x, this->y - v.y);
		return p2;
	}
	Point2 operator + (const Point2& v)
	{
		Point2 p2(this->x + v.x, this->y + v.y);
		return p2;
	}
	Point2 operator / (float k)
	{
		Point2 p2(this->x / k, this->y / k);
		return p2;
	}
};

using Point2i = Point2<StatsInt>;
using Point2f = Point2<float>;
inline double normL2(const Point2f& p) { return p.normL2(); }

struct Pixel2: public Point2i
{
	PixIntens inten;
	Pixel2(StatsInt x_, StatsInt y_, PixIntens i_): Point2(x_, y_), inten(i_) {}
	
	bool operator == (const Pixel2& p2)
	{
		return this->x == p2.x && this->y == p2.y;
	}
	Pixel2 operator - ()
	{
		Pixel2 p2(-(this->x), -(this->y), this->inten);
		return p2;
	}
	Pixel2 operator - (const Pixel2& v) const
	{
		Pixel2 p2(this->x - v.x, this->y - v.y, this->inten);
		return p2;
	}
	Pixel2 operator + (const Pixel2& v) const
	{
		Pixel2 p2(this->x + v.x, this->y + v.y, this->inten);
		return p2;
	}
	Pixel2 operator / (float k) const
	{
		Pixel2 p2(this->x / k, this->y / k, this->inten);
		return p2;
	}
	Pixel2 operator * (float k) const
	{
		Pixel2 p2(this->x * k, this->y * k, this->inten);
		return p2;
	}
	operator Point2f () const { Point2f p(this->x, this->y); return p; }
};

//Pixel2 operator + (const Pixel2& v1, const Pixel2& v2)
//{
//	Pixel2 p2(v1.x + v2.x, v1.y + v2.y, (v1.inten+v2.inten)/2);
//	return p2;
//}

// Inherited from WNDCHRM, used for Feret and Martin statistics calculation
struct Statistics 
{
	int min, max, mode;
	double mean, median, stdev;
};

Statistics ComputeCommonStatistics2 (std::vector<double>& Data);

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


// Label record - structure aggregating label's running statistics and sums
struct LR
{
	std::vector <Pixel2> raw_pixels;	
	
	//==== Pixel intensity statistics

	StatsInt pixelCount;	// Area
	StatsInt labelPrevCount;
	StatsInt labelPrevIntens;
	StatsReal labelMeans;
	//std::shared_ptr<std::unordered_set<PixIntens>> labelUniqueIntensityValues;
	StatsInt labelMedians;
	StatsInt labelMins;
	StatsInt labelMaxs;
	StatsInt labelMassEnergy;
	StatsReal labelVariance;
	StatsReal labelStddev;	
	StatsReal centroid_x;
	StatsReal centroid_y;
	StatsReal labelM2;
	StatsReal labelM3;
	StatsReal labelM4;
	StatsReal labelSkewness;
	StatsReal labelKurtosis;
	StatsReal labelMAD;
	StatsReal labelRMS;		// Root Mean Squared (RMS) is the square-root of the mean of all the squared intensity values. It is another measure of the magnitude of the image values.
	std::shared_ptr<Histo> labelHistogram;
	StatsReal labelP10;
	StatsReal labelP25;
	StatsReal labelP75;
	StatsReal labelP90;
	StatsReal labelIQR;
	StatsReal labelEntropy;
	StatsReal labelMode;
	StatsReal labelUniformity;
	StatsReal labelRMAD;

	//==== Morphology
	StatsInt aabb_xmin;
	StatsInt aabb_xmax;
	StatsInt aabb_ymin;
	StatsInt aabb_ymax;
	int num_neighbors;
	void init_aabb (StatsInt x, StatsInt y);
	void update_aabb (StatsInt x, StatsInt y);
	StatsReal extent;	// Calculated in reduce(), not accumulated online

	// --ellipse related
	StatsReal major_axis_length;
	StatsReal minor_axis_length;
	StatsReal eccentricity;
	StatsReal orientation;

	StatsReal	aspectRatio;
	StatsReal	equivDiam;
	StatsInt	perimeter;
	StatsReal	circularity;
	StatsReal	convHullArea;
	StatsReal	solidity;

	// --extrema
	StatsReal	
		extremaP1y, extremaP1x,
		extremaP2y, extremaP2x,
		extremaP3y, extremaP3x,
		extremaP4y, extremaP4x,
		extremaP5y, extremaP5x,
		extremaP6y, extremaP6x,
		extremaP7y, extremaP7x,
		extremaP8y, extremaP8x;

	// --Feret
	StatsReal	maxFeretDiameter,
		maxFeretAngle,
		minFeretDiameter,
		minFeretAngle,
		feretStats_minDiameter,	// ratios[59]
		feretStats_maxDiameter,	// ratios[60]
		feretStats_meanDiameter,	// ratios[61]
		feretStats_medianDiameter,	// ratios[62]
		feretStats_stdDiameter,	// ratios[63]
		feretStats_modeDiameter;	// ratios[64]

	// --Martin
	StatsReal	
		martinStats_minDiameter,	// ratios[59]
		martinStats_maxDiameter,	// ratios[60]
		martinStats_meanDiameter,	// ratios[61]
		martinStats_medianDiameter,	// ratios[62]
		martinStats_stdDiameter,	// ratios[63]
		martinStats_modeDiameter;	// ratios[64]

	// --Nassenstein
	StatsReal
		nassStats_minDiameter,	// ratios[59]
		nassStats_maxDiameter,	// ratios[60]
		nassStats_meanDiameter,	// ratios[61]
		nassStats_medianDiameter,	// ratios[62]
		nassStats_stdDiameter,	// ratios[63]
		nassStats_modeDiameter;	// ratios[64]

	// --Euler
	long euler_number;

	// --hexagonality & polygonality
	double polygonality_ave, hexagonality_ave, hexagonality_stddev;

	// --Circle fitting
	double diameter_min_enclosing_circle;
};

void init_label_record(LR& lr, int x, int y, int label, PixIntens intensity);
void update_label_record(LR& lr, int x, int y, int label, PixIntens intensity);
void reduce_neighbors (int labels_collision_radius);

extern std::unordered_map <int, LR> labelData;
extern std::unordered_map <int, std::shared_ptr<std::mutex>> labelMutexes;
extern std::unordered_set <int> uniqueLabels; // Relates to a single intensity-label file pair


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
	bool retval = aabbNoOverlap(r1.aabb_xmin, r1.aabb_xmax, r1.aabb_ymin, r1.aabb_ymax,
		r2.aabb_xmin, r2.aabb_xmax, r2.aabb_ymin, r2.aabb_ymax, radius); 
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
