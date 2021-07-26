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
int ingestDataset (std::vector<std::string> & intensFiles, std::vector<std::string> & labelFiles, int numFastloaderThreads, int numSensemakerThreads, std::string outputDir);
bool save_features (std::string inputFpath, std::string outputDir);
void showCmdlineHelp();
int checkAndReadDataset(
	// input
	std::string& dirIntens, std::string& dirLabels, std::string& dirOut,
	// output
	std::vector<std::string>& intensFiles, std::vector<std::string>& labelFiles);

using PixIntens = unsigned int;
using StatsInt = unsigned long;
using StatsReal = double;
using Histo = OnlineHistogram<PixIntens>;

void init_feature_buffers();
void update_label_stats (int x, int y, int label, PixIntens intensity);
void update_label_stats_parallel (int x, int y, int label, PixIntens intensity);
void print_label_stats();
void print_by_label(const char* featureName, std::unordered_map<int, StatsInt> L, int numColumns = 8); 
void print_by_label(const char* featureName, std::unordered_map<int, StatsReal> L, int numColumns = 4);
void clearLabelStats();
void reduce_all_labels();

struct Pixel2
{
	StatsInt x, y;
	PixIntens inten;
	Pixel2(StatsInt x_, StatsInt y_, PixIntens i_): x(x_), y(y_), inten(i_) {}
	bool operator == (const Pixel2& p2)
	{
		return this->x == p2.x && this->y == p2.y;
	}
};

// Label record - structure aggregating label's running statistics and sums
struct LR
{
	//==== Pixel intensity statistics
	StatsInt pixelCount;
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
	StatsReal labelRMS;
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
	StatsReal	maxFeretDiameter;
	StatsReal	maxFeretAngle;
	StatsReal	minFeretDiameter;
	StatsReal	minFeretAngle; 

	std::vector <Pixel2> raw_pixels;
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

inline int spat_hash_2d (StatsInt x, StatsInt y, int m)
{
	auto h = x * 73856093;
	h = h ^ y * 19349663;
	// hash   hash  z × 83492791	// For the future
	// hash   hash  l × 67867979
	auto retval = h % m;
	return retval;
}
