#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include "sensemaker.h"

std::unordered_map <int, StatsInt> labelCounts;
std::unordered_map <int, StatsInt> labelMeans;
std::unordered_map <int, std::shared_ptr<std::unordered_set<PixIntens>>> labelValues;
std::unordered_map <int, StatsInt> labelMedians;
std::unordered_map <int, StatsInt> labelMins;
std::unordered_map <int, StatsInt> labelMaxs;
std::unordered_map <int, StatsInt> labelEnergy;

void clearLabelStats()
{
	labelCounts.clear();
	labelMeans.clear();
	labelValues.clear();
	labelMins.clear();
	labelMaxs.clear();
}

/*
 *
 * This function should be called per each pixel.
 *
 */

void update_label_stats (int label, PixIntens intensity)
{
	auto it = labelMeans.find(label);
	if (it == labelMeans.end())
	{
		// Count of pixels belonging to the label
		labelCounts[label] = 1;

		// Mean
		labelMeans[label] = intensity;

		// Median. Cache intensity values per label for the median calculation
		std::shared_ptr<std::unordered_set<PixIntens>> ptr = std::make_shared <std::unordered_set<PixIntens>>();
		ptr->insert(intensity);
		labelValues[label] = ptr;

		// Min
		labelMins[label] = intensity;

		// Max
		labelMaxs[label] = intensity;

        //Energy
        labelEnergy[label] = intensity;
    }
	else
	{
		// Count of pixels belonging to the label
        auto count = labelCounts[label];
		labelCounts[label] = count + 1;

		// Mean
		auto mean = (labelMeans[label]*(count-1) + intensity) / count;
		labelMeans[label] = mean;

		// Median
		auto ptr = labelValues[label];
		ptr->insert(intensity); 

		// Min 
		labelMins[label] = std::min (labelMins[label], (StatsInt)intensity);

		// Max
		labelMaxs[label] = std::min(labelMins[label], (StatsInt)intensity);

        //Energy
        labelEnergy[label] = labelEnergy[label] + intensity;
	}
}

/*
 * 
 * This function should be called once after a file pair processing is finished.
 *
 */

void do_partial_stats_reduction()
{
	for (auto& lv : labelValues)
	{
		// Sort unique intensities
		std::vector<int> A{ lv.second->begin(), lv.second->end() };
		std::sort (A.begin(), A.end());

		// Pick the median
		if (A.size() % 2 != 0)
		{
			int median = A[A.size() / 2];
			labelMedians[lv.first] = median;
		}
		else
		{
			int right = A[A.size() / 2],
				left = A[A.size() / 2 - 1],	// Middle left and right values
				ave = (right + left) / 2;
			labelMedians[lv.first] = ave;
		}
	}
}

/*
void OtherStatistics(double* output)  
{

    readOnlyPixels pix_plane = ReadablePixels();

    double SqrdTmpN = 0;
    double TrpdTmpN = 0;
    double QuadTmpN = 0;
    double meanAbsoluteSum = 0;

    //Energy is a measure of the magnitude of voxel values in an image.
    //A larger values implies a greater sum of the squares of these values.
    double C_Constant = 0;  //C_Constant is optional value, which shifts the intensities to prevent negative values for tmp below
    double Energy = 0;

    for (unsigned int y = 0; y < height; ++y)
        for (unsigned int x = 0; x < width; ++x) {
            if (isnan(pix_plane(y, x))) continue;
            double tmp = pix_plane(y, x);
            Energy += (tmp + C_Constant) * (tmp + C_Constant);
            double tmpN = tmp - stats.mean();
            meanAbsoluteSum += abs(tmpN);
            SqrdTmpN += tmpN * tmpN;
            TrpdTmpN += tmpN * tmpN * tmpN;
            QuadTmpN += tmpN * tmpN * tmpN * tmpN;
        }

    double Variance = SqrdTmpN / stats.n();
    double STDEV = sqrt(Variance);
    double Skewness = (TrpdTmpN / stats.n()) / pow(STDEV, 3);
    double Kurtosis = (QuadTmpN / stats.n()) / pow(Variance, 2);
    output[5] = STDEV;
    output[6] = Skewness;
    output[7] = Kurtosis;

    //Mean Absolute Deviation (MAD)
    //Mean Absolute Deviation is the mean distance of all intensity values
    //from the Mean Value of the image array.
    double meanAbsoluteDeviation = meanAbsoluteSum / stats.n();
    output[8] = meanAbsoluteDeviation;

    output[9] = Energy;

    //Root Mean Squared (RMS)
    //RMS is the square-root of the mean of all the squared intensity values.
    //It is another measure of the magnitude of the image values.
    double RMS = sqrt(Energy / stats.n());
    output[10] = RMS;

    //Make a Histogram
    int intMax = (int)ceil(stats.max());
    int intMin = (int)floor(stats.min());
    int Size = intMax - intMin + 1;
    int* histBins = new int[Size];

    for (int i = 0; i < Size; ++i) histBins[i] = 0;

    for (unsigned int y = 0; y < height; ++y)
        for (unsigned int x = 0; x < width; ++x) {
            if (!std::isnan(pix_plane(y, x))) {
                ++histBins[(int)floor(pix_plane(y, x)) - intMin];
            }
        }

    double MaxValue = 0;
    int maxBinIndex = -1;
    //Entropy specifies the uncertainty/randomness in the image values.
    //It measures the average amount of information required to encode the image values.
    double entropy = 0.0;

    //Uniformity is a measure of the sum of the squares of each intensity value.
    //This is a measure of the homogeneity of the image array, where a greater uniformity
    //implies a greater homogeneity or a smaller range of discrete intensity values.
    double uniformity = 0;

    double CumulativeBinEntry = 0;
    double P10 = 0; //10th Percentile
    double P25 = 0; //25th Percentile
    double P75 = 0; //75th Percentile
    double P90 = 0; //90th Percentile
    bool P10Flag = true;
    bool P25Flag = true;
    bool P75Flag = true;
    bool P90Flag = true;

    //Loop pver all the bins
    for (int i = 0; i < Size; i++) {
        double binEntry = (double)histBins[i] / stats.n();
        if (fabs(binEntry) < 1e-15) continue;
        entropy -= binEntry * log2(binEntry);  //if bin is not empty

        if (binEntry > MaxValue) { MaxValue = binEntry; maxBinIndex = i; }

        uniformity += binEntry * binEntry;

        CumulativeBinEntry += binEntry;

        if (CumulativeBinEntry >= 0.1 && P10Flag) {
            P10 = intMin + i;
            P10Flag = false;
        }
        if (CumulativeBinEntry >= 0.25 && P25Flag) {
            P25 = intMin + i;
            P25Flag = false;
        }
        if (CumulativeBinEntry >= 0.75 && P75Flag) {
            P75 = intMin + i;
            P75Flag = false;
        }
        if (CumulativeBinEntry >= 0.9 && P90Flag) {
            P90 = intMin + i;
            P90Flag = false;
        }
    }

    int ModeValue = maxBinIndex + intMin;

    output[11] = entropy;
    output[12] = (double)ModeValue;
    output[13] = uniformity;

    output[14] = P10;
    output[15] = P25;
    output[16] = P75;
    output[17] = P90;
    double InterquartileRange = P75 - P25;
    output[18] = InterquartileRange;

    //Robust Mean Absolute Deviation (rMAD)
    //Robust Mean Absolute Deviation is the mean distance of all intensity values from the Mean Value
    //calculated on the subset of image array with gray levels in between, or equal to the 10th and 90th percentile.
    double tmpSum = 0;
    int tmpCount = 0;
    for (unsigned int y = 0; y < height; ++y)
        for (unsigned int x = 0; x < width; ++x) {
            double tmp = pix_plane(y, x);
            if (isnan(tmp)) continue;
            if (tmp >= P10 && tmp <= P90) { tmpSum += tmp; tmpCount++; }
        }

    double rMADMean = tmpSum / tmpCount;

    tmpSum = 0;
    for (unsigned int y = 0; y < height; ++y)
        for (unsigned int x = 0; x < width; ++x) {
            double tmp = pix_plane(y, x);
            if (isnan(tmp)) continue;
            if (tmp >= P10 && tmp <= P90) { tmpSum += abs(tmp - rMADMean); }
        }

    double rMAD = tmpSum / tmpCount;
    output[19] = rMAD;

    delete[] histBins;
}

*/



