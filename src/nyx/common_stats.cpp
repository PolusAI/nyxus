#include <algorithm>
#include "helpers/helpers.h"

Statistics ComputeCommonStatistics2 (std::vector<double> & Data) {

    Statistics output;

    //==== Do we have a degenerate case?
    if (Data.size() == 0)
    {
        output.max = output.min = output.mean = output.median = output.mode = output.stdev = 0.0;
        return output;
    }

    //==== Process a benign case

    output.max = *max_element(Data.begin(), Data.end());
    output.min = *min_element(Data.begin(), Data.end());

    double sum = 0;
    for (int i = 0; i < Data.size(); i++) sum += Data[i];
    output.mean = sum / Data.size();

    double sumSqrd = 0;
    for (int i = 0; i < Data.size(); i++) sumSqrd += (Data[i] - output.mean) * (Data[i] - output.mean);
    output.stdev = sqrt(sumSqrd / Data.size());

    //Make a Histogram
    int intMax = (int)ceil(output.max);
    int intMin = (int)floor(output.min);
    int binCounts = intMax - intMin + 1;

    std::vector<int> histBins(binCounts, 0); //--Avoid pointers-- int* histBins = new int[binCounts];

    for (int i = 0; i < binCounts; ++i) histBins[i] = 0;
    for (int i = 0; i < Data.size(); i++) ++histBins[(int)Data[i] - intMin];

    double MaxValue = 0;
    int maxBinIndex = -1;
    //Loop over all the bins
    for (int i = 0; i < binCounts; i++) {
        if (histBins[i] > MaxValue) { MaxValue = histBins[i]; maxBinIndex = i; }
    }
    output.mode = maxBinIndex + intMin;

    //--Avoid pointers-- delete[] histBins;

    std::sort(Data.begin(), Data.end());
    double median;

    size_t half = Data.size() / 2;
    if (Data.size() % 2 == 0) {
        nth_element(Data.begin(), Data.begin() + half, Data.end());
        median = Data[half];
        nth_element(Data.begin(), Data.begin() + half - 1, Data.end());
        median += (Data[half - 1]);
        median /= 2.0;
    }
    else {
        nth_element(Data.begin(), Data.begin() + half, Data.end());
        median = Data[half];
    }

    output.median = median;

    return output;
}
