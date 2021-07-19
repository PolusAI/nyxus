#pragma once

//#include <cmath> //#include <math.h>
#include <algorithm> 
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <tuple>

#define N_HISTO_BINS 20	// Max number of histogram channels. 5% step.

template <class PixIntens>
class OnlineHistogram0
{
public:
	OnlineHistogram0() :
		numObs(0),
		useCounts2(false)
	{
		// Reserve buffers for N_HIST_CHANNELS channels
		counts.reserve(N_HISTO_BINS);
		counts2.reserve(N_HISTO_BINS);
	}
	
	void reset()
	{
		numObs = 0;
		counts.clear();
		counts2.clear();
		useCounts2 = false;
		left_bounds.clear();
		right_bounds.clear();
	}
	
	// Diagnostic output
	void print(std::string remark = "")
	{
		unsigned int totAllBins = 0;
		for (int i = 0; i < N_HISTO_BINS; i++)
			totAllBins += counts[i];

		std::cout << "\nHistogram " << remark << std::endl << numObs << " observations, all bins total = " << totAllBins << std::endl;
		for (int i = 0; i < counts.size(); i++)
		{
			std::cout << '[' << i << "]\t" << left_bounds[i] << " -- " << right_bounds[i] << "\t\t" << counts[i] << std::endl;
		}
	}

	// Appends a value to the histogram
	void add_observation (PixIntens x)
	{
		// Count observations for the sanity check
		numObs++;

		if (counts.size() == 0)
		{
			// No histogram initially. Allocate
			double l = x,
				r = l + 1,
				s = (r - l) / double(N_HISTO_BINS);
			for (int i = 0; i < N_HISTO_BINS; i++)
			{
				double a = l + s * i,
					b = a + s;
				left_bounds.push_back(a);
				right_bounds.push_back(b);
				counts.push_back(0);
				counts2.push_back(0);
			}
			// Find the bin
			int bi = findBinIndex(x);
			if (bi < 0 || bi >= N_HISTO_BINS)	// Sanity check
				throw std::exception();
			counts[bi] = counts[bi] + 1;
		}
		else
		{
			// Regular histogram

			// Find the bin
			int bi = findBinIndex(x);
			if (bi >= 0)	// No need to extend the bounds
				counts[bi] = counts[bi] + 1;
			else	// We should extend the histogram
			{
				// Default new bounds same as old ones
				double newL = left_bounds[0],
					newR = right_bounds[N_HISTO_BINS - 1];

				// New (extended) bounds
				switch (bi)
				{
				case -1:
					// Insert to the right
					newR = x + 1;
					break;
				case -2:
					// Insert to the left
					newL = x;
					break;
				default:
					// Something insane. Throw an exception
					throw std::exception();
					break;
				}

				// New step
				double newS = (newR - newL) / double(N_HISTO_BINS);

				// Transfer the old -> new histogram 

				// --Step 1
				std::fill(counts2.begin(), counts2.end(), 0); // Zero the helper histogram's bins

				// --Step 2
				for (int j = 0; j < N_HISTO_BINS; j++)	// Iterate new bins
				{
					double new_l = RoundN (double(j) * newS),
						new_r = RoundN (new_l + newS);

					for (int g = 0; g < N_HISTO_BINS; g++)	// Iterate old bins
					{
						double old_l = left_bounds[g],
							old_r = right_bounds[g];

						// Case 1 -- an entire old bin goes into a new bin
						if (old_l >= new_l && old_r <= new_r)
						{
							// No split bin, we swallow it all
							counts2[j] = counts2[j] + counts[g];
							continue;
						}

						// Case 2 -- an old bin is split between new bins
						if (old_l < new_l && new_l < old_r)
						{
							// We need to split bin 'g'
							double share = old_r - new_l;
							double frac = 1.0; // (old_r - old_l) / share;
							counts2[j] = counts2[j] + counts[g] * frac;
						}
					}
				}

				// --Step 3
				counts = counts2;

				// --Step 4
				for (int i = 0; i < N_HISTO_BINS; i++)
				{
					double a = RoundN (newL + newS * double(i)),
						b = RoundN (a + newS);
					left_bounds[i] = a;
					right_bounds[i] = b;
				}

				// --Step 5: increment the counter
				int bi = findBinIndex (x);
				if (bi < 0 || bi >= N_HISTO_BINS)	// Sanity check
					throw std::exception();
				counts[bi] = counts[bi] + 1;
			}
		}
	}

	// Returns bin's number of items 
	double getBinStats(int binIdx)
	{
		return counts[binIdx];
	}

	// Returns bin's central value 
	double getBinValue(int binIdx)
	{
		double val = (left_bounds[binIdx] + right_bounds[binIdx]) / 2.0;
		return val;
	}

	// Returns estimated robust mean absolute deviation
	double getEstRMAD (int lbound = 10, int rbound = 90)
	{
		if (lbound < 0 || lbound >= 50)
			lbound = 10;
		if (rbound <= 50 || rbound > 100)
			rbound = 90;

		double percentPerBin = 100.f / float(N_HISTO_BINS);

		double robSum = 0.0;
		int robCount = 0;

		// Calculate robust mean
		for (int i = 0; i < N_HISTO_BINS; i++)
		{
			double binSlice = percentPerBin * i;
			if (binSlice >= lbound && binSlice <= rbound)
			{
				robSum += getBinValue(i);
				robCount += counts[i];
			}
		}

		// Empty histogram?
		if (robCount == 0)
			return 0.0;

		double robMean = robSum / robCount;

		// Calculate robust MAD
		robSum = 0.0;
		for (int i = 0; i < N_HISTO_BINS; i++)
		{
			double binSlice = percentPerBin * i;
			if (binSlice >= lbound && binSlice <= rbound)
			{
				robSum += std::abs (getBinValue(i) - robMean);
			}
		}
		double robMAD = robSum / robCount;
		return robMAD;
	}

protected:
	// Returns the bin index for the value 'x'. In case of an error, returns an (illegal) negative bin index
	int findBinIndex(PixIntens x)
	{
		// x is to the left from the current histogram?
		if (x < left_bounds[0])
			return -2;

		// x is to the right from the current histogram?
		if (x > right_bounds[N_HISTO_BINS - 1])
			return -1;

		// Brute force search
		for (int i = 0; i < N_HISTO_BINS; i++)
			if (left_bounds[i] <= x && x <= right_bounds[i])
				return i;

		// Sanity
		throw std::exception();
		return -3;
	}

	// Helper function
	inline double RoundN(double x)
	{
		const double roundFactor = 1000.0;
		auto retval = std::round(x * roundFactor) / roundFactor;
		return retval;
	}

	// Buffers and state variables
	unsigned int numObs;
	std::vector <float> counts, counts2;
	bool useCounts2;
	std::vector <double> left_bounds, right_bounds;
};



template <class PixIntens>
class OnlineHistogram
{
public:
	OnlineHistogram() 
	{
		for (int i = 0; i < N_HISTO_BINS + 1; i++)
		{
			binCounts.push_back(-1);
			binEdges.push_back(-1);
		}
	}

	void reset()
	{
		uniqIntensities.clear();
		intensityCounts.clear();		

		for (int i = 0; i < N_HISTO_BINS + 1; i++)
		{
			binCounts[i] = -1;
			binEdges[i] = -1;
		}
	}

	// Diagnostic output
	void print(std::string remark = "")
	{
		for (int i = 0; i <= N_HISTO_BINS; i++)
		{
			std::cout << "\t[" << i << "] " << binEdges[i];
			std::cout << " = " << binCounts[i] << "\t";

			for (int k = 0; k < binCounts[i]; k++)
				std::cout << '*';

			std::cout << std::endl;
		}
	}

	// Appends a value to the histogram
	void add_observation (PixIntens x)
	{
		// Put 'x' in the set
		uniqIntensities.insert(x);

		// Increment x's counter
		if (intensityCounts.find(x) == intensityCounts.end())
			intensityCounts[x] = 1;
		else
			intensityCounts[x] = intensityCounts[x] + 1;
	}

	void build_histogram()
	{
		bool notYetInit = true;
		binW =0, min_ =0, max_ =0;	// Default to some value
		for (auto x : uniqIntensities)
			if (notYetInit)
			{
				// Initialize the range
				min_ = max_ = x;
				notYetInit = false;
			}
			else
			{
				min_ = std::min(min_, x);
				max_ = std::max(max_, x);
			}

		// Fix max to prevent index overflow
		max_ += 1;

		binW = (max_ - min_) / N_HISTO_BINS;

		// Zero bin counts
		for (int i = 0; i < N_HISTO_BINS; i++)
			binCounts[i] = 0;

		// Calculate bin edges
		binEdges[0] = min_;
		binEdges[N_HISTO_BINS] = max_;
		for (int i = 1; i < N_HISTO_BINS; i++)
			binEdges[i] = min_ + binW * i;

		for (auto x : uniqIntensities)
		{
			auto n = intensityCounts[x];
			int idx = int( (x - min_) / binW );
			if (idx >= N_HISTO_BINS)
				idx = N_HISTO_BINS;
			binCounts[idx] = binCounts[idx] + 1;
		}
	}

	PixIntens get_median()
	{
		// --Sort unique intensities
		std::vector<int> A { uniqIntensities.begin(), uniqIntensities.end() };
		std::sort (A.begin(), A.end());

		// --Pick the median
		if (A.size() % 2 != 0)
		{
			int median = A[A.size() / 2];
			return median;
		}
		else
		{
			int right = A[A.size() / 2],
				left = A[A.size() / 2 - 1],	// Middle left and right values
				ave = (right + left) / 2;
			return ave;
		}
	}

	PixIntens get_mode()
	{
		int maxCnt = -1;
		PixIntens mode;
		for (auto it = intensityCounts.begin(); it != intensityCounts.end(); it++)
		{
			if (maxCnt < 0)
			{
				mode = it->first;
				maxCnt = it->second;
			}
			else
				if (maxCnt < it->second)
				{
					mode = it->first;
					maxCnt = it->second;
				}
		}

		return mode;
	}

	// Returns
	//	[0] median
	// 	[1] mode
	//	[2-5] p10, p25, 075, p90
	//	[6] IQR
	//	[7] RMAD
	//	[8] entropy
	//	[9] uniformity
	std::tuple<PixIntens, PixIntens, double, double, double, double, double, double, double, double> get_stats()
	{
		PixIntens median = get_median();
		PixIntens mode = get_mode();

		double p10 =0, p25 =0, p75 =0, p90 =0, iqr =0, rmad =0, entropy =0, uniformity =0;

		// %
		auto ppb = 100.0 / N_HISTO_BINS;
		for (int i = 0; i < N_HISTO_BINS; i++)
		{
			if (ppb * i <= 10)
				p10 += binCounts[i];
			if (ppb * i <= 25)
				p25 += binCounts[i];
			if (ppb * i <= 75)
				p75 += binCounts[i];
			if (ppb * i <= 90)
				p90 += binCounts[i];
		}

		// IQR
		iqr = p75 - p25;

		// RMAD 10-90 %
		double range = max_ - min_;
		double lowBound = min_ + range * 0.1, 
			uprBound = min_ + range * 0.9; 
		double mean = 0.0;
		int n = 0;
		for (auto x : uniqIntensities)
			if (x >= lowBound && x <= uprBound)
			{
				auto cnt = intensityCounts[x];
				mean += x * cnt;
				n += cnt;
			}
		mean /= n;

		double sum = 0.0;
		for (auto x : uniqIntensities)
			if (x >= lowBound && x <= uprBound)
			{
				auto absDelta = std::abs (intensityCounts[x] - mean);
				sum += absDelta;
			}
		rmad = sum / n;

		// entropy & uniformity
		entropy = 0.0;
		uniformity = 0.0;
		for (auto x : uniqIntensities)
		{
			auto cnt = intensityCounts[x];
			double binEntry = cnt / n;
			if (fabs(binEntry) < 1e-15)
				continue;

			// entropy
			entropy -= binEntry * log2(binEntry);  //if bin is not empty

			// uniformity
			uniformity += binEntry * binEntry;		
		}

		// Finally
		return { median, mode, p10, p25, p75, p90, iqr, rmad, entropy, uniformity };
	}


protected:
	std::unordered_set <PixIntens> uniqIntensities;
	std::unordered_map <PixIntens, int> intensityCounts;
	std::vector <int> binCounts; // Populated in build_histogram(). Always has size N_HISTO_BINS
	std::vector <double> binEdges; // Populated in build_histogram(). Always has size (N_HISTO_BINS+1)
	PixIntens min_, max_;
	double binW;
};


