#include "histogram.h"

OnlineHistogram::OnlineHistogram()
{
	for (int i = 0; i < N_HISTO_BINS + 1; i++)
	{
		binCounts.push_back(-1);
		binEdges.push_back(-1);
	}
}

void OnlineHistogram::reset()
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
void OnlineHistogram::print(bool logScale, std::string header_text)
{
	std::cout << header_text << "\n";
	if (logScale)
		std::cout << "\tLogarithmic scale\n";
	for (int i = 0; i <= N_HISTO_BINS; i++)
	{
		std::cout << "\t[" << i << "]\t" << binEdges[i];
		std::cout << " = " << binCounts[i] << "\t\t";

		int binCnt = logScale ? int(10 * std::log(binCounts[i]) + 0.5) : binCounts[i];
		for (int k = 0; k < binCnt; k++)
			std::cout << '*';

		std::cout << "\n";
	}
}

// Appends a value to the histogram
void OnlineHistogram::add_observation(HistoItem x)
{
	// Put 'x' in the set
	uniqIntensities.insert(x);

	// Increment x's counter
	if (intensityCounts.find(x) == intensityCounts.end())
		intensityCounts[x] = 1;
	else
		intensityCounts[x] = intensityCounts[x] + 1;
}

void OnlineHistogram::build_histogram()
{
	bool notYetInit = true;
	binW = 0, min_ = 0, max_ = 0;	// Default to some value
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

	binW = (max_ - min_) / double(N_HISTO_BINS);

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
		int idx = int((x - min_) / binW);
		if (idx >= N_HISTO_BINS)
			idx = N_HISTO_BINS;
		binCounts[idx] = binCounts[idx] + 1;
	}
}

HistoItem OnlineHistogram::get_median()
{
	// --Sort unique intensities
	std::vector<int> A{ uniqIntensities.begin(), uniqIntensities.end() };
	std::sort(A.begin(), A.end());

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

HistoItem OnlineHistogram::get_mode()
{
	int maxCnt = -1;
	HistoItem mode;
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
std::tuple<HistoItem, HistoItem, double, double, double, double, double, double, double, double> OnlineHistogram::get_stats()
{
	HistoItem median = get_median();
	HistoItem mode = get_mode();

	double p10 = 0, p25 = 0, p75 = 0, p90 = 0, iqr = 0, rmad = 0, entropy = 0, uniformity = 0;

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

	// Correction for the case where no items fall into the 10-90% range 
	if (n == 0)
		n = 1;

	mean /= n;

	double sum = 0.0;
	for (auto x : uniqIntensities)
		if (x >= lowBound && x <= uprBound)
		{
			auto absDelta = std::abs(intensityCounts[x] - mean);
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