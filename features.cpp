#include <unordered_map>
#include <vector>
#include <iostream>

// Ordered map
std::unordered_map <int, unsigned int> labelCounts;
std::unordered_map <int, int> labelMeans;
std::unordered_map <int, std::shared_ptr<std::vector<int>>> labelValues;

void clearLabelStats()
{
	labelCounts.clear();
	labelMeans.clear();
	labelValues.clear();
}

void updateLabelStats (int label, int intensity)
{
	auto it = labelMeans.find(label);
	if (it == labelMeans.end())
	{
		// Cardinality
		labelCounts[label] = 1;

		// Mean
		labelMeans[label] = intensity;

		// Median. Cache intensity values per label for the median calculation
		std::shared_ptr<std::vector<int>> ptr = std::make_shared <std::vector<int>>();
		ptr->push_back(intensity);
		labelValues[label] = ptr;
	}
	else
	{
		// Cardinality
		auto count = labelCounts[label] + 1;
		labelCounts[label] = count;

		// Mean
		auto mean = (labelMeans[label]*(count-1) + intensity) / count;
		labelMeans[label] = mean;

		// Median
		auto ptr = labelValues[label];
		ptr->push_back(intensity);
	}
}

void printLabelStats()
{
	std::cout << "\tnumber of processed labels " << labelMeans.size() << std::endl;
	
	// Print stats by label
	for (auto i = labelValues.begin(); i != labelValues.end(); i++)
	{
		auto p = *i;
		auto s = i->second;

		std::cout << i->first << " : " << s->size() << '\n';
	}
}




