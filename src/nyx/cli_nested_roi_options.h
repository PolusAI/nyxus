#pragma once

#include <string>
#include <vector>

class NestedRoiOptions
{
public:
	// Parses 'raw*', set 'defined_' and 'ermsg'
	bool parse_input();

	// True if all the options are non-empty and parsed without errors
	bool defined();

	// True if all the options were given some values and parse_input() can be called
	bool empty();

	std::string get_last_er_msg();

	std::string rawChannelSignature, rawParentChannelNo, rawChildChannelNo, rawAggregationMethod;

	const std::string& channel_signature() const { return channelSignature_; }
	const int parent_channel_number() const { return parentChannelNo; }
	const int child_channel_number() const { return childChannelNo; }

	enum Aggregations { aNONE = 0, aSUM, aMEAN, aMIN, aMAX, aWMA };
	const Aggregations aggregation_method() const { return aggrMethod; }

private:
	bool defined_ = false;
	std::string ermsg = "";

	std::string channelSignature_ = "",	// matches NESTEDROI_CHNL_SIGNATURE
		parentChannel_ = "",			// matches NESTEDROI_PARENT_CHNL
		childChannel_ = "",				// matches NESTEDROI_CHILD_CHNL
		aggregationMethod_ = "";		// matches NESTEDROI_AGGREGATION_METHOD

	int parentChannelNo, childChannelNo;

	Aggregations aggrMethod = Aggregations::aNONE;
};