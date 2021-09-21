//
// Part of future G-test of Sensemaker
//

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <unordered_map>
#include <unordered_set>

#include "histogram.h"
#include "sensemaker.h"
#include "version.h"

bool test_histogram()
{
	int data [] = {
			239,	237,	257,	256,	250,
			296,	297,	298,	325,	357,
			346,	304,	333,	334,	263,
			321,	330,	335,	324,	360,
			407,	395,	372,	380,	363,
			311,	268,	369,	377,	367,
			379,	419,	417,	401,	395,
			412,	379,	339,	316,	396,
			386,	417,	440,	454,	441,
			431,	438,	406,	381,	352,
			313,	333,	375,	428,	430,
			443,	423,	396,	357,	328,
			322,	302,	321,	312,	304,
			277,	266,	205,	205,	208
	};

	using Histo2 = OnlineHistogram;

	auto h = std::make_shared <Histo2>();
	for (auto x : data)
		h->add_observation(x);
	h->build_histogram();
	h->print(true, "");
	auto [mean_, mode_, p10_, p25_, p75_, p90_, iqr_, rmad_, entropy_, uniformity_] = h->get_stats();
	return true;
}
