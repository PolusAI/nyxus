#pragma once

#include <string>
#include <vector>

class GaborOptions
{
public:
	// Parses 'raw*', set 'defined_' and 'ermsg'
	bool parse_input();	

	// Returns true if all 'raw*' are empty
	bool empty();

	std::string get_summary_text();
	std::string get_last_er_msg();

	// intentionally public to be accessed by Environment
	std::string rawFreqs = "",	// matches GABOR_FREQS
		rawGamma = "",			// matches GABOR_GAMMA
		rawSig2lam = "",		// matches GABOR_SIG2LAM
		rawKerSize = "",		// matches GABOR_KERSIZE
		rawF0 = "",				// matches GABOR_F0
		rawTheta = "",			// matches GABOR_THETA
		rawGrayThreshold = "";	// matches GABOR_THRESHOLD

private:
	bool defined_ = false;
	std::string ermsg = "";

	float gamma;
};