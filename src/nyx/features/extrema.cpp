#include "extrema.h"


ExtremaFeatures::ExtremaFeatures (const std::vector<Pixel2>& roi_pixels)
{
	int TopMostIndex = -1;
	int LowestIndex = -1;
	int LeftMostIndex = -1;
	int RightMostIndex = -1;

	for (auto& pix : roi_pixels)
	{
		if (TopMostIndex == -1 || pix.y < (StatsInt)TopMostIndex)
			TopMostIndex = pix.y;
		if (LowestIndex == -1 || pix.y > (StatsInt)LowestIndex)
			LowestIndex = pix.y;

		if (LeftMostIndex == -1 || pix.x < (StatsInt)LeftMostIndex)
			LeftMostIndex = pix.x;
		if (RightMostIndex == -1 || pix.x > (StatsInt)RightMostIndex)
			RightMostIndex = pix.x;
	}

	int TopMost_MostLeftIndex = -1;
	int TopMost_MostRightIndex = -1;
	int Lowest_MostLeftIndex = -1;
	int Lowest_MostRightIndex = -1;
	int LeftMost_Top = -1;
	int LeftMost_Bottom = -1;
	int RightMost_Top = -1;
	int RightMost_Bottom = -1;

	for (auto& pix : roi_pixels)
	{
		// Find leftmost and rightmost x-pixels of the top 
		if (pix.y == TopMostIndex && (TopMost_MostLeftIndex == -1 || pix.x < (StatsInt)TopMost_MostLeftIndex))
			TopMost_MostLeftIndex = pix.x;
		if (pix.y == TopMostIndex && (TopMost_MostRightIndex == -1 || pix.x > (StatsInt)TopMost_MostRightIndex))
			TopMost_MostRightIndex = pix.x;

		// Find leftmost and rightmost x-pixels of the bottom
		if (pix.y == LowestIndex && (Lowest_MostLeftIndex == -1 || pix.x < (StatsInt)Lowest_MostLeftIndex))
			Lowest_MostLeftIndex = pix.x;
		if (pix.y == LowestIndex && (Lowest_MostRightIndex == -1 || pix.x > (StatsInt)Lowest_MostRightIndex))
			Lowest_MostRightIndex = pix.x;

		// Find top and bottom y-pixels of the leftmost
		if (pix.x == LeftMostIndex && (LeftMost_Top == -1 || pix.y < (StatsInt)LeftMost_Top))
			LeftMost_Top = pix.y;
		if (pix.x == LeftMostIndex && (LeftMost_Bottom == -1 || pix.y > (StatsInt)LeftMost_Bottom))
			LeftMost_Bottom = pix.y;

		// Find top and bottom y-pixels of the rightmost
		if (pix.x == RightMostIndex && (RightMost_Top == -1 || pix.y < (StatsInt)RightMost_Top))
			RightMost_Top = pix.y;
		if (pix.x == RightMostIndex && (RightMost_Bottom == -1 || pix.y > (StatsInt)RightMost_Bottom))
			RightMost_Bottom = pix.y;
	}

	y1 = TopMostIndex;
	x1 = TopMost_MostLeftIndex;

	y2 = TopMostIndex;
	x2 = TopMost_MostRightIndex;

	y3 = RightMost_Top;
	x3 = RightMostIndex;

	y4 = RightMost_Bottom;
	x4 = RightMostIndex;

	y5 = LowestIndex;
	x5 = Lowest_MostRightIndex;

	y6 = LowestIndex;
	x6 = Lowest_MostLeftIndex;

	y7 = LeftMost_Bottom;
	x7 = LeftMostIndex;

	y8 = LeftMost_Top;
	x8 = LeftMostIndex;
}

std::tuple<int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int> ExtremaFeatures::get_values()
{
	return 
	{ 
		x1, y1, 
		x2, y2, 
		x3, y3, 
		x4, y4, 
		x5, y5, 
		x6, y6, 
		x7, y7, 
		x8, y8 
	};
}

void ExtremaFeatures::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		ExtremaFeatures ef (r.raw_pixels);
		auto [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8] = ef.get_values();
		r.fvals[EXTREMA_P1_Y][0] = y1;
		r.fvals[EXTREMA_P1_X][0] = x1;
		r.fvals[EXTREMA_P2_Y][0] = y2;
		r.fvals[EXTREMA_P2_X][0] = x2;
		r.fvals[EXTREMA_P3_Y][0] = y3;
		r.fvals[EXTREMA_P3_X][0] = x3;
		r.fvals[EXTREMA_P4_Y][0] = y4;
		r.fvals[EXTREMA_P4_X][0] = x4;
		r.fvals[EXTREMA_P5_Y][0] = y5;
		r.fvals[EXTREMA_P5_X][0] = x5;
		r.fvals[EXTREMA_P6_Y][0] = y6;
		r.fvals[EXTREMA_P6_X][0] = x6;
		r.fvals[EXTREMA_P7_Y][0] = y7;
		r.fvals[EXTREMA_P7_X][0] = x7;
		r.fvals[EXTREMA_P8_Y][0] = y8;
		r.fvals[EXTREMA_P8_X][0] = x8;
	}
}

