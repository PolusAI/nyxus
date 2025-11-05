#include "extrema.h"

using namespace Nyxus;

bool ExtremaFeature::required(const FeatureSet& fs)
{
	return fs.anyEnabled (ExtremaFeature::featureset);
}

ExtremaFeature::ExtremaFeature() : FeatureMethod("ExtremaFeature") 
{
	provide_features (ExtremaFeature::featureset);
}

void ExtremaFeature::calculate (LR& r, const Fsettings& s)
{
	int TopMost = r.aabb.get_ymin();
	int Lowest = r.aabb.get_ymax();
	int LeftMost = r.aabb.get_xmin();
	int RightMost = r.aabb.get_xmax();

	int TopMost_MostLeft = -1;
	int TopMost_MostRight = -1;
	int Lowest_MostLeft = -1;
	int Lowest_MostRight = -1;
	int LeftMost_Top = -1;
	int LeftMost_Bottom = -1;
	int RightMost_Top = -1;
	int RightMost_Bottom = -1;

	for (Pixel2 p : r.raw_pixels)
	{
		// Find leftmost and rightmost x-pixels of the top 
		if (p.y == TopMost && (TopMost_MostLeft == -1 || p.x < (StatsInt)TopMost_MostLeft))
			TopMost_MostLeft = p.x;
		if (p.y == TopMost && (TopMost_MostRight == -1 || p.x > (StatsInt)TopMost_MostRight))
			TopMost_MostRight = p.x;

		// Find leftmost and rightmost x-pixels of the bottom
		if (p.y == Lowest && (Lowest_MostLeft == -1 || p.x < (StatsInt)Lowest_MostLeft))
			Lowest_MostLeft = p.x;
		if (p.y == Lowest && (Lowest_MostRight == -1 || p.x > (StatsInt)Lowest_MostRight))
			Lowest_MostRight = p.x;

		// Find top and bottom y-pixels of the leftmost
		if (p.x == LeftMost && (LeftMost_Top == -1 || p.y < (StatsInt)LeftMost_Top))
			LeftMost_Top = p.y;
		if (p.x == LeftMost && (LeftMost_Bottom == -1 || p.y > (StatsInt)LeftMost_Bottom))
			LeftMost_Bottom = p.y;

		// Find top and bottom y-pixels of the rightmost
		if (p.x == RightMost && (RightMost_Top == -1 || p.y < (StatsInt)RightMost_Top))
			RightMost_Top = p.y;
		if (p.x == RightMost && (RightMost_Bottom == -1 || p.y > (StatsInt)RightMost_Bottom))
			RightMost_Bottom = p.y;
	}

	y1 = TopMost;
	x1 = TopMost_MostLeft;
	y2 = TopMost;
	x2 = TopMost_MostRight;
	y3 = RightMost_Top;
	x3 = RightMost;
	y4 = RightMost_Bottom;
	x4 = RightMost;
	y5 = Lowest;
	x5 = Lowest_MostRight;
	y6 = Lowest;
	x6 = Lowest_MostLeft;
	y7 = LeftMost_Bottom;
	x7 = LeftMost;
	y8 = LeftMost_Top;
	x8 = LeftMost;
}

void ExtremaFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader& imloader)
{
	int TopMost = r.aabb.get_ymin();
	int Lowest = r.aabb.get_ymax();
	int LeftMost = r.aabb.get_xmin();
	int RightMost = r.aabb.get_xmax();

	int TopMost_MostLeft = -1;
	int TopMost_MostRight = -1;
	int Lowest_MostLeft = -1;
	int Lowest_MostRight = -1;
	int LeftMost_Top = -1;
	int LeftMost_Bottom = -1;
	int RightMost_Top = -1;
	int RightMost_Bottom = -1;

	for (size_t i = 0; i < r.raw_pixels_NT.size(); i++)
	{
		Pixel2 p = r.raw_pixels_NT.get_at(i);

		// Find leftmost and rightmost x-pixels of the top 
		if (p.y == TopMost && (TopMost_MostLeft == -1 || p.x < (StatsInt)TopMost_MostLeft))
			TopMost_MostLeft = p.x;
		if (p.y == TopMost && (TopMost_MostRight == -1 || p.x > (StatsInt)TopMost_MostRight))
			TopMost_MostRight = p.x;

		// Find leftmost and rightmost x-pixels of the bottom
		if (p.y == Lowest && (Lowest_MostLeft == -1 || p.x < (StatsInt)Lowest_MostLeft))
			Lowest_MostLeft = p.x;
		if (p.y == Lowest && (Lowest_MostRight == -1 || p.x > (StatsInt)Lowest_MostRight))
			Lowest_MostRight = p.x;

		// Find top and bottom y-pixels of the leftmost
		if (p.x == LeftMost && (LeftMost_Top == -1 || p.y < (StatsInt)LeftMost_Top))
			LeftMost_Top = p.y;
		if (p.x == LeftMost && (LeftMost_Bottom == -1 || p.y > (StatsInt)LeftMost_Bottom))
			LeftMost_Bottom = p.y;

		// Find top and bottom y-pixels of the rightmost
		if (p.x == RightMost && (RightMost_Top == -1 || p.y < (StatsInt)RightMost_Top))
			RightMost_Top = p.y;
		if (p.x == RightMost && (RightMost_Bottom == -1 || p.y > (StatsInt)RightMost_Bottom))
			RightMost_Bottom = p.y;
	}

	y1 = TopMost;
	x1 = TopMost_MostLeft;
	y2 = TopMost;
	x2 = TopMost_MostRight;
	y3 = RightMost_Top;
	x3 = RightMost;
	y4 = RightMost_Bottom;
	x4 = RightMost;
	y5 = Lowest;
	x5 = Lowest_MostRight;
	y6 = Lowest;
	x6 = Lowest_MostLeft;
	y7 = LeftMost_Bottom;
	x7 = LeftMost;
	y8 = LeftMost_Top;
	x8 = LeftMost;
}

std::tuple<int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int> ExtremaFeature::get_values()
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

void ExtremaFeature::save_value (std::vector<std::vector<double>>& fvals)
{
	fvals [(int)Feature2D::EXTREMA_P1_Y][0] = y1;
	fvals [(int)Feature2D::EXTREMA_P1_X][0] = x1;
	fvals [(int)Feature2D::EXTREMA_P2_Y][0] = y2;
	fvals [(int)Feature2D::EXTREMA_P2_X][0] = x2;
	fvals [(int)Feature2D::EXTREMA_P3_Y][0] = y3;
	fvals [(int)Feature2D::EXTREMA_P3_X][0] = x3;
	fvals [(int)Feature2D::EXTREMA_P4_Y][0] = y4;
	fvals [(int)Feature2D::EXTREMA_P4_X][0] = x4;
	fvals [(int)Feature2D::EXTREMA_P5_Y][0] = y5;
	fvals [(int)Feature2D::EXTREMA_P5_X][0] = x5;
	fvals [(int)Feature2D::EXTREMA_P6_Y][0] = y6;
	fvals [(int)Feature2D::EXTREMA_P6_X][0] = x6;
	fvals [(int)Feature2D::EXTREMA_P7_Y][0] = y7;
	fvals [(int)Feature2D::EXTREMA_P7_X][0] = x7;
	fvals [(int)Feature2D::EXTREMA_P8_Y][0] = y8;
	fvals [(int)Feature2D::EXTREMA_P8_X][0] = x8;
}

void ExtremaFeature::extract (LR& r, const Fsettings& s)
{
	ExtremaFeature ef;
	ef.calculate (r, s);
	ef.save_value (r.fvals);
}

void ExtremaFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		extract (r, s);
	}
}

