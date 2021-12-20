#include "fractal_dim.h"
#include "image_matrix.h"

FractalDimension::FractalDimension (const std::vector<Pixel2>& cloud, const AABB& aabb)
{
	Power2PaddedImageMatrix pim (cloud, aabb);
	
	// Debug
	// pim.print("Padded");

	std::vector<std::pair<int, int>> curve;
	std::vector<std::pair<int, int>> curveLog;

	int s = pim.width;	// tile size
	for (; s > 1; s /= 2)
	{
		int cnt = 0;
		int n_tiles = pim.width / s;
		for (int r=0; r<n_tiles; r++)
			for (int c = 0; c < n_tiles; c++)
			{
				if (pim.tile_contains_signal(r, c, s))
					cnt++;
			}
		curve.push_back({ s, cnt });
	}

	// Do we have data?
	if (curve.size() == 0)
		return;

	// Post-process
	std::vector <double> X, Y;
	// -- log
	for (int i = 0; i < curve.size(); i++)
	{
		X.push_back (std::log(curve[i].first));
		Y.push_back (std::log(curve[i].second));
	}
	// -- gradient
	double y_0 = Y[1] - Y[0], 
		y_n = Y[curve.size() - 1] - Y[curve.size() - 2];
	for (int i = 1; i < curve.size()-1; i++)
	{
		Y[i] = (Y[i+1] - Y[i-1]) / 2.0;
	}
	Y[0] = y_0;
	Y[curve.size() - 1] = y_n;

	// Debug
	// print_curve(curve, "fd_curve");
}

double FractalDimension::get_box_count_fd() 
{ 
	return box_count_fd; 
}

double FractalDimension::get_perimeter_fd() 
{ 
	return perim_fd; 
}

void FractalDimension::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		// Skip calculation in case of bad data
		if ((int)r.fvals[MIN][0] == (int)r.fvals[MAX][0])
			continue;

		// Calculate feature
		FractalDimension fd (r.raw_pixels, r.aabb);
		r.fvals[FRACT_DIM_BOXCOUNT][0] = fd.get_box_count_fd();
		r.fvals[FRACT_DIM_PERIMETER][0] = fd.get_perimeter_fd();
	}
}

