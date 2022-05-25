#include "fractal_dim.h"
#include "image_matrix.h"

FractalDimensionFeature::FractalDimensionFeature() : FeatureMethod("FractalDimensionFeature")
{
	provide_features({ FRACT_DIM_BOXCOUNT, FRACT_DIM_PERIMETER });
}

void FractalDimensionFeature::calculate(LR& r)
{
	Power2PaddedImageMatrix pim(r.raw_pixels, r.aabb);

	// Debug
	// pim.print("Padded");

	std::vector<std::pair<int, int>> curve;
	std::vector<std::pair<int, int>> curveLog;

	int s = pim.width;	// tile size
	for (; s > 1; s /= 2)
	{
		int cnt = 0;
		int n_tiles = pim.width / s;
		for (int r = 0; r < n_tiles; r++)
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
		X.push_back(std::log(curve[i].first));
		Y.push_back(std::log(curve[i].second));
	}

	// Skip tiny ROIs
	if (Y.size() < 2)
		return;

	// -- gradient
	double y_0 = Y[1] - Y[0],
		y_n = Y[curve.size() - 1] - Y[curve.size() - 2];
	for (int i = 1; i < curve.size() - 1; i++)
	{
		Y[i] = (Y[i + 1] - Y[i - 1]) / 2.0;
	}
	Y[0] = y_0;
	Y[curve.size() - 1] = y_n;

	// Debug
	// print_curve(curve, "fd_curve");
}

void FractalDimensionFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{}

void FractalDimensionFeature::osized_calculate(LR& r, ImageLoader& imloader)
{}

void FractalDimensionFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[FRACT_DIM_BOXCOUNT][0] = box_count_fd;
	fvals[FRACT_DIM_PERIMETER][0] = perim_fd;
}

void FractalDimensionFeature::parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		// Calculate feature
		FractalDimensionFeature fd;
		fd.calculate(r);
		fd.save_value(r.fvals);
	}
}

