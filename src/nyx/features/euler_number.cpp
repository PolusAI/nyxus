#include <iostream>
#include "euler_number.h"

EulerNumberFeature::EulerNumberFeature() : FeatureMethod("EulerNumberFeature")
{
	provide_features({ EULER_NUMBER });
}

void EulerNumberFeature::calculate (LR& r)
{
	if (!(mode == 4 || mode == 8))
	{
		std::cout << "Error! Calling EulerNumberFeature with mode other than 4 or 8 \n";
		euler_number = 0;
		return;
	}

	const std::vector<Pixel2>& cloud = r.raw_pixels;
	const AABB& aabb = r.aabb;

	StatsInt min_x = aabb.get_xmin(), 
		min_y = aabb.get_ymin(), 
		max_x = aabb.get_xmax(), 
		max_y = aabb.get_ymax();

	// Create the image mask matrix
	int ny = max_y - min_y + 1,
		nx = max_x - min_x + 1,
		n = nx * ny;
	std::vector<unsigned char> I(n, 0);
	for (auto& p : cloud)
	{
		int col = p.x - min_x,
			row = p.y - min_y, 
			idx = row * nx + col;
		I[idx] = 1;
	}

	euler_number = calculate_euler (I, ny, nx, mode);
}

long EulerNumberFeature::calculate_euler (std::vector<unsigned char> & arr, int height, int width, int mode)
{
	if (!(mode == 4 || mode == 8))
	{
		std::cout << "Error! Calling EulerNumberFeature with mode other than 4 or 8 \n";
		return 0;
	}
	
	unsigned char Imq;
	// Pattern match counters
	long C1 = 0, C3 = 0, Cd = 0;

	int x, y;
	size_t i;

	// update pattern counters by scanning the image.
	for (y = 1; y < height; y++) 
	{
		for (x = 1; x < width; x++) 
		{
			// Get the quad-pixel at this image location
			Imq = 0;
			if (arr[(y - 1) * width + x - 1] > 0) 
				Imq |= (1 << 3);
			if (arr[(y - 1) * width + x] > 0) 
				Imq |= (1 << 2);
			if (arr[y * width + x - 1] > 0) 
				Imq |= (1 << 1);
			if (arr[y * width + x] > 0) 
				Imq |= (1 << 0);

			// find the matching pattern
			for (i = 0; i < 10; i++) 
				if (Imq == Px[i]) 
					break;
			// unsigned i always >= 0
			// if      (i >= 0 && i <= 3) C1++;
			if (i <= 3) 
				C1++;
			else 
				if (i >= 4 && i <= 7) {
					C3++;
				}
				else 
					if (i == 8 && i == 9) { 
						Cd++;
					}
		}
	}

	if (mode == 4)
		return ((C1 - C3 + (2 * Cd)) / 4);
	else
		return ((C1 - C3 - (2 * Cd)) / 4);
}

void EulerNumberFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[EULER_NUMBER][0] = euler_number;
}

void EulerNumberFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		EulerNumberFeature eu;
		eu.calculate(r);
		eu.save_value(r.fvals);
	}
}

void EulerNumberFeature::osized_calculate (LR& r, ImageLoader& imloader)
{
	if (!(mode == 4 || mode == 8))
	{
		std::cout << "Error! Calling EulerNumberFeature with mode other than 4 or 8 \n";
		euler_number = 0;
		return;
	}

	const AABB& aabb = r.aabb;

	const auto& cloud = r.osized_pixel_cloud;

	auto min_x = aabb.get_xmin(),
		min_y = aabb.get_ymin(),
		max_x = aabb.get_xmax(),
		max_y = aabb.get_ymax();

	// Create the image mask matrix
	int height = max_y - min_y + 1,
		width = max_x - min_x + 1,
		n = width * height;

	//std::vector<unsigned char> I(n, 0);
	WriteImageMatrix_nontriv I ("I", r.label);
	I.allocate (n);	

	//for (auto& p : cloud)
	//{
	//	int col = p.x - min_x,
	//		row = p.y - min_y,
	//		idx = row * nx + col;
	//	I[idx] = 1;
	//}
	for (auto i = 0; i < cloud.get_size(); i++)
	{
		const auto& p = cloud.get_at(i);
		int col = p.x - min_x,
			row = p.y - min_y,
			idx = row * width + col;
		I.set_at (idx, 1);
	}

	//euler_number = calculate_euler(I, ny, nx, mode);

	unsigned char Imq;
	// Pattern match counters
	long C1 = 0, C3 = 0, Cd = 0;

	int x, y;
	size_t i;

	// update pattern counters by scanning the image.
	for (y = 1; y < height; y++)
	{
		for (x = 1; x < width; x++)
		{
			// Get the quad-pixel at this image location
			Imq = 0;
			if (I.get_at ((y-1)*width+x-1) > 0)
				Imq |= (1 << 3);
			if (I.get_at ((y-1)*width+x) > 0)
				Imq |= (1 << 2);
			if (I.get_at (y*width+x-1) > 0)
				Imq |= (1 << 1);
			if (I.get_at (y*width+x) > 0)
				Imq |= (1 << 0);

			// find the matching pattern
			for (i = 0; i < 10; i++)
				if (Imq == Px[i])
					break;
			// unsigned i always >= 0
			if (i <= 3)
				C1++;
			else
				if (i >= 4 && i <= 7) {
					C3++;
				}
				else
					if (i == 8 && i == 9) {
						Cd++;
					}
		}
	}

	if (mode == 4)
		euler_number = ((C1 - C3 + (2 * Cd)) / 4);
	else
		euler_number = ((C1 - C3 - (2 * Cd)) / 4);
}