#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <array>
#include "sensemaker.h"


#ifndef __unix
#define NOMINMAX	// Prevent converting std::min(), max(), ... into macros
#include<windows.h>
#endif

EulerNumber::EulerNumber(std::vector<Pixel2>& P, StatsInt min_x, StatsInt  min_y, StatsInt max_x, StatsInt max_y, int mode)
	{
		// Create the image mask matrix
		int ny = max_y - min_y + 1,
			nx = max_x - min_x + 1,
			n = nx * ny;
		std::vector<unsigned char> I(n, 0);
		for (auto& p : P)
		{
			int col = p.x - min_x,
				row = p.y - min_y, 
				idx = row * nx + col;
			I[idx] = 1;
		}

		euler_number = calculate(I, ny, nx, mode);
	}

long EulerNumber::calculate (std::vector<unsigned char> & arr, int height, int width, int mode)
	{
	if ( !(mode == 4 || mode == 8))
		throw std::runtime_error("Calling EulerNumber with mode other than 4 or 8");
	
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



