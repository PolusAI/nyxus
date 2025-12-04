#include <iomanip>	//xxxxxxxxxxxx setw()
#include "texture_feature.h"

//xxxxxxxxx-----------	double TextureFeature::radiomics_bin_width = 25;

void print_simplecube (const SimpleCube<PixIntens>& A, int fieldwidth)
{
	auto w = A.width(),
		h = A.height(),
		d = A.depth();
	std::cout << "WxHxD: " << w << "x" << h << "x" << d << "\n";
	for (auto z = 0; z < d; z++)
	{
		std::cout << "z=[" << z << "] (" << d << ")\n";
		// header of X-labels
		for (int x = 0; x < w; x++)
			std::cout << std::setw(fieldwidth) << x % 10;
		std::cout << "\t<X\n\n";
		for (auto y = 0; y < h; y++)
		{
			// data
			for (auto x = 0; x < w; x++)
			{
				PixIntens a = A.xyz(x, y, z);
				std::cout << std::setw(fieldwidth) << a;
			}
			// Y-label
			std::cout << "\t[" << y << "]\n";
		}
	}
}
