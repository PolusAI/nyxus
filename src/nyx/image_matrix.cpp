#include <iostream>
#include <iomanip>
#include "image_matrix.h"

void SimpleMatrix<int>::print(const std::string& head, const std::string& tail)
{
	const int Wd = 6;	// data
	const int Wi = 5;	// index

	std::cout << head << "\n";
	std::cout << std::string(Wi + Wd * this->width(), '-') << std::endl;	// Upper solid line
	std::cout << "w=" << this->width() << " h=" << this->height() << "\n";

	for (int row = 0; row < this->height(); row++)
	{
		// Hdr
		if (row == 0)
		{
			std::cout << std::setw(Wi + 2) << "";	// Wi+2 because '[' + Wi + ']'
			for (int col = 0; col < this->width(); col++)
			{
				std::cout << std::setw(Wd) << col;
			}
			std::cout << "\n";
		}

		// Row
		std::cout << "[" << std::setw(Wi) << row << "]";
		for (int col = 0; col < this->width(); col++)
		{
			std::cout << std::setw(Wd) << (int) this->operator()(row, col);
		}
		std::cout << "\n";
	}

	std::cout << std::string(Wi + Wd * this->width(), '-') << std::endl;	// Lower solid line
	std::cout << tail;
}

void ImageMatrix::print (const std::string& head, const std::string& tail)
{
	const int Wd = 6;	// data
	const int Wi = 5;	// index

	readOnlyPixels D = ReadablePixels();

	std::cout << head << "\n";
	std::cout << std::string(Wi + Wd * this->width, '-') << std::endl;	// Upper solid line
	std::cout << "w=" << this->width << " h=" << this->height << "\n";

	for (int row = 0; row < this->height; row++)
	{
		// Hdr
		if (row == 0)
		{
			std::cout << std::setw(Wi + 2) << "";	// Wi+2 because '[' + Wi + ']'
			for (int col = 0; col < this->width; col++)
			{
				std::cout << std::setw(Wd) << col;
			}
			std::cout << "\n";
		}

		// Row
		std::cout << "[" << std::setw(Wi) << row << "]";
		for (int col = 0; col < this->width; col++)
		{
			std::cout << std::setw(Wd) << (int)D(row,col);
		}
		std::cout << "\n";
	}

	std::cout << std::string(Wi + Wd * this->width, '-') << std::endl;	// Lower solid line
	std::cout << tail;
}

