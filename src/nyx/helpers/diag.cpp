#include <fstream>
#include <iostream>
#include <list>
#include <vector>
#include "../environment.h"
#include "../globals.h"
#include "../roi_cache.h"
#include "../helpers/fsystem.h"

namespace Nyxus
{

void dump_roi_metrics (const int dim, const std::string & output_dir, const size_t ram_limit, const std::string& label_fpath, const Uniqueids & uniqueLabels, const Roidata & roiData)
{
	// are we amidst a 3D scenario ?
	bool dim3 = (dim==3);

	// prepare the file name
	fs::path pseg(label_fpath);
	std::string fpath = output_dir + "/roi_metrics_" + pseg.stem().string() + ".csv";

	// fix the special 3D file name character if needed
	if (dim3)
		for (auto& ch : fpath)
			if (ch == '*')
				ch = '~';

	std::cout << "Dumping ROI metrics to " << fpath << '\n';

	std::ofstream f(fpath);
	if (f.fail())
	{
		std::cerr << "Error: cannot create file " << fpath << '\n';
		return;
	}

	// header
	f << "label, area, minx, miny, maxx, maxy, width, height, min_intens, max_intens, size_bytes, size_class \n";
	// sort labels
	std::vector<int>  sortedLabs{ uniqueLabels.begin(), uniqueLabels.end() };
	std::sort(sortedLabs.begin(), sortedLabs.end());
	// body
	for (auto lab : sortedLabs)
	{
		const LR& r = roiData.at (lab);
		auto szb = r.get_ram_footprint_estimate (uniqueLabels.size());
		std::string ovsz = szb < ram_limit ? "TRIVIAL" : "OVERSIZE";
		f << lab << ", "
			<< r.aux_area << ", "
			<< r.aabb.get_xmin() << ", "
			<< r.aabb.get_ymin() << ", "
			<< r.aabb.get_xmax() << ", "
			<< r.aabb.get_ymax() << ", "
			<< r.aabb.get_width() << ", "
			<< r.aabb.get_height() << ", "
			<< r.aux_min << ", "
			<< r.aux_max << ", "
			<< szb << ", "
			<< ovsz << ", ";
		f << "\n";
	}

	f.flush();
}

void dump_roi_pixels (const int dim, const std::string& output_dir, const std::vector<int>& batch_labels, const std::string& label_fpath, const Uniqueids& uniqueLabels, const Roidata& roiData)
{
	// no data ?
	if (batch_labels.size() == 0)
	{
		std::cerr << "Error: no ROI pixel data for file " << label_fpath << '\n';
		return;
	}

	// sort labels for reader's comfort
	std::vector<int>  srt_L{ batch_labels.begin(), batch_labels.end() };
	std::sort(srt_L.begin(), srt_L.end());

	// are we amidst a 3D scenario ?
	bool dim3 = (dim == 3);

	// prepare the file name
	fs::path pseg(label_fpath);
	std::string fpath = output_dir + "/roi_pixels_" + pseg.stem().string() + "_batch" + std::to_string(srt_L[0]) + '-' + std::to_string(srt_L[srt_L.size() - 1]) + ".csv";

	// fix the special 3D file name character if needed
	if (dim3)
		for (auto& ch : fpath)
			if (ch == '*')
				ch = '~';

	std::cout << "Dumping ROI pixels to " << fpath << '\n';

	std::ofstream f(fpath);
	if (f.fail())
	{
		std::cerr << "Error: cannot create file " << fpath << '\n';
		return;
	}

	// header
	f << "label,x,y,z,intensity, \n";

	// body
	for (auto lab : srt_L)
	{
		const LR& r = roiData.at (lab);
		if (dim3)
			for (auto& plane : r.zplanes)
				for (auto idx : plane.second)
				{
					auto& pxl = r.raw_pixels_3D[idx];
					f << lab << "," << pxl.x << ',' << pxl.y << ',' << pxl.z << ',' << pxl.inten << ',' << '\n';

				}
		else
			for (auto pxl : r.raw_pixels)
				f << lab << "," << pxl.x << ',' << pxl.y << ',' << pxl.inten << ',' << '\n';
	}

	f.flush();
}

void dump_2d_image_1d_layout(const std::vector<PixIntens>& I, const int W, const int H, const std::string& head, const std::string& tail)
{
	std::cout << head;

	// header
	for (int i = 0; i < W; i++)
		std::cout << i % 10;
	std::cout << " =X\n";
	for (int i = 0; i < W; i++)
		if (i % 10 == 0)
			std::cout << '|';
		else
			std::cout << '_';
	std::cout << "\n";

	// body
	for (int y = 0; y < H; y++)
	{
		for (int x = 0; x < W; x++)
		{
			size_t idx = x + y * (W);
			auto inte = I.at(idx);
			if (inte)
				std::cout << 'O'; // value
			else
				std::cout << char(250); // spacer
		}
		std::cout << " " << y << "=y\n";
	}

	std::cout << tail;
}

void dump_2d_image_with_vertex_chain(
	const std::vector<PixIntens>& I,
	const std::vector<Pixel2>& V,
	const int W,
	const int H,
	const std::string& head,
	const std::string& tail)
{
	std::cout << head;

	// header
	for (int i = 0; i < W; i++)
		std::cout << i % 10;
	std::cout << " =X\n";
	for (int i = 0; i < W; i++)
		if (i % 10 == 0)
			std::cout << '|';
		else
			std::cout << '_';
	std::cout << "\n";

	// body
	for (int y = 0; y < H; y++)
	{
		for (int x = 0; x < W; x++)
		{
			size_t idx = x + y * (W);
			auto inte = I.at(idx);
			if (inte)
			{
				std::string valMarker = "O"; // default
				// find a vertex with this (x,y)
				for (size_t vidx = 0; vidx < V.size(); vidx++)
				{
					const auto& v = V[vidx];
					if (v.x == x && v.y == y)
					{
						valMarker = char('a' + vidx%26);
						break;
					}
				}
				std::cout << valMarker;
			}
			else
				std::cout << char(250); // spacer
		}
		std::cout << " " << y << "\n";
	}

	std::cout << tail;
}

void dump_2d_image_with_vertex_set (
	const std::vector<PixIntens>& I,
	const std::list<Pixel2>& V,
	const int W,
	const int H,
	const std::string& head,
	const std::string& tail)
{
	std::cout << head;

	// header
	for (int i = 0; i < W; i++)
		std::cout << i % 10;
	std::cout << " =X\n";
	for (int i = 0; i < W; i++)
		if (i % 10 == 0)
			std::cout << '|';
		else
			std::cout << '_';
	std::cout << "\n";

	// body
	for (int y = 0; y < H; y++)
	{
		for (int x = 0; x < W; x++)
		{
			size_t idx = x + y * (W);
			auto inte = I.at(idx);
			if (inte)
			{
				std::string valMarker = "O"; // default
				// find a vertex with this (x,y)
				size_t vidx = 0;
				for (const auto& v : V)
				{
					if (v.x == x && v.y == y)
					{
						valMarker = char('a' + vidx % 26);
						break;
					}
					vidx++;
				}
				std::cout << valMarker;
			}
			else
				std::cout << char(250); // spacer
		}
		std::cout << " " << y << "\n";
	}

	std::cout << tail;
}

void dump_2d_image_with_halfcontour(
	const std::vector<PixIntens>& I, // border image
	const std::list<Pixel2>& unordered, // unordered contour pixels
	const std::vector<Pixel2>& ordered, // already ordered pixels
	const Pixel2& pxTip, // tip of ordering
	const int W,
	const int H,
	const std::string& head,
	const std::string& tail)
{
	std::cerr << "\nhalf-contour! tip pixel: " << pxTip.x << "," << pxTip.y << "\n";
	std::cerr << "ordered:\n";
	int i = 1;
	for (auto& pxo : ordered)
	{
		std::cerr << "\t" << pxo.x << "," << pxo.y;
		if (i++ % 10 == 0)
			std::cerr << "\n";
	}
	std::cerr << "\n";

	int neigR2 = 400;	// squared
	std::cerr << "unordered around the tip (R^2=" << neigR2 << "):\n";
	i = 1;
	for (auto& pxu : unordered)
	{
		// filter out the far neighborhood
		if (pxTip.sqdist(pxu) > neigR2)
			continue;

		std::cerr << "\t" << pxu.x << "," << pxu.y;
		if (i++ % 10 == 0)
			std::cerr << "\n";
	}
	std::cerr << "\n";

	std::cout << "\n\n\n" << "-- Contour image --\n";
	std::setw(3);
	// header
	std::cout << "\t";	// indent
	for (int i = 0; i < W; i++)
		if (i % 10 == 0)
			std::cout << '|';
		else
			std::cout << '_';
	std::cout << "\n";
	//---
	for (int y = 0; y < H; y++)
	{
		std::cout << "y=" << y << "\t";
		for (int x = 0; x < W; x++)
		{
			size_t idx = x + y * (W);
			auto inte = I.at(idx);

			// Display a pixel symbol depending on its role
			bool in_ordered = false;
			for (auto pxo : ordered)
				if (pxo.x == x && pxo.y == y)
				{
					in_ordered = true;
					break;
				}
			bool in_unordered = false;
			for (auto pxu : unordered)
				if (pxu.x == x && pxu.y == y)
				{
					in_unordered = true;
					break;
				}
			if (pxTip.x == x && pxTip.y == y)
				std::cout << 'T';
			else
				if (in_ordered)
				{
					if (ordered.at(0).x == x && ordered.at(0).y == y)
						std::cout << '1';
					else
						std::cout << 'O';
				}
				else
					if (in_unordered)
						std::cout << 'U';
					else
						if (inte == 0)
							std::cout << '+';
						else
							std::cout << ' ';

		}
		std::cout << "\n";
	}
	std::cout << "\n\n\n";
}

}