#include <fstream>
#include <iostream>
#include "../environment.h"
#include "../globals.h"
#include "../roi_cache.h"
#include "../helpers/fsystem.h"

namespace Nyxus
{

void dump_roi_metrics(const std::string& label_fpath)
{
	// are we amidst a 3D scenario ?
	bool dim3 = theEnvironment.dim();

	// prepare the file name
	fs::path pseg(label_fpath);
	std::string fpath = theEnvironment.output_dir + "/roi_metrics_" + pseg.stem().string() + ".csv";

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
		LR& r = roiData[lab];
		auto szb = r.get_ram_footprint_estimate();
		std::string ovsz = szb < theEnvironment.get_ram_limit() ? "TRIVIAL" : "OVERSIZE";
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

void dump_roi_pixels(const std::vector<int>& batch_labels, const std::string& label_fpath)
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
	bool dim3 = theEnvironment.dim();

	// prepare the file name
	fs::path pseg(label_fpath);
	std::string fpath = theEnvironment.output_dir + "/roi_pixels_" + pseg.stem().string() + "_batch" + std::to_string(srt_L[0]) + '-' + std::to_string(srt_L[srt_L.size() - 1]) + ".csv";

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
		LR& r = roiData[lab];
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

}