#include <iostream>
#include "geodetic_len_thickness.h"

using namespace Nyxus;

bool GeodeticLengthThicknessFeature::required(const FeatureSet& fs) 
{ 
	return fs.anyEnabled({ Feature2D::GEODETIC_LENGTH, Feature2D::THICKNESS });
}

GeodeticLengthThicknessFeature::GeodeticLengthThicknessFeature() : FeatureMethod("GeodeticLengthThicknessFeature")
{
	provide_features ({ Feature2D::GEODETIC_LENGTH, Feature2D::THICKNESS });
	add_dependencies ({ Feature2D::AREA_PIXELS_COUNT, Feature2D::PERIMETER });
}

void GeodeticLengthThicknessFeature::calculate (LR& r)
{
	size_t roiArea = r.aux_area,
		roiPerimeter = r.fvals[(int)Feature2D::PERIMETER][0];

	double SqRootTmp = roiPerimeter * roiPerimeter / 16 - (double)roiArea;

	// Make sure value under SqRootTmp is always positive
	if (SqRootTmp < 0) 
		SqRootTmp = 0;

	// Calculate geodetic_length with pq-formula (see above):
	geodetic_length = roiPerimeter / 4 + sqrt(SqRootTmp);

	// Calculate thickness by rewriting Equation (2):
	thickness = roiPerimeter / 2 - geodetic_length;
}

void GeodeticLengthThicknessFeature::osized_calculate (LR& r, ImageLoader&)
{
	calculate(r);	// This feature is not critical to ROI size
}

void GeodeticLengthThicknessFeature::osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {}

void GeodeticLengthThicknessFeature::save_value (std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::GEODETIC_LENGTH][0] = geodetic_length;
	fvals[(int)Feature2D::THICKNESS][0] = thickness;
}

void GeodeticLengthThicknessFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		GeodeticLengthThicknessFeature glt;
		glt.calculate(r);
		glt.save_value(r.fvals);
	}
}

