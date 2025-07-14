#pragma once

#pragma once
#include "../featureset.h"
#include "../feature_method.h"

/// @brief A contour is a vector of X and Y coordinates of all the pixels on the border of a ROI. This class uses Moore's algorithm for cnotour detection.
class D3_SurfaceFeature : public FeatureMethod
{
public:

	D3_SurfaceFeature();
	void calculate(LR& r);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate(LR& r, ImageLoader& imloader);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);
	static void parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	void cleanup_instance();
	static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
	static bool required(const FeatureSet& fs);

	const constexpr static std::initializer_list<Nyxus::Feature3D> featureset =
	{
		Nyxus::Feature3D::AREA,
		Nyxus::Feature3D::AREA_2_VOLUME,
		Nyxus::Feature3D::COMPACTNESS1,
		Nyxus::Feature3D::COMPACTNESS2,
		Nyxus::Feature3D::MESH_VOLUME,
		Nyxus::Feature3D::SPHERICAL_DISPROPORTION,
		Nyxus::Feature3D::SPHERICITY,
		Nyxus::Feature3D::VOLUME_CONVEXHULL,
		Nyxus::Feature3D::VOXEL_VOLUME,
		Nyxus::Feature3D::MAJOR_AXIS_LEN,
		Nyxus::Feature3D::MINOR_AXIS_LEN,
		Nyxus::Feature3D::LEAST_AXIS_LEN,
		Nyxus::Feature3D::ELONGATION,
		Nyxus::Feature3D::FLATNESS
	};

private:

	struct Simplex3
	{
		float a[3], b[3], c[3];	// layout: x, y, z
		Simplex3 (const float* a_, const float* b_, const float* c_)
		{
			for (int i = 0; i < 3; i++)
			{
				a[i] = a_[i];
				b[i] = b_[i];
				c[i] = c_[i];
			}
		}
	};	

	std::vector<Simplex3> hull_complex;
	
	void build_surface (LR& r);

	double fval_AREA,
		fval_AREA_2_VOLUME,
		fval_COMPACTNESS1,
		fval_COMPACTNESS2,
		fval_MESH_VOLUME,
		fval_SPHERICAL_DISPROPORTION,
		fval_SPHERICITY,
		fval_VOLUME_CONVEXHULL,
		fval_VOXEL_VOLUME,
		fval_MAJOR_AXIS_LEN,
		fval_MINOR_AXIS_LEN,
		fval_LEAST_AXIS_LEN,
		fval_ELONGATION,
		fval_FLATNESS;

};

