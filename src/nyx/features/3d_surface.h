#pragma once

#pragma once
#include <array>
#include <vector>
#include "../dataset.h"
#include "../featureset.h"
#include "../feature_method.h"
#include "../feature_settings.h"

/// @brief 3D shape features of an ROI: its surface (exposed voxel faces and the marching-cubes mesh), the convex hull of that mesh, the voxel-counting volume and the principal axes.
class D3_SurfaceFeature : public FeatureMethod
{
public:

	D3_SurfaceFeature();
	void calculate (LR& r, const Fsettings& s);
	void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
	void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);
	void save_value(std::vector<std::vector<double>>& feature_vals);
	void cleanup_instance();
	static void reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & ds);
	static void extract (LR& r, const Fsettings& s);
	static bool required(const FeatureSet& fs);

	// Volume of the convex hull of an integer lattice point cloud; 0 when the points span no volume.
	// Shared by the in-core and out-of-core paths.
	static double convex_hull_volume (const std::vector<std::array<double, 3>>& P);

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

	// Every feature of an ROI that fills its whole w x h x d box (single-ROI mode), in closed form, a
	// voxel being sx by sy by sz
	void set_whole_box (StatsInt w, StatsInt h, StatsInt d, double sx, double sy, double sz);

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

