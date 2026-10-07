#pragma once

#include <array>
#include <functional>
#include <vector>
#include "pixel.h"

namespace Nyxus
{
	/// @brief One triangle of an ROI surface mesh. Vertices carry lattice coordinates.
	struct Triangle3
	{
		double a[3], b[3], c[3];	// layout: x, y, z
	};

	/// @brief Inclusive lattice bounds that contain every voxel of an ROI. Bounds wider than the ROI
	/// are harmless: the cells outside it are empty and produce no facet.
	struct LatticeBounds
	{
		StatsInt xmin, xmax, ymin, ymax, zmin, zmax;
	};

	/// @brief Fills 'voxels' with the ROI's voxels in lattice plane z, and leaves it empty for a plane
	/// the ROI does not reach. Only each voxel's x and y are read.
	using RoiPlaneSource = std::function<void (StatsInt z, std::vector<Pixel3>& voxels)>;

	/// @brief Triangulates the ROI's surface with marching cubes at the 0.5 isolevel of the voxel
	/// mask, which is the surface IBSI's volume (mesh) is the integral of. MIRP and pyradiomics build
	/// the same kind of surface, with the same vertices; their triangles can differ from these where a
	/// cube face is ambiguous. Every vertex lands on the midpoint between an in-ROI voxel centre and an
	/// out-of-ROI one, because the field is binary and the isolevel sits halfway.
	///
	/// The cells are walked one z-layer at a time against two voxel planes, which 'planes' supplies,
	/// so the working set is two slices of the bounds rather than the ROI. 'emit' receives each
	/// triangle in a fixed order (z-layer, then y, then x), whatever the bounds.
	///
	/// The surface is closed: the case table pairs each cube face's contour by a rule that reads
	/// only that face's four corners, so two cubes sharing a face always cut it the same way and no
	/// facet is left open. The volume integral relies on that.
	void march_roi_surface (const LatticeBounds& bounds, const RoiPlaneSource& planes,
		const std::function<void (const Triangle3&)>& emit);

	/// @brief march_roi_surface() over an in-memory voxel cloud, collecting the triangles.
	void build_roi_surface_mesh (std::vector<Triangle3>& mesh, const std::vector<Pixel3>& cloud);

	/// @brief Volume enclosed by a closed triangulated surface, by the divergence theorem. Each facet's
	/// term is taken relative to a vertex of the first facet, so on lattice meshes every term is exact
	/// and the result does not change when the mesh is translated.
	double mesh_volume (const std::vector<Triangle3>& mesh);

	/// @brief Volume (mesh) of an ROI read plane by plane: the same integral as mesh_volume() over
	/// march_roi_surface()'s triangles, accumulated as they are emitted, so neither the mask nor the
	/// mesh is ever held whole. In-memory and out-of-core ROIs share it, and an ROI gets the same
	/// value from both.
	///
	/// hull_points, when given, receives the points the surface's convex hull is spanned by: of the
	/// mesh vertices on each lattice row along x, the two ends -- a vertex between two others on one
	/// line cannot be a vertex of the hull. They are in doubled coordinates, so the half-integer
	/// vertices become integers, and are ordered by row.
	double roi_mesh_volume (const LatticeBounds& bounds, const RoiPlaneSource& planes,
		std::vector<std::array<double, 3>>* hull_points = nullptr);

	/// @brief roi_mesh_volume() over an in-memory voxel cloud.
	double roi_mesh_volume (const std::vector<Pixel3>& cloud,
		std::vector<std::array<double, 3>>* hull_points = nullptr);
}
