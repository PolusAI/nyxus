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

	/// @brief Area of a triangulated lattice surface on a grid of sx x sy x sz voxels: each facet's
	/// edges are scaled along x, y and z before its area is taken. A grid that is not cubic scales each
	/// facet's area by a factor of its own orientation, so unlike the volume the area is no multiple of
	/// the lattice one.
	double mesh_area (const std::vector<Triangle3>& mesh, double sx, double sy, double sz);

	/// @brief Volume (mesh) and area (mesh) of an ROI's surface, on a grid of sx x sy x sz voxels
	struct MeshMeasures
	{
		double volume = 0.0, area = 0.0;
	};

	/// @brief Volume (mesh) and area (mesh) of an ROI read plane by plane: mesh_volume() and
	/// mesh_area() over march_roi_surface()'s triangles, accumulated as they are emitted, so neither
	/// the mask nor the mesh is ever held whole. In-memory and out-of-core ROIs share it, and an ROI
	/// gets the same values from both. The volume is taken on the lattice and times a voxel's volume,
	/// sx*sy*sz; the area is taken facet by facet on the grid.
	///
	/// hull_points, when given, receives the points the surface's convex hull is spanned by: of the
	/// mesh vertices on each lattice row along x, the two ends -- a vertex between two others on one
	/// line cannot be a vertex of the hull. They are in doubled lattice coordinates, so the
	/// half-integer vertices become integers, and are ordered by row.
	MeshMeasures roi_mesh_measures (const LatticeBounds& bounds, const RoiPlaneSource& planes,
		double sx, double sy, double sz, std::vector<std::array<double, 3>>* hull_points = nullptr);

	/// @brief roi_mesh_measures() over an in-memory voxel cloud.
	MeshMeasures roi_mesh_measures (const std::vector<Pixel3>& cloud,
		double sx, double sy, double sz, std::vector<std::array<double, 3>>* hull_points = nullptr);
}
