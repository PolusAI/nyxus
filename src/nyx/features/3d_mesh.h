#pragma once

#include <vector>
#include "pixel.h"

namespace Nyxus
{
	/// @brief One triangle of an ROI surface mesh. Vertices carry lattice coordinates, so a mesh
	/// built on an anisotropy-resampled cloud is already in physical units.
	struct Triangle3
	{
		double a[3], b[3], c[3];	// layout: x, y, z
	};

	/// @brief Triangulates the ROI's surface with marching cubes at the 0.5 isolevel of the voxel
	/// mask, which is the surface IBSI's volume (mesh) and area (mesh) are integrals of, and the one
	/// MIRP and pyradiomics build. Every vertex lands on the midpoint between an in-ROI voxel centre
	/// and an out-of-ROI one, because the field is binary and the isolevel sits halfway.
	///
	/// The mesh is closed: the case table pairs each cube face's contour by a rule that reads only
	/// that face's four corners, so two cubes sharing a face always cut it the same way and no facet
	/// is left open. mesh_volume() relies on that.
	void build_roi_surface_mesh (std::vector<Triangle3>& mesh, const std::vector<Pixel3>& cloud);

	/// @brief Volume enclosed by a closed triangulated surface, by the divergence theorem.
	double mesh_volume (const std::vector<Triangle3>& mesh);

	/// @brief Total area of a triangulated surface.
	double mesh_area (const std::vector<Triangle3>& mesh);
}
