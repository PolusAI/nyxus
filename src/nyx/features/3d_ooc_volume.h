#pragma once

// The streamed ROI as the out-of-core 3D texture passes see it: its bounding box, and the dense
// grey-binned planes read back from the disk-backed voxel cloud. Each family bins its own way and
// spans its own number of Z-planes, but all of them need the same three things -- the bbox
// geometry, a plane of the binned cube, and the set of grey levels the ROI's voxels carry -- so
// those live here once instead of in each family's osized_calculate().

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <string>
#include "pixel.h"
#include "../roi_cache.h"

namespace Nyxus
{
	// 'Bin' is the caller's binning functor, a template parameter rather than a std::function so
	// the per-voxel call inlines -- this class exists for the ROIs where that call happens a
	// billion times. Class template argument deduction names it at the construction site.
	template <class Bin>
	class OocBinnedVolume
	{
	public:
		/// @param bin turns one raw voxel value into the grey level the in-core cube holds
		/// @param background the level a cell outside the ROI carries in that cube
		/// @param window_planes how many planes the sliding window keeps (the family's Z reach)
		/// @param with_mask also keep, per plane, which cells are the ROI's -- a cell holding the
		///        background level is otherwise indistinguishable from a ROI voxel that binned to it
		OocBinnedVolume (LR& r, Bin bin, PixIntens background,
			int window_planes, bool with_mask = false)
			: roi_(r), bin_(std::move(bin)), bg_(background), with_mask_(with_mask),
			W_((int) r.aabb.get_width()), H_((int) r.aabb.get_height()), D_((int) r.aabb.get_z_depth()),
			xmin_(r.aabb.get_xmin()), ymin_(r.aabb.get_ymin()), zmin_(r.aabb.get_zmin()),
			n_voxels_(r.raw_voxels_NT.size())
		{
			int slots = (std::max) (1, window_planes);
			planes_.resize (slots);
			plane_z_.assign (slots, -1);
			if (with_mask_)
				masks_.resize (slots);
		}

		int width() const { return W_; }
		int height() const { return H_; }
		int depth() const { return D_; }
		StatsInt xmin() const { return xmin_; }
		StatsInt ymin() const { return ymin_; }
		StatsInt zmin() const { return zmin_; }
		std::size_t n_voxels() const { return n_voxels_; }
		PixIntens background() const { return bg_; }

		/// @brief True when the bounding box holds cells that are not the ROI's -- the ones the
		/// in-core cube fills with the background level.
		bool has_background() const { return n_voxels_ < (std::size_t) W_ * H_ * D_; }

		/// @brief One pass over the cloud: every distinct grey level its voxels carry, and the
		/// largest one seen.
		/// @param with_background also count the background level, when the bbox has background
		/// @param drop_zero leave level 0 out of the set (the families that read 0 as "no level")
		std::set<PixIntens> levels (bool with_background, bool drop_zero, PixIntens& max_level) const
		{
			std::set<PixIntens> U;
			max_level = 0;
			if (with_background && has_background() && ! (drop_zero && bg_ == 0))
			{
				U.insert (bg_);
				max_level = bg_;
			}
			for (std::size_t i = 0; i < n_voxels_; i++)
			{
				PixIntens b = bin_ (roi_.raw_voxels_NT[i].inten);
				if (b > max_level)
					max_level = b;
				if (! (drop_zero && b == 0))
					U.insert (b);
			}
			return U;
		}

		/// @brief The largest grey level the ROI's voxels carry (the families that need no set).
		PixIntens max_level() const
		{
			PixIntens mx = 0;
			for (std::size_t i = 0; i < n_voxels_; i++)
			{
				PixIntens b = bin_ (roi_.raw_voxels_NT[i].inten);
				if (b > mx)
					mx = b;
			}
			return mx;
		}

		/// @brief Plane 'lz' of the binned cube (W*H, background-filled), loaded on demand and
		/// kept until the window wraps. Planes are addressed in the ROI's own local coordinates.
		const std::vector<PixIntens>& plane (int lz)
		{
			load (lz);
			return planes_[slot (lz)];
		}

		/// @brief Which cells of plane 'lz' are the ROI's (1) and which are background (0).
		/// The volume must have been built with_mask.
		const std::vector<unsigned char>& mask (int lz)
		{
			if (! with_mask_)
				throw std::logic_error ("OocBinnedVolume::mask(): this volume was built without the ROI mask");
			load (lz);
			return masks_[slot (lz)];
		}

	private:
		int slot (int lz) const { return lz % (int) planes_.size(); }

		void load (int lz)
		{
			if (lz < 0 || lz >= D_)
				throw std::out_of_range ("OocBinnedVolume: plane " + std::to_string (lz) + " is outside the ROI's " + std::to_string (D_) + " planes");

			int sl = slot (lz);
			if (plane_z_[sl] == lz)
				return;

			std::vector<PixIntens>& pl = planes_[sl];
			pl.assign ((std::size_t) W_ * H_, bg_);
			if (with_mask_)
				masks_[sl].assign ((std::size_t) W_ * H_, 0);

			roi_.raw_voxels_NT.read_slab ((std::size_t) (zmin_ + lz), slab_);
			for (const auto& v : slab_)
			{
				int lx = (int) v.x - (int) xmin_,
					ly = (int) v.y - (int) ymin_;
				if (lx >= 0 && lx < W_ && ly >= 0 && ly < H_)
				{
					pl[(std::size_t) ly * W_ + lx] = bin_ (v.inten);
					if (with_mask_)
						masks_[sl][(std::size_t) ly * W_ + lx] = 1;
				}
			}
			plane_z_[sl] = lz;
		}

		LR& roi_;
		Bin bin_;
		PixIntens bg_;
		bool with_mask_;
		int W_, H_, D_;
		StatsInt xmin_, ymin_, zmin_;
		std::size_t n_voxels_;

		std::vector<std::vector<PixIntens>> planes_;
		std::vector<std::vector<unsigned char>> masks_;
		std::vector<int> plane_z_;
		std::vector<Pixel3> slab_;
	};

	/// @brief Grey level -> its row in 'I', for every level up to 'max_level': the binary search
	/// the per-voxel hot loops would otherwise do, precomputed.
	inline std::vector<int> ooc_row_lut (const std::vector<PixIntens>& I, PixIntens max_level)
	{
		std::vector<int> lut ((std::size_t) max_level + 1, 0);
		for (PixIntens v = 0; v <= max_level; v++)
			lut[v] = (int) (std::lower_bound (I.begin(), I.end(), v) - I.begin());
		return lut;
	}
}
