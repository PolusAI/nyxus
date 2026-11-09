#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <array>
#include <deque>
#include <iomanip>
#include <stdexcept>
#include <string>
#include "moments.h"
#include "contour.h"
#include "../globals.h"
#include "../roi_cache.h"	// Required by the reduction function
#include "../environment.h"		// regular or whole slide mode
#include "image_matrix_nontriv.h"

using namespace Nyxus;

bool ContourFeature::required (const FeatureSet & fset)
{
	return fset.anyEnabled(
		{
		// own features
		Feature2D::PERIMETER,
		Feature2D::DIAMETER_EQUAL_PERIMETER,
		Feature2D::EDGE_INTEGRATED_INTENSITY,
		Feature2D::EDGE_MAX_INTENSITY,
		Feature2D::EDGE_MIN_INTENSITY,
		Feature2D::EDGE_MEAN_INTENSITY,
		Feature2D::EDGE_STDDEV_INTENSITY,
		// dependencies:
		Feature2D::CONVEX_HULL_AREA,
		Feature2D::SOLIDITY,
		Feature2D::CIRCULARITY,
		// the circles, measured to the contour
		Feature2D::DIAMETER_MIN_ENCLOSING_CIRCLE,
		Feature2D::DIAMETER_INSCRIBING_CIRCLE,
		Feature2D::DIAMETER_CIRCUMSCRIBING_CIRCLE,
		// weighted spatial moments
		Feature2D::WEIGHTED_SPAT_MOMENT_00,
		Feature2D::WEIGHTED_SPAT_MOMENT_01,
		Feature2D::WEIGHTED_SPAT_MOMENT_02,
		Feature2D::WEIGHTED_SPAT_MOMENT_03,
		Feature2D::WEIGHTED_SPAT_MOMENT_10,
		Feature2D::WEIGHTED_SPAT_MOMENT_11,
		Feature2D::WEIGHTED_SPAT_MOMENT_12,
		Feature2D::WEIGHTED_SPAT_MOMENT_20,
		Feature2D::WEIGHTED_SPAT_MOMENT_21,
		Feature2D::WEIGHTED_SPAT_MOMENT_30,
		// weighted central moments
		Feature2D::WEIGHTED_CENTRAL_MOMENT_02,
		Feature2D::WEIGHTED_CENTRAL_MOMENT_03,
		Feature2D::WEIGHTED_CENTRAL_MOMENT_11,
		Feature2D::WEIGHTED_CENTRAL_MOMENT_12,
		Feature2D::WEIGHTED_CENTRAL_MOMENT_20,
		Feature2D::WEIGHTED_CENTRAL_MOMENT_21,
		Feature2D::WEIGHTED_CENTRAL_MOMENT_30,
		// weighted normalized central moments
		Feature2D::WT_NORM_CTR_MOM_02,
		Feature2D::WT_NORM_CTR_MOM_03,
		Feature2D::WT_NORM_CTR_MOM_11,
		Feature2D::WT_NORM_CTR_MOM_12,
		Feature2D::WT_NORM_CTR_MOM_20,
		Feature2D::WT_NORM_CTR_MOM_21,
		Feature2D::WT_NORM_CTR_MOM_30,
		// weighted Hu's moments 1-7 
		Feature2D::WEIGHTED_HU_M1,
		Feature2D::WEIGHTED_HU_M2,
		Feature2D::WEIGHTED_HU_M3,
		Feature2D::WEIGHTED_HU_M4,
		Feature2D::WEIGHTED_HU_M5,
		Feature2D::WEIGHTED_HU_M6,
		Feature2D::WEIGHTED_HU_M7,

		// -- intensity raw moments
		Nyxus::Feature2D::IMOM_RM_00,
		Nyxus::Feature2D::IMOM_RM_01,
		Nyxus::Feature2D::IMOM_RM_02,
		Nyxus::Feature2D::IMOM_RM_03,
		Nyxus::Feature2D::IMOM_RM_10,
		Nyxus::Feature2D::IMOM_RM_11,
		Nyxus::Feature2D::IMOM_RM_12,
		Nyxus::Feature2D::IMOM_RM_13,
		Nyxus::Feature2D::IMOM_RM_20,
		Nyxus::Feature2D::IMOM_RM_21,
		Nyxus::Feature2D::IMOM_RM_22,
		Nyxus::Feature2D::IMOM_RM_23,
		Nyxus::Feature2D::IMOM_RM_30,
		// -- intensity central moments
		Nyxus::Feature2D::IMOM_CM_00,
		Nyxus::Feature2D::IMOM_CM_01,
		Nyxus::Feature2D::IMOM_CM_02,
		Nyxus::Feature2D::IMOM_CM_03,
		Nyxus::Feature2D::IMOM_CM_10,
		Nyxus::Feature2D::IMOM_CM_11,
		Nyxus::Feature2D::IMOM_CM_12,
		Nyxus::Feature2D::IMOM_CM_13,
		Nyxus::Feature2D::IMOM_CM_20,
		Nyxus::Feature2D::IMOM_CM_21,
		Nyxus::Feature2D::IMOM_CM_22,
		Nyxus::Feature2D::IMOM_CM_23,
		Nyxus::Feature2D::IMOM_CM_30,
		Nyxus::Feature2D::IMOM_CM_31,
		Nyxus::Feature2D::IMOM_CM_32,
		Nyxus::Feature2D::IMOM_CM_33,
		// -- intensity normalized raw moments
		Nyxus::Feature2D::IMOM_NRM_00,
		Nyxus::Feature2D::IMOM_NRM_01,
		Nyxus::Feature2D::IMOM_NRM_02,
		Nyxus::Feature2D::IMOM_NRM_03,
		Nyxus::Feature2D::IMOM_NRM_10,
		Nyxus::Feature2D::IMOM_NRM_11,
		Nyxus::Feature2D::IMOM_NRM_12,
		Nyxus::Feature2D::IMOM_NRM_13,
		Nyxus::Feature2D::IMOM_NRM_20,
		Nyxus::Feature2D::IMOM_NRM_21,
		Nyxus::Feature2D::IMOM_NRM_22,
		Nyxus::Feature2D::IMOM_NRM_23,
		Nyxus::Feature2D::IMOM_NRM_30,
		Nyxus::Feature2D::IMOM_NRM_31,
		Nyxus::Feature2D::IMOM_NRM_32,
		Nyxus::Feature2D::IMOM_NRM_33,
		// -- intensity normalized central moments
		Nyxus::Feature2D::IMOM_NCM_02,
		Nyxus::Feature2D::IMOM_NCM_03,
		Nyxus::Feature2D::IMOM_NCM_11,
		Nyxus::Feature2D::IMOM_NCM_12,
		Nyxus::Feature2D::IMOM_NCM_20,
		Nyxus::Feature2D::IMOM_NCM_21,
		Nyxus::Feature2D::IMOM_NCM_30,
		// -- intensity Hu's moments 1-7 
		Nyxus::Feature2D::IMOM_HU1,
		Nyxus::Feature2D::IMOM_HU2,
		Nyxus::Feature2D::IMOM_HU3,
		Nyxus::Feature2D::IMOM_HU4,
		Nyxus::Feature2D::IMOM_HU5,
		Nyxus::Feature2D::IMOM_HU6,
		Nyxus::Feature2D::IMOM_HU7,
		// -- intensity weighted raw moments
		Nyxus::Feature2D::IMOM_WRM_00,
		Nyxus::Feature2D::IMOM_WRM_01,
		Nyxus::Feature2D::IMOM_WRM_02,
		Nyxus::Feature2D::IMOM_WRM_03,
		Nyxus::Feature2D::IMOM_WRM_10,
		Nyxus::Feature2D::IMOM_WRM_11,
		Nyxus::Feature2D::IMOM_WRM_12,
		Nyxus::Feature2D::IMOM_WRM_20,
		Nyxus::Feature2D::IMOM_WRM_21,
		Nyxus::Feature2D::IMOM_WRM_30,
		// -- intensity weighted central moments
		Nyxus::Feature2D::IMOM_WCM_02,
		Nyxus::Feature2D::IMOM_WCM_03,
		Nyxus::Feature2D::IMOM_WCM_11,
		Nyxus::Feature2D::IMOM_WCM_12,
		Nyxus::Feature2D::IMOM_WCM_20,
		Nyxus::Feature2D::IMOM_WCM_21,
		Nyxus::Feature2D::IMOM_WCM_30,
		// -- intensity weighted normalized central moments
		Nyxus::Feature2D::IMOM_WNCM_02,
		Nyxus::Feature2D::IMOM_WNCM_03,
		Nyxus::Feature2D::IMOM_WNCM_11,
		Nyxus::Feature2D::IMOM_WNCM_12,
		Nyxus::Feature2D::IMOM_WNCM_20,
		Nyxus::Feature2D::IMOM_WNCM_21,
		Nyxus::Feature2D::IMOM_WNCM_30,
		// -- intensity weighted Hu's moments 1-7 
		Nyxus::Feature2D::IMOM_WHU1,
		Nyxus::Feature2D::IMOM_WHU2,
		Nyxus::Feature2D::IMOM_WHU3,
		Nyxus::Feature2D::IMOM_WHU4,
		Nyxus::Feature2D::IMOM_WHU5,
		Nyxus::Feature2D::IMOM_WHU6,
		Nyxus::Feature2D::IMOM_WHU7,

		// misc
		Feature2D::ROI_RADIUS_MEAN, 
		Feature2D::ROI_RADIUS_MAX, 
		Feature2D::ROI_RADIUS_MEDIAN,
		Feature2D::FRAC_AT_D, 
		Feature2D::MEAN_FRAC, 
		Feature2D::RADIAL_CV
		});
}

ContourFeature::ContourFeature() : FeatureMethod("ContourFeature")
{
	provide_features (ContourFeature::featureset);
}

namespace
{
	// The 8 neighbour directions, clockwise on screen (y pointing down) starting East.
	const int DIR_DX[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	const int DIR_DY[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };

	int direction_of (int dx, int dy)
	{
		static const int lookup[3][3] =
		{
			{ 5, 6, 7 },	// dy = -1: NW, N, NE
			{ 4, -1, 0 },	// dy =  0: W, -, E
			{ 3, 2, 1 }		// dy = +1: SW, S, SE
		};
		return lookup[dy + 1][dx + 1];
	}

	// The padded image of a ROI and its border labels, held in RAM. The padded image holds each ROI
	// pixel's intensity + 1 and 0 elsewhere; a label is 0 for background, 1 for a ROI pixel no border
	// has reached, and +-k for a pixel of border k.
	class RamGrid
	{
	public:
		RamGrid (const std::vector<PixIntens>& P, int W, int H) : P_(P), W_(W), f_(P.size(), 0)
		{
			for (int y = 1; y < H - 1; y++)
				for (int x = 1; x < W - 1; x++)
					f_[x + y * W] = P[x + y * W] != 0 ? 1 : 0;
		}
		int label (int x, int y) const { return f_[x + y * W_]; }
		void set_label (int x, int y, int v) { f_[x + y * W_] = v; }
		PixIntens padded (int x, int y) const { return P_[x + y * W_]; }
		// the labels of row y, current as of the call
		const int* row (int y) { return f_.data() + (size_t) y * W_; }

	private:
		const std::vector<PixIntens>& P_;
		int W_;
		std::vector<int> f_;
	};

	// The same two images held on disk, for a ROI too large to hold them in RAM. The caller fills the
	// padded image; the label image starts at 0 and holds only the labels a walk writes (+-k, k >= 2),
	// a ROI pixel no border has reached reading as 1 off the padded image. Rows are read whole into a
	// cache of at most ROW_CACHE_BYTES and evicted oldest first, as a walk visits a few neighbouring
	// rows at a time; a label is written to the disk and to its cached row alike.
	class DiskGrid
	{
	public:
		DiskGrid (WriteImageMatrix_nontriv& P, WriteImageMatrix_nontriv& f, int W) : P_(P), f_(f), W_(W),
			capacity_ (std::max ((size_t) 3, ROW_CACHE_BYTES / ((size_t) W * (sizeof(int) + sizeof(PixIntens)))))
		{}
		int label (int x, int y) { return load (y).label[x]; }
		void set_label (int x, int y, int v)
		{
			f_.set_at (x + (size_t) y * W_, v);
			load (y).label[x] = v;
		}
		PixIntens padded (int x, int y) { return load (y).padded[x]; }
		// the labels of row y, valid until the next call that reads another row
		const int* row (int y) { return load (y).label.data(); }

	private:
		static constexpr size_t ROW_CACHE_BYTES = 8 << 20;

		struct Row
		{
			std::vector<int> label;
			std::vector<PixIntens> padded;
		};

		Row& load (int y)
		{
			auto it = rows_.find (y);
			if (it != rows_.end())
				return it->second;

			if (rows_.size() >= capacity_)
			{
				rows_.erase (order_.front());
				order_.pop_front();
			}
			Row& r = rows_[y];
			order_.push_back (y);

			P_.get_row (y, buf_);
			r.padded.assign (buf_.begin(), buf_.end());
			f_.get_row (y, buf_);
			r.label.resize (W_);
			for (int x = 0; x < W_; x++)
			{
				int written = (int) buf_[x];
				r.label[x] = written != 0 ? written : (r.padded[x] != 0 ? 1 : 0);
			}
			return r;
		}

		WriteImageMatrix_nontriv& P_;
		WriteImageMatrix_nontriv& f_;
		int W_;
		size_t capacity_;
		std::unordered_map<int, Row> rows_;
		std::deque<int> order_;
		std::vector<double> buf_;
	};
	// Follows one border of the grid's labels from its start pixel (x0, y0), whose background
	// neighbour lies in direction 'from', and returns the border as a closed walk of 8-connected
	// steps beginning at (x0, y0), an outer border clockwise on screen and a hole border
	// counterclockwise. Suzuki & Abe (1985), steps 3.1-3.5: every pixel of the border is labelled
	// 'nbd', or '-nbd' where the walk examined its background East neighbour, so the raster scan
	// neither restarts this border nor misses the next one.
	template <class Grid>
	std::vector<Pixel2> follow_border (Grid& g, int x0, int y0, int from, int nbd)
	{
		auto pixel = [&](int x, int y) { return Pixel2 (x, y, g.padded (x, y) - 1); };	// undecorate the intensity

		std::vector<Pixel2> walk { pixel (x0, y0) };

		// 3.1: the first ROI neighbour clockwise from the background neighbour
		int d1 = -1;
		for (int k = 0; k < 8 && d1 < 0; k++)
		{
			int d = (from + k) % 8;
			if (g.label (x0 + DIR_DX[d], y0 + DIR_DY[d]) != 0)
				d1 = d;
		}
		if (d1 < 0)
		{
			// an isolated pixel is a border of its own
			g.set_label (x0, y0, -nbd);
			return walk;
		}

		const int x1 = x0 + DIR_DX[d1], y1 = y0 + DIR_DY[d1];
		int x2 = x1, y2 = y1,	// the walk's previous pixel
			x3 = x0, y3 = y0;	// the walk's current pixel
		while (true)
		{
			// 3.3: the first ROI neighbour of the current pixel counterclockwise after the previous one
			const int dprev = direction_of (x2 - x3, y2 - y3);
			bool eastIsBackground = false;
			int d4 = dprev;
			for (int k = 1; k <= 8; k++)
			{
				int d = (dprev - k + 16) % 8;
				if (g.label (x3 + DIR_DX[d], y3 + DIR_DY[d]) != 0)
				{
					d4 = d;
					break;
				}
				if (d == 0)
					eastIsBackground = true;
			}
			const int x4 = x3 + DIR_DX[d4], y4 = y3 + DIR_DY[d4];

			// 3.4: label the current pixel
			if (eastIsBackground)
				g.set_label (x3, y3, -nbd);
			else
				if (g.label (x3, y3) == 1)
					g.set_label (x3, y3, nbd);

			// 3.5: back at the start, about to repeat the first step. The walk ran counterclockwise on
			// screen around an outer border; it is returned in the opposite direction, from the same
			// start pixel, so an outer border runs clockwise on screen. The approximate distance
			// searches (Pixel2::min_sqdist and co.) depend on that order.
			if (x4 == x0 && y4 == y0 && x3 == x1 && y3 == y1)
			{
				std::reverse (walk.begin() + 1, walk.end());
				return walk;
			}

			x2 = x3;
			y2 = y3;
			x3 = x4;
			y3 = y4;
			walk.push_back (pixel (x3, y3));
		}
	}

	// Every border of the ROI held in 'g', a padded image of w + 2 by h + 2 pixels, in the raster
	// order of their start pixels. See gather_multicontour.
	template <class Grid>
	void trace_borders (Grid& g, int w, int h, std::vector<std::vector<Pixel2>>& borders, std::vector<bool>* is_hole)
	{
		borders.clear();
		if (is_hole)
			is_hole->clear();

		int nbd = 1;	// border labels start at 2
		for (int y = 1; y <= h; y++)
		{
			const int* f = g.row (y);
			for (int x = 1; x <= w; x++)
			{
				int from;
				if (f[x] == 1 && f[x - 1] == 0)
					from = 4;	// an outer border starts here: its background neighbour is West
				else
					if (f[x] >= 1 && f[x + 1] == 0)
						from = 0;	// a hole border starts here: its background neighbour is East
					else
						continue;

				nbd++;
				borders.push_back (follow_border (g, x, y, from, nbd));
				if (is_hole)
					is_hole->push_back (from == 0);

				// the walk may have read other rows since, and labelled pixels of this one
				f = g.row (y);
			}
		}
	}
}

/*static*/ void ContourFeature::gather_multicontour (
	// out
	std::vector<std::vector<Pixel2>>& multicountour,
	// in
	const std::vector<PixIntens>& P,	// padded image
	int w,
	int h,
	int verbose,
	// out, optional
	std::vector<bool>* is_hole)
{
	// Every border of the ROI -- the outer border of each 8-connected fragment and the border of each
	// hole -- as a closed walk of 8-connected steps over the ROI pixels with a 4-neighbour outside
	// the ROI. Raster-scan border following of Suzuki & Abe (1985), "Topological structural analysis
	// of digitized binary images by border following", the algorithm behind OpenCV's findContours.
	// Coordinates are those of the padded image; its one-pixel frame is read as background.

	const int W = w + 2, H = h + 2;
	if (P.size() != (size_t)W * (size_t)H)
		throw std::invalid_argument ("gather_multicontour: padded image of " + std::to_string(P.size()) + " pixels is not " + std::to_string(W) + " x " + std::to_string(H));

	RamGrid g (P, W, H);
	trace_borders (g, w, h, multicountour, is_hole);

	if (verbose >= 4)
		for (size_t k = 0; k < multicountour.size(); k++)
			dump_2d_image_with_vertex_chain (P, multicountour[k], W, H, "\n\nBorder " + std::to_string(k + 1) + "\n", "||K||=" + std::to_string(multicountour[k].size()) + "\n");
}

/*static*/ void ContourFeature::buildRegularContour(LR& r, const Fsettings& s)
{
	// cast ROI's pixel cloud to a padded image 

	int width = r.aabb.get_width(),
		height = r.aabb.get_height(),
		base_x = r.aabb.get_xmin(),
		base_y = r.aabb.get_ymin();
	int paddingColor = 0;
	std::vector<PixIntens> paddedImage((height + 2) * (width + 2), paddingColor);

	// Rebuild the contour cache from scratch for this ROI.
	r.multicontour_.clear();
	r.multicontour_hole_.clear();

	for (auto px : r.raw_pixels)
	{
		auto x = px.x - base_x + 1,
			y = px.y - base_y + 1;
		paddedImage.at(x + y * (width + 2)) = px.inten + 1;	// we build a contour keeping corresponding intensities
	}

	if (STNGS_VERBOSLVL(s) >= 4)
		dump_2d_image_1d_layout(paddedImage, width + 2, height + 2, "\n\n\n ContourFeature / buildRegularContour / Padded image ROI " + std::to_string(r.aabb.get_width()) + " W " + std::to_string(r.aabb.get_height()) + " H \n", "\n\n\n");

	// gather all the contours
	gather_multicontour (r.multicontour_, paddedImage, width, height, STNGS_VERBOSLVL(s), &r.multicontour_hole_);

	// padded-image positions back to absolute: the padded image starts one pixel before the AABB
	for (std::vector<Pixel2> &K : r.multicontour_)
		for (Pixel2 &p : K)
		{
			p.x += base_x - 1;
			p.y += base_y - 1;
		}
}

void ContourFeature::buildRegularContour_nontriv (LR& r, const Fsettings& s)
{
	// The contour buildRegularContour() traces, of the same padded image held on disk: a ROI reaches
	// here because its box is too large for RAM, and the padded image and its border labels are each
	// the size of that box. The label image is filled by the walks alone (see DiskGrid).

	int width = r.aabb.get_width(),
		height = r.aabb.get_height(),
		base_x = r.aabb.get_xmin(),
		base_y = r.aabb.get_ymin();
	const int W = width + 2, H = height + 2;

	WriteImageMatrix_nontriv paddedImage ("paddedImage", r.label),
		labels ("contourLabels", r.label);
	// allocate() writes from the file's current position, so each image is allocated once, before
	// any pixel is set
	paddedImage.allocate (W, H, 0);
	labels.allocate (W, H, 0);

	for (auto px : r.raw_pixels_NT)
	{
		r.aabb.require_contains (px.x, px.y);
		size_t i = (px.x - base_x + 1) + (size_t) (px.y - base_y + 1) * W;
		paddedImage.set_at (i, (PixIntens) (px.inten + 1));	// as buildRegularContour() stores it
	}

	DiskGrid g (paddedImage, labels, W);
	trace_borders (g, width, height, r.multicontour_, &r.multicontour_hole_);

	if (STNGS_VERBOSLVL(s) >= 4)
		for (size_t k = 0; k < r.multicontour_.size(); k++)
			std::cout << "ContourFeature / buildRegularContour_nontriv / ROI " << r.label << " border " << k + 1 << " ||K||=" << r.multicontour_[k].size() << "\n";

	// padded-image positions back to absolute: the padded image starts one pixel before the AABB
	for (std::vector<Pixel2> &K : r.multicontour_)
		for (Pixel2 &p : K)
		{
			p.x += base_x - 1;
			p.y += base_y - 1;
		}
}

void ContourFeature::buildWholeSlideContour(LR& r)
{
	// Create 4 slide corner vertices of slide's max intensity
	PixIntens maxI = r.aux_max;
	Pixel2 tl (r.aabb.get_xmin(), r.aabb.get_ymin(), maxI),
		tr (r.aabb.get_xmax(), r.aabb.get_ymin(), maxI),
		bl (r.aabb.get_xmin(), r.aabb.get_ymax(), maxI),
		br (r.aabb.get_xmax(), r.aabb.get_ymax(), maxI);

	std::vector<Pixel2> K;
	K.push_back(tl);
	K.push_back(tr);
	K.push_back(br);
	K.push_back(bl);
	r.multicontour_.clear();
	r.multicontour_.push_back (K);
	r.multicontour_hole_.clear();
}

void ContourFeature::calculate (LR& r, const Fsettings& s)
{
	if (STNGS_SINGLEROI(s))
		buildWholeSlideContour(r);
	else
		buildRegularContour(r, s);

	measure_contour (r);
}

void ContourFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{}

void ContourFeature::osized_calculate (LR& r, const Fsettings& s, ImageLoader& imloader)
{
	// a whole slide is refused before it reaches the out-of-core path, so only a segmented ROI does
	buildRegularContour_nontriv(r, s);

	measure_contour (r);
}

void ContourFeature::measure_contour (LR& r)
{
	// the ROI's contour pixels, each once, kept on the ROI for every feature that reads the contour
	r.merge_multicontour();
	const std::vector<Pixel2>& K = r.contour_pixels_;

	// is there any contour?
	if (K.empty())
	{
		fval_PERIMETER =
		fval_DIAMETER_EQUAL_PERIMETER =
		fval_EDGE_MEAN_INTENSITY =
		fval_EDGE_STDDEV_INTENSITY =
		fval_EDGE_MAX_INTENSITY =
		fval_EDGE_MIN_INTENSITY =
		fval_EDGE_INTEGRATEDINTENSITY = 0;
		return;
	}

	// ok to calculate features
	// PERIMETER is the summed length of the ROI's closed contours, each walked back to its start
	fval_PERIMETER = 0;
	for (const std::vector<Pixel2>& C : r.multicontour_)
	{
		size_t clen = C.size();
		if (clen < 2)
			continue;
		for (size_t i = 0; i < clen; i++)
			fval_PERIMETER += std::sqrt (C[i == 0 ? clen - 1 : i - 1].sqdist(C[i]));
	}

	fval_DIAMETER_EQUAL_PERIMETER = fval_PERIMETER / M_PI;
	auto [cmin, cmax, cmean, cstddev] = calc_min_max_mean_stddev_intensity (K);
	fval_EDGE_MEAN_INTENSITY = cmean;
	fval_EDGE_STDDEV_INTENSITY = cstddev;
	fval_EDGE_MAX_INTENSITY  = cmax;
	fval_EDGE_MIN_INTENSITY = cmin;

	fval_EDGE_INTEGRATEDINTENSITY = 0;
	for (const Pixel2 &px : K)
		fval_EDGE_INTEGRATEDINTENSITY += px.inten;
}

void ContourFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[(int)Feature2D::PERIMETER][0] = fval_PERIMETER;
	fvals[(int)Feature2D::DIAMETER_EQUAL_PERIMETER][0] = fval_DIAMETER_EQUAL_PERIMETER;
	fvals[(int)Feature2D::EDGE_MEAN_INTENSITY][0] = fval_EDGE_MEAN_INTENSITY;
	fvals[(int)Feature2D::EDGE_STDDEV_INTENSITY][0] = fval_EDGE_STDDEV_INTENSITY;
	fvals[(int)Feature2D::EDGE_MAX_INTENSITY][0] = fval_EDGE_MAX_INTENSITY;
	fvals[(int)Feature2D::EDGE_MIN_INTENSITY][0] = fval_EDGE_MIN_INTENSITY;
	fvals[(int)Feature2D::EDGE_INTEGRATED_INTENSITY][0] = fval_EDGE_INTEGRATEDINTENSITY;
}

void ContourFeature::cleanup_instance()
{}

std::tuple<double, double, double, double> ContourFeature::calc_min_max_mean_stddev_intensity (const std::vector<Pixel2>& contour_pixels)
{
	Moments4 m;
	for (auto px : contour_pixels)
		m.add(px.inten);

	StatsReal min_ = m.min__(),
		max_ = m.max__(),
		mean_ = m.mean(),
		stddev_ = m.std();

	return { min_, max_, mean_, stddev_ };
}

void ContourFeature::extract (LR& r, const Fsettings& s)
{
	ContourFeature f;
	f.calculate (r, s);
	f.save_value (r.fvals);
}

void ContourFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & fst, const Dataset & _)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		ContourFeature f;
		f.calculate (r, fst);
		f.save_value (r.fvals);
	}
}


