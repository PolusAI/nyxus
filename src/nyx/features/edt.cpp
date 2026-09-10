#include <algorithm>
#include "edt.h"

namespace Nyxus
{
	namespace
	{
		// A cell no site reaches. Distinguished from a real distance so that the row pass can leave
		// siteless columns out of its envelope instead of feeding it a sentinel magnitude.
		constexpr int64_t NO_SITE = -1;

		// Whether the rational a1/b1 is at most a2/b2, both denominators positive. Cross-multiplied
		// so that the envelope's breakpoints stay in integer arithmetic and cannot round to the
		// wrong side of a comparison.
		inline bool ratio_le (int64_t a1, int64_t b1, int64_t a2, int64_t b2)
		{
			return a1 * b2 <= a2 * b1;
		}
	}

	void exact_sqedt (
		const std::vector<Pixel2>& sites,
		StatsInt xmin,
		StatsInt ymin,
		size_t W,
		size_t H,
		std::vector<int64_t>& sqdist)
	{
		sqdist.assign (W * H, NO_SITE);

		if (W == 0 || H == 0)
			return;

		// Mark the sites falling inside the raster. A site is the only cell that ends either pass
		// holding zero, which is what the later passes recognize it by.
		bool any = false;
		for (const auto& s : sites)
		{
			int64_t x = (int64_t)s.x - (int64_t)xmin,
				y = (int64_t)s.y - (int64_t)ymin;
			if (x < 0 || y < 0 || x >= (int64_t)W || y >= (int64_t)H)
				continue;
			sqdist[(size_t)y * W + (size_t)x] = 0;
			any = true;
		}

		// With no site inside the raster every distance is undefined. Leave the buffer marked so and
		// let the caller decide, rather than inventing a finite value.
		if (! any)
			return;

		// Column pass: the vertical distance to the nearest site in the same column, by a forward and
		// a backward sweep, in place. A column with no site keeps NO_SITE throughout.
		for (size_t x = 0; x < W; x++)
		{
			int64_t d = NO_SITE;
			for (size_t y = 0; y < H; y++)
			{
				size_t i = y * W + x;
				if (sqdist[i] == 0)
					d = 0;
				else
				{
					if (d != NO_SITE)
						d++;
					sqdist[i] = d;
				}
			}

			d = NO_SITE;
			for (size_t y = H; y-- > 0; )
			{
				size_t i = y * W + x;
				if (sqdist[i] == 0)
					d = 0;
				else
				{
					if (d != NO_SITE)
						d++;
					if (d != NO_SITE && (sqdist[i] == NO_SITE || d < sqdist[i]))
						sqdist[i] = d;
				}
			}
		}

		// Row pass: the lower envelope of the parabolas f(q) = (x - q)^2 + g(q)^2 raised at every
		// column q the column pass reached. Every row has at least one, because a column carrying a
		// site is finite at every row, so every cell resolves to a finite distance.
		std::vector<int64_t> g (W);		// the row's column-pass distances, read while the row is overwritten
		std::vector<size_t> v (W);		// the envelope's parabola vertices, left to right
		std::vector<int64_t> zn (W), zd (W);	// breakpoint between v[k-1] and v[k], as a rational

		for (size_t y = 0; y < H; y++)
		{
			int64_t* row = &sqdist[y * W];
			std::copy (row, row + W, g.begin());

			int k = -1;

			for (size_t q = 0; q < W; q++)
			{
				if (g[q] == NO_SITE)
					continue;

				int64_t fq = g[q] * g[q] + (int64_t)q * (int64_t)q;

				if (k < 0)
				{
					k = 0;
					v[0] = q;
					continue;
				}

				while (true)
				{
					int64_t gv = g[v[k]],
						fv = gv * gv + (int64_t)v[k] * (int64_t)v[k],
						num = fq - fv,
						den = 2 * ((int64_t)q - (int64_t)v[k]);

					// The new parabola takes over left of the previous breakpoint, so that
					// breakpoint and the parabola it belonged to leave the envelope.
					if (k > 0 && ratio_le (num, den, zn[k], zd[k]))
					{
						k--;
						continue;
					}

					k++;
					v[k] = q;
					zn[k] = num;
					zd[k] = den;
					break;
				}
			}

			// Walk the row left to right, advancing through the envelope's segments.
			int kmax = k;
			k = 0;
			for (size_t q = 0; q < W; q++)
			{
				while (k < kmax && zn[k + 1] < (int64_t)q * zd[k + 1])
					k++;

				int64_t dx = (int64_t)q - (int64_t)v[k],
					gv = g[v[k]];
				row[q] = dx * dx + gv * gv;
			}
		}
	}
}
