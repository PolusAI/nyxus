#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "glszm.h"
#include "../environment.h"

GLSZMFeature::GLSZMFeature() : FeatureMethod("GLSZMFeature")
{
	provide_features({ 
		GLSZM_SAE,
		GLSZM_LAE,
		GLSZM_GLN,
		GLSZM_GLNN,
		GLSZM_SZN,
		GLSZM_SZNN,
		GLSZM_ZP,
		GLSZM_GLV,
		GLSZM_ZV,
		GLSZM_ZE,
		GLSZM_LGLZE,
		GLSZM_HGLZE,
		GLSZM_SALGLE,
		GLSZM_SAHGLE,
		GLSZM_LALGLE,
		GLSZM_LAHGLE });
}

void GLSZMFeature::osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {} // Not suporting

void GLSZMFeature::osized_calculate (LR& r, ImageLoader& imloader)
{
	//==== Check if the ROI is degenerate (equal intensity)
	if (r.aux_min == r.aux_max)
		return;

	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;
	std::vector<ACluster> Z;

	//==== While scanning clusters, learn unique intensities 
	std::unordered_set<PixIntens> U;

	int maxZoneArea = 0;

	// Copy the image matrix
	ReadImageMatrix_nontriv M(r.aabb);	//-- auto M = r.aux_image_matrix;

	WriteImageMatrix_nontriv D ("GLSZMFeature_osized_calculate_D", r.label);	//-- pixData& D = M.WriteablePixels();
	D.allocate (r.aabb.get_width(), r.aabb.get_height(), 0);

	//M.print("initial\n");

	// Number of zones
	const int VISITED = -1;
	for (int row = 0; row < M.get_height(); row++)
		for (int col = 0; col < M.get_width(); col++)
		{
			// Find a non-blank pixel
			auto pi = D.get_at (row, col);
			if (pi == 0 || int(pi) == VISITED)
				continue;

			// Found a gray pixel. Find same-intensity neighbourhood of it.
			std::vector<std::tuple<int, int>> history;
			int x = col, y = row;
			int zoneArea = 1;
			D.set_at(y, x, VISITED);
			// 
			for (;;)
			{
				if (D.safe(y, x + 1) && D.get_at(y, x + 1) == pi)
				{
					D.set_at(y, x + 1, VISITED);
					zoneArea++;

					//M.print("After x+1,y");

					// Remember this pixel
					history.push_back({ x,y });
					// Advance 
					x = x + 1;
					// Proceed
					continue;
				}
				if (D.safe(y + 1, x + 1) && D.get_at(y + 1, x + 1) == pi)
				{
					D.set_at(y + 1, x + 1, VISITED);
					zoneArea++;

					//M.print("After x+1,y+1");

					history.push_back({ x,y });
					x = x + 1;
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x) && D.get_at(y + 1, x) == pi)
				{
					D.set_at(y + 1, x, VISITED);
					zoneArea++;

					//M.print("After x,y+1");

					history.push_back({ x,y });
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x - 1) && D.get_at(y + 1, x - 1) == pi)
				{
					D.set_at(y + 1, x - 1, VISITED);
					zoneArea++;

					//M.print("After x-1,y+1");

					history.push_back({ x,y });
					x = x - 1;
					y = y + 1;
					continue;
				}

				// Return from the branch
				if (history.size() > 0)
				{
					// Recollect the coordinate where we diverted from
					std::tuple<int, int> prev = history[history.size() - 1];
					history.pop_back();
				}

				// We are done exploring this cluster
				break;
			}

			// Done scanning a cluster. Perform 3 actions:
			// --1
			U.insert(pi);

			// --2
			maxZoneArea = std::max(maxZoneArea, zoneArea);

			//std::stringstream ss;
			//ss << "End of cluster " << x << "," << y;
			//M.print (ss.str());

			// --3
			ACluster clu = { pi, zoneArea };
			Z.push_back(clu);
		}

	//M.print("finished");


	//==== Fill the SZ-matrix

	Ng = (decltype(Ng))U.size();
	Ns = maxZoneArea;
	Nz = (decltype(Nz))Z.size();
	Np = 1;

	// --Set to vector to be able to know each intensity's index
	std::vector<PixIntens> I(U.begin(), U.end());
	std::sort(I.begin(), I.end());	// Optional

	// --allocate the matrix
	P.allocate(Ns, Ng);

	// --iterate zones and fill the matrix
	for (auto& z : Z)
	{
		// row
		auto iter = std::find(I.begin(), I.end(), z.first);
		int row = int(iter - I.begin());
		// col
		int col = z.second - 1;	// 0-based => -1
		auto& k = P.xy(col, row);
		k++;
	}
}

void GLSZMFeature::calculate(LR& r)
{
	//==== Check if the ROI is degenerate (equal intensity)
	if (r.aux_min == r.aux_max)
		return;
	 
	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;
	std::vector<ACluster> Z;

	//==== While scanning clusters, learn unique intensities 
	std::unordered_set<PixIntens> U;

	int maxZoneArea = 0;

	// Copy the image matrix
	auto M = r.aux_image_matrix;
	pixData& D = M.WriteablePixels();

	// Squeeze the intensity range
	PixIntens piRange = r.aux_max - r.aux_min;		// Prepare ROI's intensity range
	unsigned int nGrays = theEnvironment.get_coarse_gray_depth();
	for (size_t i = 0; i < D.size(); i++)
		D[i] = Nyxus::to_grayscale (D[i], r.aux_min, piRange, nGrays);

	// Number of zones
	const int VISITED = -1;
	for (int row=0; row<M.height; row++)
		for (int col = 0; col < M.width; col++)
		{
			// Find a non-blank pixel
			auto pi = D.yx(row, col);
			if (pi == 0 || int(pi)==VISITED)
				continue;

			// Found a gray pixel. Find same-intensity neighbourhood of it.
			std::vector<std::tuple<int, int>> history;
			int x = col, y = row;
			int zoneArea = 1;
			D.yx(y,x) = VISITED;
			// 
			for(;;)
			{
				if (D.safe(y,x+1) && D.yx(y,x+1) == pi)
				{
					D.yx(y,x+1) = VISITED;
					zoneArea++;

					//M.print("After x+1,y");

					// Remember this pixel
					history.push_back({x,y});
					// Advance 
					x = x + 1;
					// Proceed
					continue;
				}
				if (D.safe(y + 1, x + 1) && D.yx(y + 1, x+1) == pi)
				{
					D.yx(y + 1, x+1) = VISITED;
					zoneArea++;

					//M.print("After x+1,y+1");

					history.push_back({ x,y });
					x = x + 1;
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x) && D.yx(y + 1, x) == pi)
				{
					D.yx(y + 1, x) = VISITED;
					zoneArea++;

					//M.print("After x,y+1");

					history.push_back({ x,y });
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x - 1) && D.yx(y + 1, x-1) == pi)
				{
					D.yx(y + 1, x-1) = VISITED;
					zoneArea++;

					//M.print("After x-1,y+1");

					history.push_back({ x,y });
					x = x - 1;
					y = y + 1;
					continue;
				}

				// Return from the branch
				if (history.size() > 0)
				{
					// Recollect the coordinate where we diverted from
					std::tuple<int, int> prev = history[history.size() - 1];
					history.pop_back();

				}

				// We are done exploring this cluster
				break;
			}

			// Done scanning a cluster. Perform 3 actions:
			// --1
			U.insert(pi);

			// --2
			maxZoneArea = std::max(maxZoneArea, zoneArea);

			//std::stringstream ss;
			//ss << "End of cluster " << x << "," << y;
			//M.print (ss.str());

			// --3
			ACluster clu = {pi, zoneArea};
			Z.push_back (clu);
		}

	//M.print("finished");


	//==== Fill the SZ-matrix

	Ng = (decltype(Ng)) U.size();
	Ns = maxZoneArea;
	Nz = (decltype(Nz)) Z.size();
	Np = 1;	

	// --Set to vector to be able to know each intensity's index
	std::vector<PixIntens> I (U.begin(), U.end());
	std::sort (I.begin(), I.end());	// Optional

	// --allocate the matrix
	P.allocate (Ns, Ng);
	
	// --iterate zones and fill the matrix
	for (auto& z : Z)
	{
		// row
		auto iter = std::find(I.begin(), I.end(), z.first);
		int row = int (iter - I.begin());
		// col
		int col = z.second - 1;	// 0-based => -1
		auto & k = P.xy(col, row);
		k++;
	}
}

void GLSZMFeature::save_value (std::vector<std::vector<double>>& fvals)
{
	fvals[GLSZM_SAE][0] = calc_SAE();
	fvals[GLSZM_LAE][0] = calc_LAE();
	fvals[GLSZM_GLN][0] = calc_GLN();
	fvals[GLSZM_GLNN][0] = calc_GLNN();
	fvals[GLSZM_SZN][0] = calc_SZN();
	fvals[GLSZM_SZNN][0] = calc_SZNN();
	fvals[GLSZM_ZP][0] = calc_ZP();
	fvals[GLSZM_GLV][0] = calc_GLV();
	fvals[GLSZM_ZV][0] = calc_ZV();
	fvals[GLSZM_ZE][0] = calc_ZE();
	fvals[GLSZM_LGLZE][0] = calc_LGLZE();
	fvals[GLSZM_HGLZE][0] = calc_HGLZE();
	fvals[GLSZM_SALGLE][0] = calc_SALGLE();
	fvals[GLSZM_SAHGLE][0] = calc_SAHGLE();
	fvals[GLSZM_LALGLE][0] = calc_LALGLE();
	fvals[GLSZM_LAHGLE][0] = calc_LAHGLE();
}

// 1. Small Area Emphasis
double GLSZMFeature::calc_SAE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i=1; i<=Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			f += P.matlab(i,j) / (j * j);
		}
	}
	double retval = f / double(Nz);

	return retval;
}

// 2. Large Area Emphasis
double GLSZMFeature::calc_LAE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			f += P.matlab(i, j) * double (j * j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 3. Gray Level Non - Uniformity
double GLSZMFeature::calc_GLN()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double sum = 0.0;
		for (int j = 1; j <= Ns; j++)
		{
			sum += P.matlab(i,j);
		}
		f += sum * sum;
	}
	double retval = f / double(Nz);
	return retval;
}

// 4. Gray Level Non - Uniformity Normalized
double GLSZMFeature::calc_GLNN()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		double sum = 0.0;
		for (int j = 1; j <= Ns; j++)
		{
			sum += P.matlab(i,j); 
		}
		f += sum * sum;
	}
	double retval = f / double(Nz * Nz);
	return retval;
}

// 5. Size - Zone Non - Uniformity
double GLSZMFeature::calc_SZN()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ns; i++)
	{
		double sum = 0.0;
		for (int j = 1; j <= Ng; j++)
		{
			sum += P.matlab(j,i); 
		}
		f += sum * sum;
	}
	double retval = f / double(Nz);
	return retval;
}

// 6. Size - Zone Non - Uniformity Normalized
double GLSZMFeature::calc_SZNN()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ns; i++)
	{
		double sum = 0.0;
		for (int j = 1; j <= Ng; j++)
		{
			sum += P.matlab(j,i); 
		}
		f += sum * sum;
	}
	double retval = f / double(Nz * Nz);
	return retval;
}

// 7. Zone Percentage
double GLSZMFeature::calc_ZP()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double retval = double(Nz) / double(Np);
	return retval;
}

// 8. Gray Level Variance
double GLSZMFeature::calc_GLV()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double mu = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			mu += P.matlab(i, j) * i;  
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			double mu2 = (i - mu) * (i - mu);
			f += P.matlab(i,j) * mu2;
		}
	}
	return f;
}

// 9. Zone Variance
double GLSZMFeature::calc_ZV()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double mu = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			mu += P.matlab(i,j) * double(j);
		}
	}

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			double mu2 = (j - mu) * (j - mu);
			f += P.matlab(i, j) * mu2;
		}
	}
	return f;
}

// 10. Zone Entropy
double GLSZMFeature::calc_ZE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			double entrTerm = log2(P.matlab(i,j) + EPS);
			f += P.matlab(i,j) * entrTerm;
		}
	}
	double retval = -f;
	return retval;
}

// 11. Low Gray Level Zone Emphasis
double GLSZMFeature::calc_LGLZE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			f += P.matlab(i,j) / double(i*i);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 12. High Gray Level Zone Emphasis
double GLSZMFeature::calc_HGLZE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j <= Ns; j++)
		{
			f += P.matlab(i,j) * double(i*i);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 13. Small Area Low Gray Level Emphasis
double GLSZMFeature::calc_SALGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i <= Ng; i++)
	{
		for (int j = 1; j < Ns; j++)
		{
			f += P.matlab(i,j) / double(i * i * j * j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 14. Small Area High Gray Level Emphasis
double GLSZMFeature::calc_SAHGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i < Ng; i++)
	{
		for (int j = 1; j < Ns; j++)
		{
			f += P.matlab(i,j) * double(i * i) / double(j * j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 15. Large Area Low Gray Level Emphasis
double GLSZMFeature::calc_LALGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i < Ng; i++)
	{
		for (int j = 1; j < Ns; j++)
		{
			f += P.matlab(i,j) * double(j * j) / double(i * i);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

// 16. Large Area High Gray Level Emphasis
double GLSZMFeature::calc_LAHGLE()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double f = 0.0;
	for (int i = 1; i < Ng; i++)
	{
		for (int j = 1; j < Ns; j++)
		{
			f += P.matlab(i,j) * double(i * i * j * j);
		}
	}
	double retval = f / double(Nz);
	return retval;
}

void GLSZMFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		if (r.has_bad_data())
			continue;

		GLSZMFeature f;
		f.calculate(r);
		f.save_value(r.fvals);
	}
}

