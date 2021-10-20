#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "glszm.h"

const double EPS = 2.2e-16;
const double BAD_ROI_FVAL = 0.0;


void GLSZM_features::initialize(int minI, int maxI, const ImageMatrix& im)
{
	//==== Check if the ROI is degenerate (equal intensity)
	if (minI == maxI)
	{
		bad_roi_data = true;
		return;
	}
	 
	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;
	std::vector<ACluster> Z;

	//==== While scanning clusters, learn unique intensities 
	std::unordered_set<PixIntens> U;

	int maxZoneArea = 0;

	// Copy the image matrix
	auto M = im;
	pixData& D = M.MutablePixels();

	//M.print("initial\n");

	// Number of zones
	const int VISITED = -1;
	for (int row=0; row<M.height; row++)
		for (int col = 0; col < M.width; col++)
		{
			// Find a non-blank pixel
			auto pi = D(row, col);
			if (pi == 0 || int(pi)==VISITED)
				continue;

			// Found a gray pixel. Find same-intensity neighbourhood of it.
			std::vector<std::tuple<int, int>> history;
			int x = col, y = row;
			int zoneArea = 1;
			D(y,x) = VISITED;
			// 
			for(;;)
			{
				if (D.safe(y,x+1) && D(y,x+1) == pi)
				{
					D(y,x+1) = VISITED;
					zoneArea++;

					//M.print("After x+1,y");

					// Remember this pixel
					history.push_back({x,y});
					// Advance 
					x = x + 1;
					// Proceed
					continue;
				}
				if (D.safe(y + 1, x + 1) && D(y + 1, x+1) == pi)
				{
					D(y + 1, x+1) = VISITED;
					zoneArea++;

					//M.print("After x+1,y+1");

					history.push_back({ x,y });
					x = x + 1;
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x) && D(y + 1, x) == pi)
				{
					D(y + 1, x) = VISITED;
					zoneArea++;

					//M.print("After x,y+1");

					history.push_back({ x,y });
					y = y + 1;
					continue;
				}
				if (D.safe(y + 1, x - 1) && D(y + 1, x-1) == pi)
				{
					D(y + 1, x-1) = VISITED;
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
		auto & k = P(col, row);
		k++;
	}

}

// 1. Small Area Emphasis
double GLSZM_features::calc_SAE()
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
double GLSZM_features::calc_LAE()
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
double GLSZM_features::calc_GLN()
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
double GLSZM_features::calc_GLNN()
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
double GLSZM_features::calc_SZN()
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
double GLSZM_features::calc_SZNN()
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
double GLSZM_features::calc_ZP()
{
	// Prevent using bad data 
	if (bad_roi_data)
		return BAD_ROI_FVAL;

	double retval = double(Nz) / double(Np);
	return retval;
}

// 8. Gray Level Variance
double GLSZM_features::calc_GLV()
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
double GLSZM_features::calc_ZV()
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
double GLSZM_features::calc_ZE()
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
double GLSZM_features::calc_LGLZE()
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
double GLSZM_features::calc_HGLZE()
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
double GLSZM_features::calc_SALGLE()
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
double GLSZM_features::calc_SAHGLE()
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
double GLSZM_features::calc_LALGLE()
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
double GLSZM_features::calc_LAHGLE()
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
