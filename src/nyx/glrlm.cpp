#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "glrlm.h"

const double EPS = 2.2e-16;
const double BAD_ROI_FVAL = 0.0;

std::vector<double> GLRLM_features::rotAngles = {0, 45, 90, 135};

void GLRLM_features::initialize (int minI, int maxI, const ImageMatrix& im)
{
	//==== Check if the ROI is degenerate (equal intensity)
	if (minI == maxI)
	{
		bad_roi_data = true;
		return;
	}

	//--debug-- im.print("initial ROI\n");

	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;
	using AngleZones = std::vector<ACluster>;
	//--unnec--	std::vector<AngleZones> angles_Z;

	//==== While scanning clusters, learn unique intensities 
	using AngleUniqInte = std::unordered_set<PixIntens>;
	//--unnec--	std::vector<AngleUniqInte>  angles_U;

	//==== Iterate angles 0,45,90,135
	for (int angleIdx = 0; angleIdx < 4; angleIdx++)
	{
		// Clusters at angle 'angleIdx'
		AngleZones Z;

		// Unique intensities at angle 'angleIdx'
		AngleUniqInte U;

		// We need it to estimate the x-dimension of matrix P
		int maxZoneArea = 0;

		// Copy the image matrix. We'll use it to maintain state of cluster scanning 
		auto M = im;
		pixData& D = M.WriteablePixels();

		// Number of zones
		const int VISITED = -1;

		// Scan the image and check non-blank pixels' clusters
		for (int row = 0; row < M.height; row++)
			for (int col = 0; col < M.width; col++)
			{
				// Find a non-blank pixel
				auto pi = D(row, col);
				if (pi == 0 || int(pi) == VISITED)
					continue;

				// Found a non-blank (gray) pixel. Find same-intensity neighbourhood of it.
				std::vector<std::tuple<int, int>> history;
				int x = col, y = row;
				int zoneArea = 1;
				D(y,x) = VISITED;

				// State machine scanning the rest of the cluster
				for (;;)
				{
					// angleIdx==0 === 0 degrees
					if (angleIdx==0 && D.safe(y, x + 1) && D(y, x + 1) == pi)
					{
						D(y, x + 1) = VISITED;
						zoneArea++;

						//--debug-- M.print("After x+1,y");

						// Remember this pixel
						history.push_back({ x,y });
						// Advance 
						x = x + 1;
						// Proceed
						continue;
					}

					// angleIdx==1 === 45 degrees
					if (D.safe(y + 1, x + 1) && D(y + 1, x + 1) == pi)
					{
						D(y + 1, x + 1) = VISITED;
						zoneArea++;

						//--debug-- M.print("After x+1,y+1");

						history.push_back({ x,y });
						x = x + 1;
						y = y + 1;
						continue;
					}

					// angleIdx==2 === 90 degrees
					if (D.safe(y + 1, x) && D(y + 1, x) == pi)
					{
						D(y + 1, x) = VISITED;
						zoneArea++;

						//--debug-- M.print("After x,y+1");

						history.push_back({ x,y });
						y = y + 1;
						continue;
					}

					// angleIdx==3 === 135 degrees
					if (D.safe(y + 1, x - 1) && D(y + 1, x - 1) == pi)
					{
						D(y + 1, x - 1) = VISITED;
						zoneArea++;

						//--debug-- M.print("After x-1,y+1");

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

		//==== Fill the zone matrix

		int Ng = (decltype(Ng))U.size();
		int Nr = maxZoneArea;
		int Nz = (decltype(Nz))Z.size();
		int Np = 1;

		// --Set to vector to be able to know each intensity's index
		std::vector<PixIntens> I(U.begin(), U.end());
		std::sort(I.begin(), I.end());	// Optional

		// --allocate the matrix
		P_matrix P;
		P.allocate (Nr, Ng);

		// --iterate zones and fill the matrix
		for (auto& z : Z)
		{
			// row
			auto iter = std::find(I.begin(), I.end(), z.first);
			int row = int(iter - I.begin());
			// col
			int col = z.second - 1;	// 0-based => -1
			// update the matrix
			auto& k = P(col, row);
			k++;
		}

		// --save this angle's results
		angles_P.push_back (P);
		angles_Ng.push_back (Ng);
		angles_Nr.push_back (Nr);
		angles_Np.push_back (Np);
		//--unnec-- angles_U.push_back (U);
		//--unnec-- angles_Z.push_back (Z);
	}
}


// 1. Short Run Emphasis 
// ai - angle index
void GLRLM_features::calc_SRE (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) / (j * j);
			}
		}

		double retval = f / double(Nr);
		af.push_back (retval);
	}
}

// 2. Long Run Emphasis 
void GLRLM_features::calc_LRE (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) * j*j;
			}
		}

		double retval = f / double(Nr);
		af.push_back(retval);
	}
}

// 3. Gray Level Non-Uniformity 
void GLRLM_features::calc_GLN (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			double sum = 0.0;
			for (int j = 1; j <= Nr; j++)
			{
				sum += P.matlab(i, j);
			}
			f += sum * sum;
		}

		double retval = f / double(Nr);
		af.push_back(retval);
	}
}

// 4. Gray Level Non-Uniformity Normalized 
void GLRLM_features::calc_GLNN (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			double sum = 0.0;
			for (int j = 1; j <= Nr; j++)
			{
				sum += P.matlab(i, j);
			}
			f += sum * sum;
		}

		double retval = f / double(Nr*Nr);
		af.push_back(retval);
	}
}

// 5. Run Length Non-Uniformity
void GLRLM_features::calc_RLN (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int j = 1; j <= Nr; j++)
		{
			double sum = 0.0;
			for (int i = 1; i <= Ng; i++)
			{
				sum += P.matlab(i, j);
			}
			f += sum * sum;
		}

		double retval = f / double(Nr);
		af.push_back(retval);
	}
}

// 6. Run Length Non-Uniformity Normalized 
void GLRLM_features::calc_RLNN (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int j = 1; j <= Nr; j++)
		{
			double sum = 0.0;
			for (int i = 1; i <= Ng; i++)
			{
				sum += P.matlab(i, j);
			}
			f += sum * sum;
		}

		double retval = f / double(Nr*Nr);
		af.push_back(retval);
	}
}

// 7. Run Percentage
void GLRLM_features::calc_RP (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Np = angles_Np[ai],
			Nr = angles_Nr[ai];

		double retval = double(Nr / Np);
		af.push_back(retval);
	}
}

// 8. Gray Level Variance 
void GLRLM_features::calc_GLV (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double mu = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				mu += P.matlab(i, j) * i;
			}
		}

		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				double mu2 = (i - mu) * (i - mu);
				f += P.matlab(i, j) * mu2;
			}
		}
		af.push_back (f);
	}
}

// 9. Run Variance 
void GLRLM_features::calc_RV (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double mu = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				mu += P.matlab(i, j) * j;
			}
		}

		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				double mu2 = (j - mu) * (j - mu);
				f += P.matlab(i, j) * mu2;
			}
		}
		af.push_back(f);
	}
}

// 10. Run Entropy 
void GLRLM_features::calc_RE (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				double entrTerm = log2(P.matlab(i, j) + EPS);
				f += P.matlab(i, j) * entrTerm;
			}
		}
		double retval = -f;
		af.push_back(retval);
	}
}

// 11. Low Gray Level Run Emphasis 
void GLRLM_features::calc_LGLRE (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) / double(i * i);
			}
		}
		double retval = f / double(Nr);
		af.push_back(retval);
	}
}

// 12. High Gray Level Run Emphasis 
void GLRLM_features::calc_HGLRE (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) * double(i * i);
			}
		}
		double retval = f / double(Nr);
		af.push_back(retval);
	}
}

// 13. Short Run Low Gray Level Emphasis 
void GLRLM_features::calc_SRLGLE (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) / double(i * i * j * j);
			}
		}
		double retval = f / double(Nr);
		af.push_back(retval);
	}
}

// 14. Short Run High Gray Level Emphasis 
void GLRLM_features::calc_SRHGLE (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) * double(i * i) / double(j * j);
			}
		}
		double retval = f / double(Nr);
		af.push_back(retval);
	}
}

// 15. Long Run Low Gray Level Emphasis 
void GLRLM_features::calc_LRLGLE (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) * double(j * j) / double(i * i);
			}
		}
		double retval = f / double(Nr);
		af.push_back(retval);
	}
}

// 16. Long Run High Gray Level Emphasis 
void GLRLM_features::calc_LRHGLE (AngledFtrs& af)
{
	af.clear();

	// Prevent using bad data 
	if (bad_roi_data)
	{
		af.resize(4, BAD_ROI_FVAL);
		return;
	}

	for (int ai = 0; ai < 4; ai++)
	{
		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		// Calculate
		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				f += P.matlab(i, j) * double(i * i * j * j);
			}
		}
		double retval = f / double(Nr);
		af.push_back(retval);
	}
}
