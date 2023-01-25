#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include "glrlm.h"
#include "../environment.h"

GLRLMFeature::GLRLMFeature() : FeatureMethod("GLRLMFeature")
{
	provide_features({
		GLRLM_SRE,
		GLRLM_LRE,
		GLRLM_GLN,
		GLRLM_GLNN,
		GLRLM_RLN,
		GLRLM_RLNN,
		GLRLM_RP,
		GLRLM_GLV,
		GLRLM_RV,
		GLRLM_RE,
		GLRLM_LGLRE,
		GLRLM_HGLRE,
		GLRLM_SRLGLE,
		GLRLM_SRHGLE,
		GLRLM_LRLGLE,
		GLRLM_LRHGLE });
}

void GLRLMFeature::calculate (LR& r)
{
	auto minI = r.aux_min,
		maxI = r.aux_max;
	const ImageMatrix& im = r.aux_image_matrix;

	//==== Check if the ROI is degenerate (equal intensity => no texture)
	if (minI == maxI)
	{
		// insert zero for all 4 angles to make the output expecting 4-angled values happy
		angled_SRE.resize(4, 0);	
		angled_LRE.resize(4, 0);
		angled_GLN.resize(4, 0);
		angled_GLNN.resize(4, 0);
		angled_RLN.resize(4, 0);
		angled_RLNN.resize(4, 0);
		angled_RP.resize(4, 0);
		angled_GLV.resize(4, 0);
		angled_RV.resize(4, 0);
		angled_RE.resize(4, 0);
		angled_LGLRE.resize(4, 0);
		angled_HGLRE.resize(4, 0);
		angled_SRLGLE.resize(4, 0);
		angled_SRHGLE.resize(4, 0);
		angled_LRLGLE.resize(4, 0);
		angled_LRHGLE.resize(4, 0);

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

	// Prepare ROI's intensity range
	PixIntens piRange = r.aux_max - r.aux_min;

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

		// Squeeze the intensity range
		unsigned int nGrays = theEnvironment.get_coarse_gray_depth();

		for (size_t i = 0; i < D.size(); i++)
			D[i] = Nyxus::to_grayscale (D[i], r.aux_min, piRange, nGrays, Environment::ibsi_compliance);
		

		// Number of zones
		const int VISITED = -1;

		// Scan the image and check non-blank pixels' clusters
		for (int row = 0; row < M.height; row++)
			for (int col = 0; col < M.width; col++)
			{
				// Find a non-blank pixel
				auto pi = D.yx(row, col);
				if (pi == 0 || int(pi) == VISITED)
					continue;

				// Found a non-blank (gray) pixel. Find same-intensity neighbourhood of it.
				std::vector<std::tuple<int, int>> history;
				int x = col, y = row;
				int zoneArea = 1;
				D.yx(y,x) = VISITED;

				// State machine scanning the rest of the cluster
				for (;;)
				{
					// angleIdx==0 === 0 degrees
					if (angleIdx==0 && D.safe(y, x + 1) && D.yx(y, x + 1) == pi)
					{
						D.yx(y, x + 1) = VISITED;
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
					if (angleIdx==1 && D.safe(y + 1, x + 1) && D.yx(y + 1, x + 1) == pi)
					{
						D.yx(y + 1, x + 1) = VISITED;
						zoneArea++;

						//--debug-- M.print("After x+1,y+1");

						history.push_back({ x,y });
						x = x + 1;
						y = y + 1;
						continue;
					}

					// angleIdx==2 === 90 degrees
					if (angleIdx==2 && D.safe(y + 1, x) && D.yx(y + 1, x) == pi)
					{
						D.yx(y + 1, x) = VISITED;
						zoneArea++;

						//--debug-- M.print("After x,y+1");

						history.push_back({ x,y });
						y = y + 1;
						continue;
					}

					// angleIdx==3 === 135 degrees
					if (angleIdx== 3 && D.safe(y + 1, x - 1) && D.yx(y + 1, x - 1) == pi)
					{
						D.yx(y + 1, x - 1) = VISITED;
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

				// --3
				ACluster clu = { pi, zoneArea };
				Z.push_back(clu);
			}

		// count non-zero pixels
		int count = 0;

		for (const auto& px: im.ReadablePixels()) {
			if(px !=0) ++count;
		}

		//==== Fill the zone matrix

		int Ng = Environment::ibsi_compliance ? 
			*std::max_element(std::begin(im.ReadablePixels()), std::end(im.ReadablePixels())) : (decltype(Ng))U.size();
		int Nr = maxZoneArea;
		int Nz = (decltype(Nz))Z.size();
		int Np = count;

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
			int row = (Environment::ibsi_compliance) ? z.first - 1 : int(iter - I.begin());
			// col
			int col = z.second - 1;	// 0-based => -1
			// update the matrix
			auto& k = P.xy(col, row);
			k++;
		}

		// --save this angle's results
		angles_P.push_back (P);
		angles_Ng.push_back (Ng);
		angles_Nr.push_back (Nr);
		angles_Np.push_back (Np);
		//--unnec-- angles_U.push_back (U);
		//--unnec-- angles_Z.push_back (Z);

		double sum = 0;
		for (int i = 1; i <= Ng; ++i) {
			for (int j = 1; j <= Nr; ++j) {
				sum += P.matlab(i, j);
			}	
		}

		sum_p.push_back(sum);
	}

	calc_SRE (angled_SRE);
	calc_LRE (angled_LRE);
	calc_GLN (angled_GLN);
	calc_GLNN (angled_GLNN);
	calc_RLN (angled_RLN);
	calc_RLNN (angled_RLNN);
	calc_RP (angled_RP);
	calc_GLV (angled_GLV);
	calc_RV (angled_RV);
	calc_RE (angled_RE);
	calc_LGLRE (angled_LGLRE);
	calc_HGLRE (angled_HGLRE);
	calc_SRLGLE (angled_SRLGLE);
	calc_SRHGLE (angled_SRHGLE);
	calc_LRLGLE (angled_LRLGLE);
	calc_LRHGLE (angled_LRHGLE);
}

// Not supporting the online mode
void GLRLMFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {} // Not supporting

void GLRLMFeature::osized_calculate(LR& r, ImageLoader& imloader)
{
	auto minI = r.aux_min,
		maxI = r.aux_max;

	//==== Check if the ROI is degenerate (equal intensity => no texture)
	if (minI == maxI)
	{
		// insert zero for all 4 angles to make the output expecting 4-angled values happy
		angled_SRE.resize(4, 0);
		angled_LRE.resize(4, 0);
		angled_GLN.resize(4, 0);
		angled_GLNN.resize(4, 0);
		angled_RLN.resize(4, 0);
		angled_RLNN.resize(4, 0);
		angled_RP.resize(4, 0);
		angled_GLV.resize(4, 0);
		angled_RV.resize(4, 0);
		angled_RE.resize(4, 0);
		angled_LGLRE.resize(4, 0);
		angled_HGLRE.resize(4, 0);
		angled_SRLGLE.resize(4, 0);
		angled_SRHGLE.resize(4, 0);
		angled_LRLGLE.resize(4, 0);
		angled_LRHGLE.resize(4, 0);

		bad_roi_data = true;
		return;
	}

	ReadImageMatrix_nontriv im(r.aabb); //-- const ImageMatrix& im = r.aux_image_matrix;

	//--debug-- im.print("initial ROI\n");

	//==== Make a list of intensity clusters (zones)
	using ACluster = std::pair<PixIntens, int>;
	using AngleZones = std::vector<ACluster>;
	//--unnec--	std::vector<AngleZones> angles_Z;

	//==== While scanning clusters, learn unique intensities 
	using AngleUniqInte = std::unordered_set<PixIntens>;
	//--unnec--	std::vector<AngleUniqInte>  angles_U;

	//==== Iterate angles 0, 45, 90, 135
	for (int angleIdx = 0; angleIdx < 4; angleIdx++)
	{
		// Clusters at angle 'angleIdx'
		AngleZones Z;

		// Unique intensities at angle 'angleIdx'
		AngleUniqInte U;

		// We need it to estimate the x-dimension of matrix P
		int maxZoneArea = 0;

		// Copy the image matrix. We'll use it to maintain state of cluster scanning 
		
		//-- auto M = im;
		//-- pixData& D = M.WriteablePixels();
		WriteImageMatrix_nontriv D("GLRLMFeature_osized_calculate_D", r.label);
		D.init_with_cloud(r.raw_pixels_NT, r.aabb);

		// Number of zones
		const int VISITED = -1;

		// Scan the image and check non-blank pixels' clusters
		for (int row = 0; row < im.get_height(); row++)
			for (int col = 0; col < im.get_width(); col++)
			{
				// Find a non-blank pixel
				auto pi = D.get_at (row, col);
				if (pi == 0 || int(pi) == VISITED)
					continue;

				// Found a non-blank (gray) pixel. Find same-intensity neighbourhood of it.
				std::vector<std::tuple<int, int>> history;
				int x = col, y = row;
				int zoneArea = 1;
				D.set_at (y, x, VISITED);

				// State machine scanning the rest of the cluster
				for (;;)
				{
					// angleIdx==0 === 0 degrees
					if (angleIdx == 0 && D.safe(y, x + 1) && D.get_at(y, x + 1) == pi)
					{
						D.set_at(y, x + 1, VISITED);
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
					if (D.safe(y + 1, x + 1) && D.get_at(y + 1, x + 1) == pi)
					{
						D.set_at(y + 1, x + 1, VISITED);
						zoneArea++;

						//--debug-- M.print("After x+1,y+1");

						history.push_back({ x,y });
						x = x + 1;
						y = y + 1;
						continue;
					}

					// angleIdx==2 === 90 degrees
					if (D.safe(y + 1, x) && D.get_at(y + 1, x) == pi)
					{
						D.set_at(y + 1, x, VISITED);
						zoneArea++;

						//--debug-- M.print("After x,y+1");

						history.push_back({ x,y });
						y = y + 1;
						continue;
					}

					// angleIdx==3 === 135 degrees
					if (D.safe(y + 1, x - 1) && D.get_at(y + 1, x - 1) == pi)
					{
						D.set_at(y + 1, x - 1, VISITED);
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
		P.allocate(Nr, Ng);

		// --iterate zones and fill the matrix
		for (auto& z : Z)
		{
			// row
			auto iter = std::find(I.begin(), I.end(), z.first);
			int row = int(iter - I.begin());
			// col
			int col = z.second - 1;	// 0-based => -1
			// update the matrix
			auto& k = P.xy(col, row);
			k++;
		}

		// --save this angle's results
		angles_P.push_back(P);
		angles_Ng.push_back(Ng);
		angles_Nr.push_back(Nr);
		angles_Np.push_back(Np);
		//--unnec-- angles_U.push_back (U);
		//--unnec-- angles_Z.push_back (Z);

	}

	calc_SRE(angled_SRE);
	calc_LRE(angled_LRE);
	calc_GLN(angled_GLN);
	calc_GLNN(angled_GLNN);
	calc_RLN(angled_RLN);
	calc_RLNN(angled_RLNN);
	calc_RP(angled_RP);
	calc_GLV(angled_GLV);
	calc_RV(angled_RV);
	calc_RE(angled_RE);
	calc_LGLRE(angled_LGLRE);
	calc_HGLRE(angled_HGLRE);
	calc_SRLGLE(angled_SRLGLE);
	calc_SRHGLE(angled_SRHGLE);
	calc_LRLGLE(angled_LRLGLE);
	calc_LRHGLE(angled_LRHGLE);
}

void GLRLMFeature::save_value(std::vector<std::vector<double>>& fvals)
{
	fvals[GLRLM_SRE] = angled_SRE;
	fvals[GLRLM_LRE] = angled_LRE;
	fvals[GLRLM_GLN] = angled_GLN;
	fvals[GLRLM_GLNN] = angled_GLNN;
	fvals[GLRLM_RLN] = angled_RLN;
	fvals[GLRLM_RLNN] = angled_RLNN;
	fvals[GLRLM_RP] = angled_RP;
	fvals[GLRLM_GLV] = angled_GLV;
	fvals[GLRLM_RV] = angled_RV;
	fvals[GLRLM_RE] = angled_RE;
	fvals[GLRLM_LGLRE] = angled_LGLRE;
	fvals[GLRLM_HGLRE] = angled_HGLRE;
	fvals[GLRLM_SRLGLE] = angled_SRLGLE;
	fvals[GLRLM_SRHGLE] = angled_SRHGLE;
	fvals[GLRLM_LRLGLE] = angled_LRLGLE;
	fvals[GLRLM_LRHGLE] = angled_LRHGLE;
}


// 1. Short Run Emphasis 
// ai - angle index
void GLRLMFeature::calc_SRE (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Ng = angles_Ng[ai],
			Nr = angles_Nr[ai];
		const SimpleMatrix<int>& P = angles_P[ai];

		double f = 0.;
		std::vector<double> rj(Nr+1, 0.);
		for (int i = 1; i <= Ng; ++i) {
			for (int j = 1; j <= Nr; ++j) {
				rj[j] += P.matlab(i, j);
			}
		}

		for (int j = 1; j <= Nr; ++j) {
			f +=  rj[j] / (j * j);
		}

		double retval = f / double(sum_p[ai]);
		af.push_back (retval);
	}
}

// 2. Long Run Emphasis 
void GLRLMFeature::calc_LRE (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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

		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 3. Gray Level Non-Uniformity 
void GLRLMFeature::calc_GLN (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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

		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 4. Gray Level Non-Uniformity Normalized 
void GLRLMFeature::calc_GLNN (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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

		double retval = f / double(sum_p[ai]*sum_p[ai]);
		af.push_back(retval);
	}
}

// 5. Run Length Non-Uniformity
void GLRLMFeature::calc_RLN (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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

		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 6. Run Length Non-Uniformity Normalized 
void GLRLMFeature::calc_RLNN (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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

		double retval = f / double(sum_p[ai]*sum_p[ai]);
		af.push_back(retval);
	}
}

// 7. Run Percentage
void GLRLMFeature::calc_RP (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		// Get ahold of the requested angle's matrix and its related N parameters 
		int Np = angles_Np[ai];

		double retval = double(sum_p[ai] / Np);
		af.push_back(retval);
	}
}

// 8. Gray Level Variance 
void GLRLMFeature::calc_GLV (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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
				mu += P.matlab(i, j)/sum_p[ai] * i;
			}
		}

		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				double mu2 = (i - mu) * (i - mu);
				f += P.matlab(i, j)/sum_p[ai] * mu2;
			}
		}
		af.push_back (f);
	}
}

// 9. Run Variance 
void GLRLMFeature::calc_RV (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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
				mu += P.matlab(i, j)/sum_p[ai] * j;
			}
		}

		double f = 0.0;
		for (int i = 1; i <= Ng; i++)
		{
			for (int j = 1; j <= Nr; j++)
			{
				double mu2 = (j - mu) * (j - mu);
				f += P.matlab(i, j)/sum_p[ai] * mu2;
			}
		}
		af.push_back(f);
	}
}

// 10. Run Entropy 
void GLRLMFeature::calc_RE (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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
				double entrTerm = fast_log10(P.matlab(i, j)/sum_p[ai] + EPS) / LOG10_2;
				f += P.matlab(i, j)/sum_p[ai] * entrTerm;
			}
		}
		double retval = -f;
		af.push_back(retval);
	}
}

// 11. Low Gray Level Run Emphasis 
void GLRLMFeature::calc_LGLRE (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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
		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 12. High Gray Level Run Emphasis 
void GLRLMFeature::calc_HGLRE (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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
		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 13. Short Run Low Gray Level Emphasis 
void GLRLMFeature::calc_SRLGLE (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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
		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 14. Short Run High Gray Level Emphasis 
void GLRLMFeature::calc_SRHGLE (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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
		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 15. Long Run Low Gray Level Emphasis 
void GLRLMFeature::calc_LRLGLE (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}

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
		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

// 16. Long Run High Gray Level Emphasis 
void GLRLMFeature::calc_LRHGLE (AngledFtrs& af)
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
		if (sum_p[ai] == 0) {
			af.push_back(0.0);
			continue;
		}
		
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
		double retval = f / double(sum_p[ai]);
		af.push_back(retval);
	}
}

void GLRLMFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
	for (auto i = start; i < end; i++)
	{
		int lab = (*ptrLabels)[i];
		LR& r = (*ptrLabelData)[lab];

		GLRLMFeature glrlm;
		glrlm.calculate(r);
		glrlm.save_value(r.fvals);
	}
}

