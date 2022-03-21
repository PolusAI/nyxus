#include "glcm.h"

void GLCMFeature::osized_calculate (LR& r, ImageLoader& imloader)
{
	// Calculate normalized graytones
	OOR_ReadMatrix G (imloader, r.aabb); 
	G.apply_normalizing_range (r.aux_min, r.aux_max, 255.0); 

	int Angles[] = { 0, 45, 90, 135 },
		nAngs = sizeof(Angles) / sizeof(Angles[0]);
	for (int i = 0; i < nAngs; i++)
		Extract_Texture_Features_nontriv (distance_parameter, Angles[i], G);
}

void GLCMFeature::Extract_Texture_Features_nontriv (
	int distance,
	int angle,
	const OOR_ReadMatrix& grays)
{
	int nrows = grays.get_height();
	int ncols = grays.get_width();

	int tone_LUT [PGM_MAXMAXVAL + 1]; // LUT mapping gray tone(0-255) to matrix indicies 
	int tone_count = 0; // number of tones actually in the img. atleast 1 less than 255 

	// Determine the number of different gray tones (not maxval) 
	for (int row = PGM_MAXMAXVAL; row >= 0; --row)
		tone_LUT[row] = -1;
	for (int row = nrows - 1; row >= 0; --row)
		for (int col = 0; col < ncols; ++col)
		{
			size_t v = grays.get_normed_at(row, col);
			tone_LUT [v] = v;
		}

	for (int row = PGM_MAXMAXVAL; row >= 0; --row)
		if (tone_LUT[row] != -1)
			tone_count++;

	// Use the number of different tones to build LUT 
	for (int row = 0, itone = 0; row <= PGM_MAXMAXVAL; row++)
		if (tone_LUT[row] != -1)
			tone_LUT[row] = itone++;

	// Allocate Px and Py vectors
	std::vector<double> Px(tone_count * 2), Py(tone_count);

	// Compute gray-tone spatial dependence matrix 
	SimpleMatrix<double> P_matrix(tone_count, tone_count);

	if (angle == 0)
		CoOcMat_Angle_0_nontriv (P_matrix, distance, grays, tone_LUT, tone_count);
	else if (angle == 45)
		CoOcMat_Angle_45_nontriv (P_matrix, distance, grays, tone_LUT, tone_count);
	else if (angle == 90)
		CoOcMat_Angle_90_nontriv (P_matrix, distance, grays, tone_LUT, tone_count);
	else if (angle == 135)
		CoOcMat_Angle_135_nontriv (P_matrix, distance, grays, tone_LUT, tone_count);
	else 
	{
		std::cout << "Error: Cannot create co-occurence matrix for unsupported angle " << angle << "\n";
		return;
	}

	// Compute the statistics for the spatial dependence matrix
	fvals_ASM.push_back(f1_asm(P_matrix, tone_count));
	fvals_contrast.push_back(f2_contrast(P_matrix, tone_count));
	fvals_correlation.push_back(f3_corr(P_matrix, tone_count, Px));
	fvals_variance.push_back(f4_var(P_matrix, tone_count));
	fvals_IDM.push_back(f5_idm(P_matrix, tone_count));
	fvals_sum_avg.push_back(f6_savg(P_matrix, tone_count, Px));
	double se = f8_sentropy(P_matrix, tone_count, Px);
	fvals_sum_entropy.push_back(se);
	fvals_sum_var.push_back(f7_svar(P_matrix, tone_count, se, Px));
	fvals_entropy.push_back(f9_entropy(P_matrix, tone_count));
	fvals_diff_var.push_back(f10_dvar(P_matrix, tone_count, Px));
	fvals_diff_entropy.push_back(f11_dentropy(P_matrix, tone_count, Px));
	fvals_meas_corr1.push_back(f12_icorr(P_matrix, tone_count, Px, Py));
	fvals_meas_corr2.push_back(f13_icorr(P_matrix, tone_count, Px, Py));
	fvals_max_corr_coef.push_back(0.0); // f14_maxcorr(P_matrix, tone_count);
}

// Compute gray-tone spatial dependence matrix 
void GLCMFeature::CoOcMat_Angle_0_nontriv (
	// out
	SimpleMatrix<double>& matrix,
	// in
	int distance,
	const OOR_ReadMatrix& grays,
	const int* tone_LUT,
	int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	int count = 0; // normalizing factor 

	// zero out matrix 
	matrix.fill(0.0);

	int rows = grays.get_height(),
		cols = grays.get_width();

	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col)
		{
			// only non-zero values count
			if (grays.get_normed_at(row, col) == 0)
				continue;

			// find x tone 
			if (col + d < cols && grays.get_normed_at(row, col + d))
			{
				x = tone_LUT[(int)grays.get_normed_at(row,col)];
				y = tone_LUT[(int)grays.get_normed_at(row, col + d)];
				matrix.xy(x, y)++;
				matrix.xy(y, x)++;
				count += 2;
			}
		}

	// normalize matrix 
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			if (count == 0)   // protect from error 
				matrix.xy(itone, jtone) = 0;
			else
				matrix.xy(itone, jtone) /= count;
}

void GLCMFeature::CoOcMat_Angle_90_nontriv (
	// out
	SimpleMatrix<double>& matrix,
	// in
	int distance,
	const OOR_ReadMatrix& grays,
	const int* tone_LUT,
	int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	int count = 0; // normalizing factor 

	matrix.fill(0.0);

	int rows = grays.get_height(),
		cols = grays.get_width();

	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			// only non-zero values count
			if (grays.get_normed_at(row, col) == 0)
				continue;

			// find x tone 
			if (row + d < rows && grays.get_normed_at(row + d, col)) {
				x = tone_LUT[(int)grays.get_normed_at(row, col)];
				y = tone_LUT[(int)grays.get_normed_at(row + d, col)];
				matrix.xy(x, y)++;		
				matrix.xy(y, x)++;		
				count += 2;
			}
		}

	// normalize matrix 
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			if (count == 0)
				matrix.xy(itone, jtone) = 0;	
			else
				matrix.xy(itone, jtone) /= count;
}

void GLCMFeature::CoOcMat_Angle_45_nontriv (
	// out
	SimpleMatrix<double>& matrix,
	// in
	int distance,
	const OOR_ReadMatrix& grays,
	const int* tone_LUT,
	int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	int count = 0; // normalizing factor 

	matrix.fill(0.0);

	int rows = grays.get_height(),
		cols = grays.get_width();

	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			// only non-zero values count
			if (grays.get_normed_at(row, col) == 0)
				continue;

			// find x tone
			if (row + d < rows && col - d >= 0 && grays.get_normed_at(row + d, col - d)) {
				x = tone_LUT[(int)grays.get_normed_at(row, col - d)];
				y = tone_LUT[(int)grays.get_normed_at(row + d, col - d)];
				matrix.xy(x, y)++;		
				matrix.xy(y, x)++;		
				count += 2;
			}
		}

	/* normalize matrix */
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			if (count == 0)
				matrix.xy(itone, jtone) = 0;	// protect from error
			else
				matrix.xy(itone, jtone) /= count;	
}

void GLCMFeature::CoOcMat_Angle_135_nontriv (
	// out
	SimpleMatrix<double>& matrix,
	// in
	int distance,
	const OOR_ReadMatrix& grays,
	const int* tone_LUT,
	int tone_count)
{
	int d = distance;
	int x, y;
	int row, col, itone, jtone;
	int count = 0; // normalizing factor 

	matrix.fill(0.0);

	int rows = grays.get_height(),
		cols = grays.get_width();

	for (row = 0; row < rows; ++row)
		for (col = 0; col < cols; ++col) {
			// only non-zero values count
			if (grays.get_normed_at(row,col) == 0)
				continue;

			// find x tone 
			if (row + d < rows && col + d < cols && grays.get_normed_at(row + d, col + d)) 
			{
				x = tone_LUT[(int)grays.get_normed_at(row, col)];
				y = tone_LUT[(int)grays.get_normed_at(row + d, col + d)];
				matrix.xy(x, y)++;		//NONOPT
				matrix.xy(y, x)++;	
				count += 2;
			}
		}

	// normalize matrix 
	for (itone = 0; itone < tone_count; ++itone)
		for (jtone = 0; jtone < tone_count; ++jtone)
			if (count == 0)
				matrix.xy(itone, jtone) = 0;	// protect from error
			else
				matrix.xy(itone, jtone) /= count;	
}

