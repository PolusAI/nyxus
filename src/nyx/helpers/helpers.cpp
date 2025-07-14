#include <numeric>
#include <vector>
#include "helpers.h"
#include "../3rdparty/dsyevj3.h"

namespace Nyxus
{

	void calc_cov_matrix (double K[3][3], const std::vector<std::vector<double>>& dataset)
	{
		K[0][0] = K[0][1] = K[0][2] =
		K[1][0] = K[1][1] = K[1][2] =
		K[2][0] = K[2][1] = K[2][2] = 0;

		if (dataset[0].size() != 3)
			return; // error

		size_t dim = 3; // assuming dataset[0].size()==3
		size_t num_observations = dataset.size(); // assuming rows are observations

		// calculate means for each variable
		std::vector<double> means(dim);
		for (size_t j = 0; j < dim; j++) 
		{
			std::vector<double> column_data;
			for (size_t i = 0; i < num_observations; i++) 
				column_data.push_back (dataset[i][j]);
			means[j] = calc_mean (column_data);
		}

		// calculate covariance for all pairs
		for (size_t i = 0; i < dim; i++)
		{
			for (size_t j = i; j < dim; j++)
			{
				// symmetric matrix, calculate the upper triangle
				std::vector<double> col_i_data;
				std::vector<double> col_j_data;
				for (size_t k = 0; k < num_observations; k++) 
				{
					col_i_data.push_back (dataset[k][i]);
					col_j_data.push_back (dataset[k][j]);
				}
				double cov = calc_covariance (col_i_data, means[i], col_j_data, means[j]);
				K[i][j] = cov;
				K[j][i] = cov; // symmetric counterpart
			}
		}
	}

	double calc_mean (const std::vector<double>& X)
	{
		double sum = std::accumulate(X.begin(), X.end(), 0.0);
		return sum / (double) X.size();
	}

	double calc_covariance (const std::vector<double>& data1, double mean1, const std::vector<double>& data2, double mean2)
	{
		double normfactor = (double)(data1.size() - 1),  // sample covariance
			sum_of_products = 0.0;

		for (size_t i = 0; i < data1.size(); i++)
			sum_of_products += (data1[i] - mean1) * (data2[i] - mean2);

		return sum_of_products / normfactor;
	}

	#define cswap(a,b) do { if(a < b) { double tmp = a; a = b; b = tmp; } } while(0)

	bool calc_eigvals (double w[3], const double A[3][3])
	{
		double Q[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
		w[0] = w[1] = w[2] = 0;

		int code = dsyevj3 (A, Q, w);

		// sort
		cswap(w[0], w[1]);
		cswap(w[1], w[2]);
		cswap(w[0], w[1]);

		return code == 0;
	}

}








