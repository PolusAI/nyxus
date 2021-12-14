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
#include "../sensemaker.h"
#include "image_matrix.h"
#include "../environment.h"

#ifndef __unix
#define NOMINMAX	// Prevent converting std::min(), max(), ... into macros
#include<windows.h>
#endif


#define RADIX 2.0
#define SIGN(x,y) ((y)<0 ? -fabs(x) : fabs(x))
#define SWAP(a,b) {y=(a);(a)=(b);(b)=y;}

/* support functions to compute f14_maxcorr */

void mkbalanced(double** a, int n)
{
	int last, j, i;
	double s, r, g, f, c, sqrdx;

	sqrdx = RADIX * RADIX;
	last = 0;
	while (last == 0) {
		last = 1;
		for (i = 1; i <= n; i++) {
			r = c = 0.0;
			for (j = 1; j <= n; j++)
				if (j != i) {
					c += fabs(a[j][i]);
					r += fabs(a[i][j]);
				}
			if (c && r) {
				g = r / RADIX;
				f = 1.0;
				s = c + r;
				while (c < g) {
					f *= RADIX;
					c *= sqrdx;
				}
				g = r * RADIX;
				while (c > g) {
					f /= RADIX;
					c /= sqrdx;
				}
				if ((c + r) / f < 0.95 * s) {
					last = 0;
					g = 1.0 / f;
					for (j = 1; j <= n; j++)
						a[i][j] *= g;
					for (j = 1; j <= n; j++)
						a[j][i] *= f;
				}
			}
		}
	}
}


void reduction(double** a, int n)
{
	int m, j, i;
	double y, x;

	for (m = 2; m < n; m++)
	{
		x = 0.0;
		i = m;
		for (j = m; j <= n; j++)
		{
			if (fabs(a[j][m - 1]) > fabs(x))
			{
				x = a[j][m - 1];
				i = j;
			}
		}
		if (i != m)
		{
			for (j = m - 1; j <= n; j++)
				SWAP(a[i][j], a[m][j])
				for (j = 1; j <= n; j++)
					SWAP(a[j][i], a[j][m])
					a[j][i] = a[j][i];
		}
		if (x)
		{
			for (i = m + 1; i <= n; i++)
			{
				if ((y = a[i][m - 1]))
				{
					y /= x;
					a[i][m - 1] = y;
					for (j = m; j <= n; j++)
						a[i][j] -= y * a[m][j];
					for (j = 1; j <= n; j++)
						a[j][m] += y * a[j][i];
				}
			}
		}
	}
}

int hessenberg(double** a, int n, double wr[], double wi[])
{
	int nn, m, l, k, j, its, i, mmin;
	double z, y, x, w, v, u, t, s, r = 0.0, q = 0.0, p = 0.0, anorm;

	anorm = fabs(a[1][1]);
	for (i = 2; i <= n; i++)
		for (j = (i - 1); j <= n; j++)
			anorm += fabs(a[i][j]);
	nn = n;
	t = 0.0;
	while (nn >= 1)
	{
		its = 0;
		do
		{
			for (l = nn; l >= 2; l--)
			{
				s = fabs(a[l - 1][l - 1]) + fabs(a[l][l]);
				if (s == 0.0)
					s = anorm;
				if ((double)(fabs(a[l][l - 1]) + s) == s)
					break;
			}
			x = a[nn][nn];
			if (l == nn)
			{
				wr[nn] = x + t;
				wi[nn--] = 0.0;
			}
			else
			{
				y = a[nn - 1][nn - 1];
				w = a[nn][nn - 1] * a[nn - 1][nn];
				if (l == (nn - 1))
				{
					p = 0.5 * (y - x);
					q = p * p + w;
					z = sqrt(fabs(q));
					x += t;
					if (q >= 0.0)
					{
						z = p + SIGN(z, p);
						wr[nn - 1] = wr[nn] = x + z;
						if (z)
							wr[nn] = x - w / z;
						wi[nn - 1] = wi[nn] = 0.0;
					}
					else
					{
						wr[nn - 1] = wr[nn] = x + p;
						wi[nn - 1] = -(wi[nn] = z);
					}
					nn -= 2;
				}
				else
				{
					if (its == 30)
					{
						return 0;
					}
					if (its == 10 || its == 20)
					{
						t += x;
						for (i = 1; i <= nn; i++)
							a[i][i] -= x;
						s = fabs(a[nn][nn - 1]) + fabs(a[nn - 1][nn - 2]);
						y = x = 0.75 * s;
						w = -0.4375 * s * s;
					}
					++its;
					for (m = (nn - 2); m >= l; m--)
					{
						z = a[m][m];
						r = x - z;
						s = y - z;
						p = (r * s - w) / a[m + 1][m] + a[m][m + 1];
						q = a[m + 1][m + 1] - z - r - s;
						r = a[m + 2][m + 1];
						s = fabs(p) + fabs(q) + fabs(r);
						p /= s;
						q /= s;
						r /= s;
						if (m == l)
							break;
						u = fabs(a[m][m - 1]) * (fabs(q) + fabs(r));
						v = fabs(p) * (fabs(a[m - 1][m - 1]) +
							fabs(z) + fabs(a[m + 1][m + 1]));
						if ((double)(u + v) == v)
							break;
					}
					for (i = m + 2; i <= nn; i++)
					{
						a[i][i - 2] = 0.0;
						if (i != (m + 2))
							a[i][i - 3] = 0.0;
					}
					for (k = m; k <= nn - 1; k++)
					{
						if (k != m)
						{
							p = a[k][k - 1];
							q = a[k + 1][k - 1];
							r = 0.0;
							if (k != (nn - 1))
								r = a[k + 2][k - 1];
							if ((x = fabs(p) + fabs(q) + fabs(r)))
							{
								p /= x;
								q /= x;
								r /= x;
							}
						}
						if ((s = SIGN(sqrt(p * p + q * q + r * r), p)))
						{
							if (k == m)
							{
								if (l != m)
									a[k][k - 1] = -a[k][k - 1];
							}
							else
								a[k][k - 1] = -s * x;
							p += s;
							x = p / s;
							y = q / s;
							z = r / s;
							q /= p;
							r /= p;
							for (j = k; j <= nn; j++)
							{
								p = a[k][j] + q * a[k + 1][j];
								if (k != (nn - 1))
								{
									p += r * a[k + 2][j];
									a[k + 2][j] -= p * z;
								}
								a[k + 1][j] -= p * y;
								a[k][j] -= p * x;
							}
							mmin = nn < k + 3 ? nn : k + 3;
							for (i = l; i <= mmin; i++)
							{
								p = x * a[i][k] + y * a[i][k + 1];
								if (k != (nn - 1))
								{
									p += z * a[i][k + 2];
									a[i][k + 2] -= p * r;
								}
								a[i][k + 1] -= p * q;
								a[i][k] -= p;
							}
						}
					}
				}
			}
		} while (l < nn - 1);
	}
	return 1;
}

void simplesrt(int n, double arr[])
{
	int i, j;
	double a;

	for (j = 2; j <= n; j++)
	{
		a = arr[j];
		i = j - 1;
		while (i > 0 && arr[i] > a)
		{
			arr[i + 1] = arr[i];
			i--;
		}
		arr[i + 1] = a;
	}
}

double* allocate_vector(int nl, int nh);
void free_matrix(double** matrix, int nrh);
double** allocate_matrix(int nrl, int nrh, int ncl, int nch);


/* Returns the Maximal Correlation Coefficient */
/* Uncomment VALIDATE_NEW_f14_maxcorr to compare against the last implementation.
*/
#define VALIDATE_NEW_f14_maxcorr 
double f14_maxcorr(double** P, int Ng) {

	if (Ng < 3) return std::numeric_limits<double>::quiet_NaN(); //MM

#ifdef VALIDATE_NEW_f14_maxcorr
	int k;
	double* px, * py, ** Q_old;
	double* x, * iy;
	double old_f = 0.0;
	int i, j;
	double f = 0.444;

	// ArrayXXd_rowmajor Q(Ng, Ng);
	//???	std::vector<std::vector<double>> Q(Ng+1, std::vector<double>(Ng+1, 0.0));

	// Eigen::VectorXcd Q_eigen_values_sorted(Ng);
	std::vector<double> Q_eigen_values_sorted (Ng, 0.0);

#else
	int i, j;
	double f = 0.0;
	ArrayXXd_rowmajor Q(Ng, Ng);
	Eigen::VectorXcd Q_eigen_values_sorted(Ng);
	ArrayXXd_rowmajor P_e(Ng, Ng);

	for (i = 0; i < Ng; ++i) {
		for (j = 0; j < Ng; ++j) {
			P_e(i, j) = P[i][j];
		}
	}

	/* Find the Q matrix */
	_Q_calc_eigen(P_e, Ng, Q);
	/* Eigenvalues */
	Q_eigen_values_sorted = Q.matrix().eigenvalues();
	if (Q_eigen_values_sorted(2).imag() == 0) {
		f = sqrt(Q_eigen_values_sorted(2).real());
	}
	else {
		f = 0.0;
	}
#endif

#ifndef VALIDATE_NEW_f14_maxcorr
	return f;
#endif
#ifdef VALIDATE_NEW_f14_maxcorr
	/* Original runtime for Q calc (with bug): 2862
	   Last best runtime for eigen optimized: 732
		+ 958 (hessenberg)
		+ 642 (reduction)
	*/
	Q_old = allocate_matrix(1, Ng + 1, 1, Ng + 1);
	x = allocate_vector(1, Ng);
	iy = allocate_vector(1, Ng);
	px = allocate_vector(0, Ng);
	py = allocate_vector(0, Ng);
	/*
	* px[i] is the (i-1)th entry in the marginal probability matrix obtained
	* by summing the rows of p[i][j]
	*/
	for (i = 0; i < Ng; ++i) {
		for (j = 0; j < Ng; ++j) {
			px[i] += P[i][j];
			py[j] += P[i][j];
		}
	}
#if 0 //???
	/* Find the Q matrix with the very slow original method. */
	for (i = 0; i < Ng; ++i) {
		for (j = 0; j < Ng; ++j) {
			Q[i + 1][j + 1] = 0;
			for (k = 0; k < Ng; ++k)
				if (px[i] && py[k])
					Q[i + 1][j + 1] += P[i][k] * P[j][k] / px[i] / py[k];
		}
	}
	for (i = 0; i < Ng; ++i) {
		for (j = 0; j < Ng; ++j) {
			if (abs((Q_old[i + 1][j + 1] - Q[i][j]) / Q[i][j]) > pow(10.0, -9.0)) {
				fprintf(stderr, "Q != Q_old\n[%d][%d]: %E %E\n", i + 1, j + 1,
					Q[i][j], Q_old[i + 1][j + 1]); 
				/* throw std::invalid_argument( "Q != Q_old" ); */
			}
		}
	}
	/* fprintf(stderr, "Q validation passes to 9 significant figures.\n"); */
#endif

	/* Balance the matrix */
	mkbalanced(Q_old, Ng);
	/* Reduction to Hessenberg Form */
	reduction(Q_old, Ng);
	/* Finding eigenvalue for nonsymetric matrix using QR algorithm */
	if (!hessenberg(Q_old, Ng, x, iy)) {
		/* computation failed ! */
		old_f = 0.0;
	}
	else {
		/* Returns the sqrt of the second largest eigenvalue of Q.
		   Sorting the x matrix first with simplesrt is crucial, or you could
		   take x[3] (I say this based on inspection of x). The simplesrt
		   function was commented out or otherwise disabled in all prior
		   versions I could find online, and predates the incorporation of
		   this file in wnd-chrm.
		*/
		simplesrt(Ng, x);
		if (x[Ng - 1] >= 0)
			old_f = sqrt(x[Ng - 1]);
	}

	if (old_f != f && abs((old_f - f) / f) > pow(10.0, -5.0)) {
		fprintf(stderr, "old_f != f: %E %E\n", old_f, f);
		fprintf(stderr, "Top eigenvalues: <old> vs <new_real, new_imaginary>\n");
		for (i = 0; i < Ng; ++i) {
			fprintf(stderr, "\t%E vs (%E, %E)\n", x[Ng - i],
				Q_eigen_values_sorted[i], // .real(),
				Q_eigen_values_sorted[i]); // .imag());
			if (i == 4) break;
		}
		/* throw std::invalid_argument( "f != old_f" ); */
	}
	/* fprintf(stderr, "f validation passes to 5 significant figures.\n"); */

	for (i = 1; i <= Ng + 1; i++) 
		free(Q_old[i] + 1);
	free(Q_old + 1);
	free((char*)px);
	free((char*)py);
	free((x + 1));
	free((iy + 1));
	return f;
#endif

}

