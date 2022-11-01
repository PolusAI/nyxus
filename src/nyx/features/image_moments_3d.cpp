#include "../environment.h"
#include "image_moments_3d.h"

VolumeMomentsFeature::VolumeMomentsFeature() : FeatureMethod("VolumeMomentsFeature")
{
    provide_features({
        // raw moments
        D3_RAW_MOMENT_000,

        D3_RAW_MOMENT_010,
        D3_RAW_MOMENT_011,
        D3_RAW_MOMENT_012,
        D3_RAW_MOMENT_013,

        D3_RAW_MOMENT_020,
        D3_RAW_MOMENT_021,
        D3_RAW_MOMENT_022,
        D3_RAW_MOMENT_023,

        D3_RAW_MOMENT_030,
        D3_RAW_MOMENT_031,
        D3_RAW_MOMENT_032,
        D3_RAW_MOMENT_033,

        D3_RAW_MOMENT_100,
        D3_RAW_MOMENT_101,
        D3_RAW_MOMENT_102,
        D3_RAW_MOMENT_103,

        D3_RAW_MOMENT_110,
        D3_RAW_MOMENT_111,
        D3_RAW_MOMENT_112,
        D3_RAW_MOMENT_113,

        D3_RAW_MOMENT_120,
        D3_RAW_MOMENT_121,
        D3_RAW_MOMENT_122,
        D3_RAW_MOMENT_123,

        D3_RAW_MOMENT_200,
        D3_RAW_MOMENT_201,
        D3_RAW_MOMENT_202,
        D3_RAW_MOMENT_203,

        D3_RAW_MOMENT_210,
        D3_RAW_MOMENT_211,
        D3_RAW_MOMENT_212,
        D3_RAW_MOMENT_213,

        D3_RAW_MOMENT_300,
        D3_RAW_MOMENT_301,
        D3_RAW_MOMENT_302,
        D3_RAW_MOMENT_303,

        // normalized raw moments
        D3_NORM_RAW_MOMENT_000,
        D3_NORM_RAW_MOMENT_010,
        D3_NORM_RAW_MOMENT_011,
        D3_NORM_RAW_MOMENT_012,
        D3_NORM_RAW_MOMENT_013,
        D3_NORM_RAW_MOMENT_020,
        D3_NORM_RAW_MOMENT_021,
        D3_NORM_RAW_MOMENT_022,
        D3_NORM_RAW_MOMENT_023,
        D3_NORM_RAW_MOMENT_030,
        D3_NORM_RAW_MOMENT_031,
        D3_NORM_RAW_MOMENT_032,
        D3_NORM_RAW_MOMENT_033,
        D3_NORM_RAW_MOMENT_100,
        D3_NORM_RAW_MOMENT_101,
        D3_NORM_RAW_MOMENT_102,
        D3_NORM_RAW_MOMENT_103,
        D3_NORM_RAW_MOMENT_200,
        D3_NORM_RAW_MOMENT_201,
        D3_NORM_RAW_MOMENT_202,
        D3_NORM_RAW_MOMENT_203,
        D3_NORM_RAW_MOMENT_300,
        D3_NORM_RAW_MOMENT_301,
        D3_NORM_RAW_MOMENT_302,
        D3_NORM_RAW_MOMENT_303,

        // central moments
        D3_CENTRAL_MOMENT_020,
        D3_CENTRAL_MOMENT_021,
        D3_CENTRAL_MOMENT_022,
        D3_CENTRAL_MOMENT_023,

        D3_CENTRAL_MOMENT_030,
        D3_CENTRAL_MOMENT_031,
        D3_CENTRAL_MOMENT_032,
        D3_CENTRAL_MOMENT_033,

        D3_CENTRAL_MOMENT_110,
        D3_CENTRAL_MOMENT_111,
        D3_CENTRAL_MOMENT_112,
        D3_CENTRAL_MOMENT_113,

        D3_CENTRAL_MOMENT_120,
        D3_CENTRAL_MOMENT_121,
        D3_CENTRAL_MOMENT_122,
        D3_CENTRAL_MOMENT_123,

        D3_CENTRAL_MOMENT_200,
        D3_CENTRAL_MOMENT_201,
        D3_CENTRAL_MOMENT_202,
        D3_CENTRAL_MOMENT_203,

        D3_CENTRAL_MOMENT_210,
        D3_CENTRAL_MOMENT_211,
        D3_CENTRAL_MOMENT_212,
        D3_CENTRAL_MOMENT_213,

        D3_CENTRAL_MOMENT_300,
        D3_CENTRAL_MOMENT_301,
        D3_CENTRAL_MOMENT_302,
        D3_CENTRAL_MOMENT_303,

        // normalized central moments
        D3_NORM_CENTRAL_MOMENT_020,
        D3_NORM_CENTRAL_MOMENT_021,
        D3_NORM_CENTRAL_MOMENT_022,
        D3_NORM_CENTRAL_MOMENT_023,

        D3_NORM_CENTRAL_MOMENT_030,
        D3_NORM_CENTRAL_MOMENT_031,
        D3_NORM_CENTRAL_MOMENT_032,
        D3_NORM_CENTRAL_MOMENT_033,

        D3_NORM_CENTRAL_MOMENT_110,
        D3_NORM_CENTRAL_MOMENT_111,
        D3_NORM_CENTRAL_MOMENT_112,
        D3_NORM_CENTRAL_MOMENT_113,

        D3_NORM_CENTRAL_MOMENT_120,
        D3_NORM_CENTRAL_MOMENT_121,
        D3_NORM_CENTRAL_MOMENT_122,
        D3_NORM_CENTRAL_MOMENT_123,

        D3_NORM_CENTRAL_MOMENT_200,
        D3_NORM_CENTRAL_MOMENT_201,
        D3_NORM_CENTRAL_MOMENT_202,
        D3_NORM_CENTRAL_MOMENT_203,

        D3_NORM_CENTRAL_MOMENT_210,
        D3_NORM_CENTRAL_MOMENT_211,
        D3_NORM_CENTRAL_MOMENT_212,
        D3_NORM_CENTRAL_MOMENT_213,

        D3_NORM_CENTRAL_MOMENT_300,
        D3_NORM_CENTRAL_MOMENT_301,
        D3_NORM_CENTRAL_MOMENT_302,
        D3_NORM_CENTRAL_MOMENT_303,
    });

    add_dependencies({ PERIMETER });
}

void VolumeMomentsFeature::calculate(LR& roidata)
{
    calcRawMoments (roidata.raw_pixels);
    calcOrigins (roidata.raw_pixels);
    calcCentralMoments (roidata.raw_pixels);
    calcNormCentralMoments (roidata.raw_pixels);
    calcNormRawMoments (roidata.raw_pixels);

    //calcHuInvariants(I);

    //ImageMatrix weighted_im(r.raw_pixels, r.aabb);
    //weighted_im.apply_distance_to_contour_weights(r.raw_pixels, r.contour);

    //const pixData& W = weighted_im.ReadablePixels();
    //calcOrigins(W);
    //calcWeightedSpatialMoments(W);
    //calcWeightedCentralMoments(W);
    //calcWeightedHuInvariants(W);
}

void VolumeMomentsFeature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {} // Not supporting online for image moments

void VolumeMomentsFeature::save_value(std::vector<std::vector<double>>& fvals)
{
    // save raw moments
    fvals[D3_RAW_MOMENT_000][0] = m_000;

    fvals[D3_RAW_MOMENT_010][0] = m_010;
    fvals[D3_RAW_MOMENT_011][0] = m_011;
    fvals[D3_RAW_MOMENT_012][0] = m_012;
    fvals[D3_RAW_MOMENT_013][0] = m_013;

    fvals[D3_RAW_MOMENT_020][0] = m_020;
    fvals[D3_RAW_MOMENT_021][0] = m_021;
    fvals[D3_RAW_MOMENT_022][0] = m_022;
    fvals[D3_RAW_MOMENT_023][0] = m_023;

    fvals[D3_RAW_MOMENT_030][0] = m_030;
    fvals[D3_RAW_MOMENT_031][0] = m_031;
    fvals[D3_RAW_MOMENT_032][0] = m_032;
    fvals[D3_RAW_MOMENT_033][0] = m_033;

    fvals[D3_RAW_MOMENT_100][0] = m_100;
    fvals[D3_RAW_MOMENT_101][0] = m_101;
    fvals[D3_RAW_MOMENT_102][0] = m_102;
    fvals[D3_RAW_MOMENT_103][0] = m_103;

    fvals[D3_RAW_MOMENT_110][0] = m_110;
    fvals[D3_RAW_MOMENT_111][0] = m_111;
    fvals[D3_RAW_MOMENT_112][0] = m_112;
    fvals[D3_RAW_MOMENT_113][0] = m_113;

    fvals[D3_RAW_MOMENT_120][0] = m_120;
    fvals[D3_RAW_MOMENT_121][0] = m_121;
    fvals[D3_RAW_MOMENT_122][0] = m_122;
    fvals[D3_RAW_MOMENT_123][0] = m_123;

    fvals[D3_RAW_MOMENT_200][0] = m_200;
    fvals[D3_RAW_MOMENT_201][0] = m_201;
    fvals[D3_RAW_MOMENT_202][0] = m_202;
    fvals[D3_RAW_MOMENT_203][0] = m_203;

    fvals[D3_RAW_MOMENT_210][0] = m_210;
    fvals[D3_RAW_MOMENT_211][0] = m_211;
    fvals[D3_RAW_MOMENT_212][0] = m_212;
    fvals[D3_RAW_MOMENT_213][0] = m_213;

    fvals[D3_RAW_MOMENT_300][0] = m_300;
    fvals[D3_RAW_MOMENT_301][0] = m_301;
    fvals[D3_RAW_MOMENT_302][0] = m_302;
    fvals[D3_RAW_MOMENT_303][0] = m_303;

    // normalized raw moments
    fvals[D3_NORM_RAW_MOMENT_000][0] = w_000;
    fvals[D3_NORM_RAW_MOMENT_010][0] = w_010;
    fvals[D3_NORM_RAW_MOMENT_011][0] = w_011;
    fvals[D3_NORM_RAW_MOMENT_012][0] = w_012;
    fvals[D3_NORM_RAW_MOMENT_013][0] = w_013;
    fvals[D3_NORM_RAW_MOMENT_020][0] = w_020;
    fvals[D3_NORM_RAW_MOMENT_021][0] = w_021;
    fvals[D3_NORM_RAW_MOMENT_022][0] = w_022;
    fvals[D3_NORM_RAW_MOMENT_023][0] = w_023;
    fvals[D3_NORM_RAW_MOMENT_030][0] = w_030;
    fvals[D3_NORM_RAW_MOMENT_031][0] = w_031;
    fvals[D3_NORM_RAW_MOMENT_032][0] = w_032;
    fvals[D3_NORM_RAW_MOMENT_033][0] = w_033;
    fvals[D3_NORM_RAW_MOMENT_100][0] = w_100;
    fvals[D3_NORM_RAW_MOMENT_101][0] = w_101;
    fvals[D3_NORM_RAW_MOMENT_102][0] = w_102;
    fvals[D3_NORM_RAW_MOMENT_103][0] = w_103;
    fvals[D3_NORM_RAW_MOMENT_200][0] = w_200;
    fvals[D3_NORM_RAW_MOMENT_201][0] = w_201;
    fvals[D3_NORM_RAW_MOMENT_202][0] = w_202;
    fvals[D3_NORM_RAW_MOMENT_203][0] = w_203;
    fvals[D3_NORM_RAW_MOMENT_300][0] = w_300;
    fvals[D3_NORM_RAW_MOMENT_301][0] = w_301;
    fvals[D3_NORM_RAW_MOMENT_302][0] = w_302;
    fvals[D3_NORM_RAW_MOMENT_303][0] = w_303;

    // save central moments
    fvals[D3_CENTRAL_MOMENT_020][0] = mu_020;
    fvals[D3_CENTRAL_MOMENT_021][0] = mu_021 ;
    fvals[D3_CENTRAL_MOMENT_022][0] = mu_022 ;
    fvals[D3_CENTRAL_MOMENT_023][0] = mu_023 ;

    fvals[D3_CENTRAL_MOMENT_030][0] = mu_030 ;
    fvals[D3_CENTRAL_MOMENT_031][0] = mu_031 ;
    fvals[D3_CENTRAL_MOMENT_032][0] = mu_032 ;
    fvals[D3_CENTRAL_MOMENT_033][0] = mu_033 ;

    fvals[D3_CENTRAL_MOMENT_110][0] = mu_110 ;
    fvals[D3_CENTRAL_MOMENT_111][0] = mu_111 ;
    fvals[D3_CENTRAL_MOMENT_112][0] = mu_112 ;
    fvals[D3_CENTRAL_MOMENT_113][0] = mu_113 ;

    fvals[D3_CENTRAL_MOMENT_120][0] = mu_120 ;
    fvals[D3_CENTRAL_MOMENT_121][0] = mu_121 ;
    fvals[D3_CENTRAL_MOMENT_122][0] = mu_122 ;
    fvals[D3_CENTRAL_MOMENT_123][0] = mu_123 ;

    fvals[D3_CENTRAL_MOMENT_200][0] = mu_200 ;
    fvals[D3_CENTRAL_MOMENT_201][0] = mu_201 ;
    fvals[D3_CENTRAL_MOMENT_202][0] = mu_202 ;
    fvals[D3_CENTRAL_MOMENT_203][0] = mu_203 ;

    fvals[D3_CENTRAL_MOMENT_210][0] = mu_210 ;
    fvals[D3_CENTRAL_MOMENT_211][0] = mu_211 ;
    fvals[D3_CENTRAL_MOMENT_212][0] = mu_212 ;
    fvals[D3_CENTRAL_MOMENT_213][0] = mu_213 ;

    fvals[D3_CENTRAL_MOMENT_300][0] = mu_300 ;
    fvals[D3_CENTRAL_MOMENT_301][0] = mu_301 ;
    fvals[D3_CENTRAL_MOMENT_302][0] = mu_302 ;
    fvals[D3_CENTRAL_MOMENT_303][0] = mu_303 ;

    // normalized central moments
    fvals[D3_NORM_CENTRAL_MOMENT_020][0] = nu_020;
    fvals[D3_NORM_CENTRAL_MOMENT_021][0] = nu_021;
    fvals[D3_NORM_CENTRAL_MOMENT_022][0] = nu_022;
    fvals[D3_NORM_CENTRAL_MOMENT_023][0] = nu_023;

    fvals[D3_NORM_CENTRAL_MOMENT_030][0] = nu_030;
    fvals[D3_NORM_CENTRAL_MOMENT_031][0] = nu_031;
    fvals[D3_NORM_CENTRAL_MOMENT_032][0] = nu_032;
    fvals[D3_NORM_CENTRAL_MOMENT_033][0] = nu_033;

    fvals[D3_NORM_CENTRAL_MOMENT_110][0] = nu_110;
    fvals[D3_NORM_CENTRAL_MOMENT_111][0] = nu_111;
    fvals[D3_NORM_CENTRAL_MOMENT_112][0] = nu_112;
    fvals[D3_NORM_CENTRAL_MOMENT_113][0] = nu_113;

    fvals[D3_NORM_CENTRAL_MOMENT_120][0] = nu_120;
    fvals[D3_NORM_CENTRAL_MOMENT_121][0] = nu_121;
    fvals[D3_NORM_CENTRAL_MOMENT_122][0] = nu_122;
    fvals[D3_NORM_CENTRAL_MOMENT_123][0] = nu_123;

    fvals[D3_NORM_CENTRAL_MOMENT_200][0] = nu_200;
    fvals[D3_NORM_CENTRAL_MOMENT_201][0] = nu_201;
    fvals[D3_NORM_CENTRAL_MOMENT_202][0] = nu_202;
    fvals[D3_NORM_CENTRAL_MOMENT_203][0] = nu_203;

    fvals[D3_NORM_CENTRAL_MOMENT_210][0] = nu_210;
    fvals[D3_NORM_CENTRAL_MOMENT_211][0] = nu_211;
    fvals[D3_NORM_CENTRAL_MOMENT_212][0] = nu_212;
    fvals[D3_NORM_CENTRAL_MOMENT_213][0] = nu_213;

    fvals[D3_NORM_CENTRAL_MOMENT_300][0] = nu_300;
    fvals[D3_NORM_CENTRAL_MOMENT_301][0] = nu_301;
    fvals[D3_NORM_CENTRAL_MOMENT_302][0] = nu_302;
    fvals[D3_NORM_CENTRAL_MOMENT_303][0] = nu_303;

    //fvals[WEIGHTED_CENTRAL_MOMENT_02][0] = wmu02;
    //fvals[WEIGHTED_CENTRAL_MOMENT_03][0] = wmu03;
    //fvals[WEIGHTED_CENTRAL_MOMENT_11][0] = wmu11;
    //fvals[WEIGHTED_CENTRAL_MOMENT_12][0] = wmu12;
    //fvals[WEIGHTED_CENTRAL_MOMENT_20][0] = wmu20;
    //fvals[WEIGHTED_CENTRAL_MOMENT_21][0] = wmu21;
    //fvals[WEIGHTED_CENTRAL_MOMENT_30][0] = wmu30;

    //fvals[NORM_CENTRAL_MOMENT_02][0] = nu02;
    //fvals[NORM_CENTRAL_MOMENT_03][0] = nu03;
    //fvals[NORM_CENTRAL_MOMENT_11][0] = nu11;
    //fvals[NORM_CENTRAL_MOMENT_12][0] = nu12;
    //fvals[NORM_CENTRAL_MOMENT_20][0] = nu20;
    //fvals[NORM_CENTRAL_MOMENT_21][0] = nu21;
    //fvals[NORM_CENTRAL_MOMENT_30][0] = nu30;

    //fvals[NORM_SPAT_MOMENT_00][0] = w00;
    //fvals[NORM_SPAT_MOMENT_01][0] = w01;
    //fvals[NORM_SPAT_MOMENT_02][0] = w02;
    //fvals[NORM_SPAT_MOMENT_03][0] = w03;
    //fvals[NORM_SPAT_MOMENT_10][0] = w10;
    //fvals[NORM_SPAT_MOMENT_20][0] = w20;
    //fvals[NORM_SPAT_MOMENT_30][0] = w30;

    //fvals[HU_M1][0] = hm1;
    //fvals[HU_M2][0] = hm2;
    //fvals[HU_M3][0] = hm3;
    //fvals[HU_M4][0] = hm4;
    //fvals[HU_M5][0] = hm5;
    //fvals[HU_M6][0] = hm6;
    //fvals[HU_M7][0] = hm7;

    //fvals[WEIGHTED_HU_M1][0] = whm1;
    //fvals[WEIGHTED_HU_M2][0] = whm2;
    //fvals[WEIGHTED_HU_M3][0] = whm3;
    //fvals[WEIGHTED_HU_M4][0] = whm4;
    //fvals[WEIGHTED_HU_M5][0] = whm5;
    //fvals[WEIGHTED_HU_M6][0] = whm6;
    //fvals[WEIGHTED_HU_M7][0] = whm7;
}

/// @brief Calculates the spatial 3D-moment of order q,p,z
double VolumeMomentsFeature::moment (const std::vector <Pixel2> & cloud, int p, int q, int z)
{
    double q_ = q,
        p_ = p,
        z_ = z, 
        sum = 0;
    for (auto & pxl : cloud)
    {
        double a = pxl.inten;
        sum += a * pow(pxl.x, p_) * pow(pxl.y, q_) * pow(pxl.z, z_);
    }
    return sum;
}

/// @brief Calculates the normalized spatial 3D-moment of order q,p,z
double VolumeMomentsFeature::normRawMom (const std::vector <Pixel2>& cloud, int p, int q, int z)
{
    double stddev = centralMom(cloud, 2,2,2);
    int w = std::max(q, p);
    double normCoef = pow(stddev, (double)w);
    double cmPQ = centralMom (cloud, p,q,z);
    double retval = cmPQ / normCoef;
    return retval;
}

void VolumeMomentsFeature::calcOrigins (const std::vector<Pixel2>& c)
{
    double m00 = moment (c, 0, 0, 0);
    originOfX = moment (c, 1, 0, 0) / m00;
    originOfY = moment (c, 0, 1, 0) / m00;
    originOfZ = moment (c, 0, 0, 1) / m00;
}

/// @brief Calculates the central 3D-moment of order q,p,z
double VolumeMomentsFeature::centralMom (const std::vector<Pixel2>& c, int p, int q, int z)
{
    double sum = 0;
    for (auto & pxl : c)
        sum += pxl.inten * pow((double(pxl.x) - originOfX), p) * pow((double(pxl.y) - originOfY), q) * pow((double(pxl.z) - originOfZ), z);
    return sum;
}

/// @brief Calculates the normalized central 3D-moment of order q,p,z
double VolumeMomentsFeature::normCentralMom (const std::vector<Pixel2>& c, int p, int q, int z)
{
    double temp = ((double(p) + double(q) + double(z)) / 3.0) + 1.0;
    double retval = centralMom(c, p,q,z) / pow(moment(c, 0,0,0), temp);
    return retval;
}

void VolumeMomentsFeature::calcRawMoments (const std::vector<Pixel2>& c)
{
    m_000 = moment(c, 0,0,0);

    m_010 = moment(c, 0,1,0);
    m_011 = moment(c, 0,1,1);
    m_012 = moment(c, 0,1,2);
    m_013 = moment(c, 0,1,3);

    m_020 = moment(c, 0,2,0);
    m_021 = moment(c, 0,2,1);
    m_022 = moment(c, 0,2,2);
    m_023 = moment(c, 0,2,3);

    m_030 = moment(c, 0,3,0);
    m_031 = moment(c, 0,3,1);
    m_032 = moment(c, 0,3,2);
    m_033 = moment(c, 0,3,3);

    m_100 = moment(c, 1,0,0);
    m_101 = moment(c, 1,0,1);
    m_102 = moment(c, 1,0,2);
    m_103 = moment(c, 1,0,3);

    m_110 = moment(c, 1,1,0);
    m_111 = moment(c, 1,1,1);
    m_112 = moment(c, 1,1,2);
    m_113 = moment(c, 1,1,3);

    m_120 = moment(c, 1,2,0);
    m_121 = moment(c, 1,2,1);
    m_122 = moment(c, 1,2,2);
    m_123 = moment(c, 1,2,3);

    m_200 = moment(c, 2,0,0);
    m_201 = moment(c, 2,0,1);
    m_202 = moment(c, 2,0,2);
    m_203 = moment(c, 2,0,3);

    m_210 = moment(c, 2,1,0);
    m_211 = moment(c, 2,1,1);
    m_212 = moment(c, 2,1,2);
    m_213 = moment(c, 2,1,3);

    m_300 = moment(c, 3,0,0);
    m_301 = moment(c, 3,0,1);
    m_302 = moment(c, 3,0,2);
    m_303 = moment(c, 3,0,3);
}

void VolumeMomentsFeature::calcNormRawMoments (const std::vector<Pixel2>& c)
{
    w_000 = normRawMom (c, 0,0,0);
    w_010 = normRawMom (c, 0,1,0); 
    w_011 = normRawMom (c, 0,1,1); 
    w_012 = normRawMom (c, 0,1,2); 
    w_013 = normRawMom (c, 0,1,3);
    w_020 = normRawMom (c, 0,2,0); 
    w_021 = normRawMom (c, 0,2,1); 
    w_022 = normRawMom (c, 0,2,2);
    w_023 = normRawMom (c, 0,2,3);
    w_030 = normRawMom (c, 0,3,0); 
    w_031 = normRawMom (c, 0,3,1); 
    w_032 = normRawMom (c, 0,3,2); 
    w_033 = normRawMom (c, 0,3,3);
    w_100 = normRawMom (c, 1,0,0); 
    w_101 = normRawMom (c, 1,0,1); 
    w_102 = normRawMom (c, 1,0,2); 
    w_103 = normRawMom (c, 1,0,3);
    w_200 = normRawMom (c, 2,0,0); 
    w_201 = normRawMom (c, 2,0,1); 
    w_202 = normRawMom (c, 2,0,2); 
    w_203 = normRawMom (c, 2,0,3);
    w_300 = normRawMom (c, 3,0,0); 
    w_301 = normRawMom (c, 3,0,1); 
    w_302 = normRawMom (c, 3,0,2); 
    w_303 = normRawMom (c, 3,0,3);
}

void VolumeMomentsFeature::calcCentralMoments (const std::vector<Pixel2>& c)
{
    mu_020 = centralMom(c, 0,2,0);
    mu_021 = centralMom(c, 0,2,1);
    mu_022 = centralMom(c, 0,2,2);
    mu_023 = centralMom(c, 0,2,3);
    mu_030 = centralMom(c, 0,3,0);
    mu_031 = centralMom(c, 0,3,1);
    mu_032 = centralMom(c, 0,3,2);
    mu_033 = centralMom(c, 0,3,3);
    mu_110 = centralMom(c, 1,1,0);
    mu_111 = centralMom(c, 1,1,1);
    mu_112 = centralMom(c, 1,1,2);
    mu_113 = centralMom(c, 1,1,3);
    mu_120 = centralMom(c, 1,2,0);
    mu_121 = centralMom(c, 1,2,1);
    mu_122 = centralMom(c, 1,2,2);
    mu_123 = centralMom(c, 1,2,3);
    mu_200 = centralMom(c, 2,0,0);
    mu_201 = centralMom(c, 2,0,1);
    mu_202 = centralMom(c, 2,0,2);
    mu_203 = centralMom(c, 2,0,3);
    mu_210 = centralMom(c, 2,1,0);
    mu_211 = centralMom(c, 2,1,1);
    mu_212 = centralMom(c, 2,1,2);
    mu_213 = centralMom(c, 2,1,3);
    mu_300 = centralMom(c, 3,0,0);
    mu_301 = centralMom(c, 3,0,1);
    mu_302 = centralMom(c, 3,0,2);
    mu_303 = centralMom(c, 3,0,3);
}

void VolumeMomentsFeature::calcNormCentralMoments(const std::vector<Pixel2>& c)
{
    nu_020 = normCentralMom (c, 0,2,0); 
    nu_021 = normCentralMom (c, 0,2,1); 
    nu_022 = normCentralMom (c, 0,2,2); 
    nu_023 = normCentralMom (c, 0,2,3);
    nu_030 = normCentralMom (c, 0,3,0); 
    nu_031 = normCentralMom (c, 0,3,1); 
    nu_032 = normCentralMom (c, 0,3,2); 
    nu_033 = normCentralMom (c, 0,3,3);
    nu_110 = normCentralMom (c, 1,1,0); 
    nu_111 = normCentralMom (c, 1,1,1); 
    nu_112 = normCentralMom (c, 1,1,2); 
    nu_113 = normCentralMom (c, 1,1,3);
    nu_120 = normCentralMom (c, 1,2,0); 
    nu_121 = normCentralMom (c, 1,2,1); 
    nu_122 = normCentralMom (c, 1,2,2); 
    nu_123 = normCentralMom (c, 1,2,3);
    nu_200 = normCentralMom (c, 2,0,0); 
    nu_201 = normCentralMom (c, 2,0,1); 
    nu_202 = normCentralMom (c, 2,0,2); 
    nu_203 = normCentralMom (c, 2,0,3);
    nu_210 = normCentralMom (c, 2,1,0); 
    nu_211 = normCentralMom (c, 2,1,1); 
    nu_212 = normCentralMom (c, 2,1,2); 
    nu_213 = normCentralMom (c, 2,1,3);
    nu_300 = normCentralMom (c, 3,0,0); 
    nu_301 = normCentralMom (c, 3,0,1); 
    nu_302 = normCentralMom (c, 3,0,2); 
    nu_303 = normCentralMom (c, 3,0,3);
}

/// @brief Calculates the features for a subset of ROIs in a thread-safe way with other ROI subsets
/// @param start Start index of the ROI label vector
/// @param end End index of the ROI label vector
/// @param ptrLabels ROI label vector
/// @param ptrLabelData ROI data
void VolumeMomentsFeature::parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        if (r.has_bad_data())
            continue;

        VolumeMomentsFeature imf;
        imf.calculate(r);
        imf.save_value(r.fvals);
    }
}

