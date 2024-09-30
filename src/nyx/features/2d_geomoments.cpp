#include "../featureset.h"

#ifdef USE_GPU
#include "../gpucache.h"
#include "../gpu/geomoments.cuh"
#endif

#include "2d_geomoments.h"


//********************** i-moms

Imoms2D_feature::Imoms2D_feature() : FeatureMethod("Imoms2D")
{
    provide_features(Imoms2D_feature::featureset);
    add_dependencies({ Nyxus::Feature2D::PERIMETER });
}

void Imoms2D_feature::calculate(LR& r)
{
    BasicGeomoms2D::calculate(r, intenAsInten);
}

void Imoms2D_feature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{
    BasicGeomoms2D::osized_add_online_pixel(x, y, intensity);
}

void Imoms2D_feature::osized_calculate(LR& r, ImageLoader& imloader)
{
    BasicGeomoms2D::osized_calculate(r, imloader);
}

void Imoms2D_feature::parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        if (r.has_bad_data())
            continue;

        Imoms2D_feature f;
        f.calculate(r);
        f.save_value(r.fvals);
    }
}

void Imoms2D_feature::save_value(std::vector<std::vector<double>>& fvals)
{
    fvals[(int)Nyxus::Feature2D::IMOM_RM_00][0] = m00;
    fvals[(int)Nyxus::Feature2D::IMOM_RM_01][0] = m01;
    fvals[(int)Nyxus::Feature2D::IMOM_RM_02][0] = m02;
    fvals[(int)Nyxus::Feature2D::IMOM_RM_03][0] = m03;
    fvals[(int)Nyxus::Feature2D::IMOM_RM_10][0] = m10;
    fvals[(int)Nyxus::Feature2D::IMOM_RM_11][0] = m11;
    fvals[(int)Nyxus::Feature2D::IMOM_RM_12][0] = m12;
    fvals[(int)Nyxus::Feature2D::IMOM_RM_13][0] = m13;
    fvals[(int)Nyxus::Feature2D::IMOM_RM_20][0] = m20;
    fvals[(int)Nyxus::Feature2D::IMOM_RM_21][0] = m21;
    fvals[(int)Nyxus::Feature2D::IMOM_RM_22][0] = m22;
    fvals[(int)Nyxus::Feature2D::IMOM_RM_23][0] = m23;
    fvals[(int)Nyxus::Feature2D::IMOM_RM_30][0] = m30;

    fvals[(int)Nyxus::Feature2D::IMOM_WRM_00][0] = wm00;
    fvals[(int)Nyxus::Feature2D::IMOM_WRM_01][0] = wm01;
    fvals[(int)Nyxus::Feature2D::IMOM_WRM_02][0] = wm02;
    fvals[(int)Nyxus::Feature2D::IMOM_WRM_03][0] = wm03;
    fvals[(int)Nyxus::Feature2D::IMOM_WRM_10][0] = wm10;
    fvals[(int)Nyxus::Feature2D::IMOM_WRM_11][0] = wm11;
    fvals[(int)Nyxus::Feature2D::IMOM_WRM_12][0] = wm12;
    fvals[(int)Nyxus::Feature2D::IMOM_WRM_20][0] = wm20;
    fvals[(int)Nyxus::Feature2D::IMOM_WRM_21][0] = wm21;
    fvals[(int)Nyxus::Feature2D::IMOM_WRM_30][0] = wm30;

    fvals[(int)Nyxus::Feature2D::IMOM_CM_00][0] = mu00;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_01][0] = mu01;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_02][0] = mu02;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_03][0] = mu03;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_10][0] = mu10;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_11][0] = mu11;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_12][0] = mu12;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_13][0] = mu13;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_20][0] = mu20;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_21][0] = mu21;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_22][0] = mu22;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_23][0] = mu23;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_30][0] = mu30;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_31][0] = mu31;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_32][0] = mu32;
    fvals[(int)Nyxus::Feature2D::IMOM_CM_33][0] = mu33;

    fvals[(int)Nyxus::Feature2D::IMOM_WCM_02][0] = wmu02;
    fvals[(int)Nyxus::Feature2D::IMOM_WCM_03][0] = wmu03;
    fvals[(int)Nyxus::Feature2D::IMOM_WCM_11][0] = wmu11;
    fvals[(int)Nyxus::Feature2D::IMOM_WCM_12][0] = wmu12;
    fvals[(int)Nyxus::Feature2D::IMOM_WCM_20][0] = wmu20;
    fvals[(int)Nyxus::Feature2D::IMOM_WCM_21][0] = wmu21;
    fvals[(int)Nyxus::Feature2D::IMOM_WCM_30][0] = wmu30;

    fvals[(int)Nyxus::Feature2D::IMOM_NCM_02][0] = nu02;
    fvals[(int)Nyxus::Feature2D::IMOM_NCM_03][0] = nu03;
    fvals[(int)Nyxus::Feature2D::IMOM_NCM_11][0] = nu11;
    fvals[(int)Nyxus::Feature2D::IMOM_NCM_12][0] = nu12;
    fvals[(int)Nyxus::Feature2D::IMOM_NCM_20][0] = nu20;
    fvals[(int)Nyxus::Feature2D::IMOM_NCM_21][0] = nu21;
    fvals[(int)Nyxus::Feature2D::IMOM_NCM_30][0] = nu30;

    fvals[(int)Nyxus::Feature2D::IMOM_NRM_00][0] = w00;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_01][0] = w01;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_02][0] = w02;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_03][0] = w03;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_10][0] = w10;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_11][0] = w11;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_12][0] = w12;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_13][0] = w13;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_20][0] = w20;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_21][0] = w21;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_22][0] = w22;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_23][0] = w23;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_30][0] = w30;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_31][0] = w31;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_32][0] = w32;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_33][0] = w33;

    fvals[(int)Nyxus::Feature2D::IMOM_HU1][0] = hm1;
    fvals[(int)Nyxus::Feature2D::IMOM_HU2][0] = hm2;
    fvals[(int)Nyxus::Feature2D::IMOM_HU3][0] = hm3;
    fvals[(int)Nyxus::Feature2D::IMOM_HU4][0] = hm4;
    fvals[(int)Nyxus::Feature2D::IMOM_HU5][0] = hm5;
    fvals[(int)Nyxus::Feature2D::IMOM_HU6][0] = hm6;
    fvals[(int)Nyxus::Feature2D::IMOM_HU7][0] = hm7;

    fvals[(int)Nyxus::Feature2D::IMOM_WNCM_02][0] = wncm02;
    fvals[(int)Nyxus::Feature2D::IMOM_WNCM_03][0] = wncm03;
    fvals[(int)Nyxus::Feature2D::IMOM_WNCM_11][0] = wncm11;
    fvals[(int)Nyxus::Feature2D::IMOM_WNCM_12][0] = wncm12;
    fvals[(int)Nyxus::Feature2D::IMOM_WNCM_20][0] = wncm20;
    fvals[(int)Nyxus::Feature2D::IMOM_WNCM_21][0] = wncm21;
    fvals[(int)Nyxus::Feature2D::IMOM_WNCM_30][0] = wncm30;

    fvals[(int)Nyxus::Feature2D::IMOM_WHU1][0] = whm1;
    fvals[(int)Nyxus::Feature2D::IMOM_WHU2][0] = whm2;
    fvals[(int)Nyxus::Feature2D::IMOM_WHU3][0] = whm3;
    fvals[(int)Nyxus::Feature2D::IMOM_WHU4][0] = whm4;
    fvals[(int)Nyxus::Feature2D::IMOM_WHU5][0] = whm5;
    fvals[(int)Nyxus::Feature2D::IMOM_WHU6][0] = whm6;
    fvals[(int)Nyxus::Feature2D::IMOM_WHU7][0] = whm7;
}

#ifdef USE_GPU

void Imoms2D_feature::gpu_process_all_rois(
    const std::vector<int>& Labels,
    std::unordered_map <int, LR>& RoiData,
    size_t batch_offset,
    size_t batch_len)
{
    for (auto i = 0; i < batch_len; i++)
    {
        size_t far_i = i + batch_offset;
        auto lab = Labels[far_i];
        LR& r = RoiData[lab];

        // Calculate features        
        Imoms2D_feature f;
        f.calculate_via_gpu(r, i);

        //---delayed until we process all the ROIs on GPU-side--->  imf.save_value (r.fvals);

        // Pull the result from GPU cache and save it
        if (!NyxusGpu::gpu_featurestatebuf.download())
        {
            std::cerr << "error in " << __FILE__ << ":" << __LINE__ << "\n";
            return;
        }

        save_values_from_gpu_buffer (RoiData, Labels, NyxusGpu::gpu_featurestatebuf, batch_offset, batch_len);
    }
}

void Imoms2D_feature::calculate_via_gpu(LR& r, size_t roi_idx)
{
    bool ok = NyxusGpu::GeoMoments2D_calculate(roi_idx, false);
    if (!ok)
        std::cerr << "Geometric moments: error calculating features on GPU\n";
}

void Imoms2D_feature::save_values_from_gpu_buffer(
    std::unordered_map <int, LR>& roidata,
    const std::vector<int>& roilabels,
    const GpuCache<gpureal>& intermediate_already_hostside,
    size_t batch_offset,
    size_t batch_len)
{
    for (size_t i = 0; i < batch_len; i++)
    {
        size_t roiidx = batch_offset + i;
        auto lbl = roilabels[roiidx];
        LR& roi = roidata[lbl];
        auto& fvals = roi.fvals;

        size_t offs = i * GpusideState::__COUNT__;
        const gpureal* ptrBuf = &intermediate_already_hostside.hobuffer[offs];

        fvals[(int)Nyxus::Feature2D::IMOM_RM_00][0] = ptrBuf[GpusideState::RM00];
        fvals[(int)Nyxus::Feature2D::IMOM_RM_01][0] = ptrBuf[GpusideState::RM01];
        fvals[(int)Nyxus::Feature2D::IMOM_RM_02][0] = ptrBuf[GpusideState::RM02];
        fvals[(int)Nyxus::Feature2D::IMOM_RM_03][0] = ptrBuf[GpusideState::RM03];
        fvals[(int)Nyxus::Feature2D::IMOM_RM_10][0] = ptrBuf[GpusideState::RM10];
        fvals[(int)Nyxus::Feature2D::IMOM_RM_11][0] = ptrBuf[GpusideState::RM11];
        fvals[(int)Nyxus::Feature2D::IMOM_RM_12][0] = ptrBuf[GpusideState::RM12];
        fvals[(int)Nyxus::Feature2D::IMOM_RM_13][0] = ptrBuf[GpusideState::RM13];
        fvals[(int)Nyxus::Feature2D::IMOM_RM_20][0] = ptrBuf[GpusideState::RM20];
        fvals[(int)Nyxus::Feature2D::IMOM_RM_21][0] = ptrBuf[GpusideState::RM21];
        fvals[(int)Nyxus::Feature2D::IMOM_RM_22][0] = ptrBuf[GpusideState::RM22];
        fvals[(int)Nyxus::Feature2D::IMOM_RM_23][0] = ptrBuf[GpusideState::RM23];
        fvals[(int)Nyxus::Feature2D::IMOM_RM_30][0] = ptrBuf[GpusideState::RM30];

        fvals[(int)Nyxus::Feature2D::IMOM_CM_00][0] = ptrBuf[GpusideState::CM00];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_01][0] = ptrBuf[GpusideState::CM01];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_02][0] = ptrBuf[GpusideState::CM02];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_03][0] = ptrBuf[GpusideState::CM03];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_10][0] = ptrBuf[GpusideState::CM10];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_11][0] = ptrBuf[GpusideState::CM11];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_12][0] = ptrBuf[GpusideState::CM12];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_13][0] = ptrBuf[GpusideState::CM13];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_20][0] = ptrBuf[GpusideState::CM20];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_21][0] = ptrBuf[GpusideState::CM21];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_22][0] = ptrBuf[GpusideState::CM22];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_23][0] = ptrBuf[GpusideState::CM23];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_30][0] = ptrBuf[GpusideState::CM30];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_31][0] = ptrBuf[GpusideState::CM31];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_32][0] = ptrBuf[GpusideState::CM32];
        fvals[(int)Nyxus::Feature2D::IMOM_CM_33][0] = ptrBuf[GpusideState::CM33];

        fvals[(int)Nyxus::Feature2D::IMOM_NRM_00][0] = ptrBuf[GpusideState::W00];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_01][0] = ptrBuf[GpusideState::W01];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_02][0] = ptrBuf[GpusideState::W02];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_03][0] = ptrBuf[GpusideState::W03];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_10][0] = ptrBuf[GpusideState::W10];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_11][0] = ptrBuf[GpusideState::W11];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_12][0] = ptrBuf[GpusideState::W12];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_13][0] = ptrBuf[GpusideState::W13];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_20][0] = ptrBuf[GpusideState::W20];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_21][0] = ptrBuf[GpusideState::W21];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_22][0] = ptrBuf[GpusideState::W22];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_23][0] = ptrBuf[GpusideState::W23];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_30][0] = ptrBuf[GpusideState::W30];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_31][0] = ptrBuf[GpusideState::W31];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_32][0] = ptrBuf[GpusideState::W32];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_33][0] = ptrBuf[GpusideState::W33];

        fvals[(int)Nyxus::Feature2D::IMOM_NCM_02][0] = ptrBuf[GpusideState::NU02];
        fvals[(int)Nyxus::Feature2D::IMOM_NCM_03][0] = ptrBuf[GpusideState::NU03];
        fvals[(int)Nyxus::Feature2D::IMOM_NCM_11][0] = ptrBuf[GpusideState::NU11];
        fvals[(int)Nyxus::Feature2D::IMOM_NCM_12][0] = ptrBuf[GpusideState::NU12];
        fvals[(int)Nyxus::Feature2D::IMOM_NCM_20][0] = ptrBuf[GpusideState::NU20];
        fvals[(int)Nyxus::Feature2D::IMOM_NCM_21][0] = ptrBuf[GpusideState::NU21];
        fvals[(int)Nyxus::Feature2D::IMOM_NCM_30][0] = ptrBuf[GpusideState::NU30];

        fvals[(int)Nyxus::Feature2D::IMOM_HU1][0] = ptrBuf[GpusideState::H1];
        fvals[(int)Nyxus::Feature2D::IMOM_HU2][0] = ptrBuf[GpusideState::H2];
        fvals[(int)Nyxus::Feature2D::IMOM_HU3][0] = ptrBuf[GpusideState::H3];
        fvals[(int)Nyxus::Feature2D::IMOM_HU4][0] = ptrBuf[GpusideState::H4];
        fvals[(int)Nyxus::Feature2D::IMOM_HU5][0] = ptrBuf[GpusideState::H5];
        fvals[(int)Nyxus::Feature2D::IMOM_HU6][0] = ptrBuf[GpusideState::H6];
        fvals[(int)Nyxus::Feature2D::IMOM_HU7][0] = ptrBuf[GpusideState::H7];

        fvals[(int)Nyxus::Feature2D::IMOM_WRM_00][0] = ptrBuf[GpusideState::WRM00];
        fvals[(int)Nyxus::Feature2D::IMOM_WRM_01][0] = ptrBuf[GpusideState::WRM01];
        fvals[(int)Nyxus::Feature2D::IMOM_WRM_02][0] = ptrBuf[GpusideState::WRM02];
        fvals[(int)Nyxus::Feature2D::IMOM_WRM_03][0] = ptrBuf[GpusideState::WRM03];
        fvals[(int)Nyxus::Feature2D::IMOM_WRM_10][0] = ptrBuf[GpusideState::WRM10];
        fvals[(int)Nyxus::Feature2D::IMOM_WRM_11][0] = ptrBuf[GpusideState::WRM11];
        fvals[(int)Nyxus::Feature2D::IMOM_WRM_12][0] = ptrBuf[GpusideState::WRM12];
        fvals[(int)Nyxus::Feature2D::IMOM_WRM_20][0] = ptrBuf[GpusideState::WRM20];
        fvals[(int)Nyxus::Feature2D::IMOM_WRM_21][0] = ptrBuf[GpusideState::WRM21];
        fvals[(int)Nyxus::Feature2D::IMOM_WRM_30][0] = ptrBuf[GpusideState::WRM30];

        fvals[(int)Nyxus::Feature2D::IMOM_WCM_02][0] = ptrBuf[GpusideState::WCM02];
        fvals[(int)Nyxus::Feature2D::IMOM_WCM_03][0] = ptrBuf[GpusideState::WCM03];
        fvals[(int)Nyxus::Feature2D::IMOM_WCM_11][0] = ptrBuf[GpusideState::WCM11];
        fvals[(int)Nyxus::Feature2D::IMOM_WCM_12][0] = ptrBuf[GpusideState::WCM12];
        fvals[(int)Nyxus::Feature2D::IMOM_WCM_20][0] = ptrBuf[GpusideState::WCM20];
        fvals[(int)Nyxus::Feature2D::IMOM_WCM_21][0] = ptrBuf[GpusideState::WCM21];
        fvals[(int)Nyxus::Feature2D::IMOM_WCM_30][0] = ptrBuf[GpusideState::WCM30];

        fvals[(int)Nyxus::Feature2D::IMOM_WNCM_02][0] = ptrBuf[GpusideState::WNU02];
        fvals[(int)Nyxus::Feature2D::IMOM_WNCM_03][0] = ptrBuf[GpusideState::WNU03];
        fvals[(int)Nyxus::Feature2D::IMOM_WNCM_11][0] = ptrBuf[GpusideState::WNU11];
        fvals[(int)Nyxus::Feature2D::IMOM_WNCM_12][0] = ptrBuf[GpusideState::WNU12];
        fvals[(int)Nyxus::Feature2D::IMOM_WNCM_20][0] = ptrBuf[GpusideState::WNU20];
        fvals[(int)Nyxus::Feature2D::IMOM_WNCM_21][0] = ptrBuf[GpusideState::WNU21];
        fvals[(int)Nyxus::Feature2D::IMOM_WNCM_30][0] = ptrBuf[GpusideState::WNU30];

        fvals[(int)Nyxus::Feature2D::IMOM_WHU1][0] = ptrBuf[GpusideState::WH1];
        fvals[(int)Nyxus::Feature2D::IMOM_WHU2][0] = ptrBuf[GpusideState::WH2];
        fvals[(int)Nyxus::Feature2D::IMOM_WHU3][0] = ptrBuf[GpusideState::WH3];
        fvals[(int)Nyxus::Feature2D::IMOM_WHU4][0] = ptrBuf[GpusideState::WH4];
        fvals[(int)Nyxus::Feature2D::IMOM_WHU5][0] = ptrBuf[GpusideState::WH5];
        fvals[(int)Nyxus::Feature2D::IMOM_WHU6][0] = ptrBuf[GpusideState::WH6];
        fvals[(int)Nyxus::Feature2D::IMOM_WHU7][0] = ptrBuf[GpusideState::WH7];
    }
}

#endif



//********************** s-moms

Smoms2D_feature::Smoms2D_feature() : FeatureMethod("Smoms2D")
{
    provide_features(Smoms2D_feature::featureset);
    add_dependencies({ Nyxus::Feature2D::PERIMETER });
}

void Smoms2D_feature::calculate(LR& r)
{
    BasicGeomoms2D::calculate(r, intenAsShape);
}

void Smoms2D_feature::osized_add_online_pixel(size_t x, size_t y, uint32_t intensity)
{
    BasicGeomoms2D::osized_add_online_pixel(x, y, intensity);
}

void Smoms2D_feature::osized_calculate(LR& r, ImageLoader& imloader)
{
    BasicGeomoms2D::osized_calculate(r, imloader);
}

void Smoms2D_feature::parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        if (r.has_bad_data())
            continue;

        Smoms2D_feature f;
        f.calculate(r);
        f.save_value(r.fvals);
    }
}

void Smoms2D_feature::save_value(std::vector<std::vector<double>>& fvals)
{
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_00][0] = m00;
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_01][0] = m01;
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_02][0] = m02;
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_03][0] = m03;
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_10][0] = m10;
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_11][0] = m11;
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_12][0] = m12;
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_13][0] = m13;
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_20][0] = m20;
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_21][0] = m21;
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_22][0] = m22;
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_23][0] = m23;
    fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_30][0] = m30;

    fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_00][0] = wm00;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_01][0] = wm01;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_02][0] = wm02;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_03][0] = wm03;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_10][0] = wm10;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_11][0] = wm11;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_12][0] = wm12;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_20][0] = wm20;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_21][0] = wm21;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_30][0] = wm30;

    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_00][0] = mu00;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_01][0] = mu01;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_02][0] = mu02;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_03][0] = mu03;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_10][0] = mu10;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_11][0] = mu11;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_12][0] = mu12;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_13][0] = mu13;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_20][0] = mu20;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_21][0] = mu21;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_22][0] = mu22;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_23][0] = mu23;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_30][0] = mu30;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_31][0] = mu31;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_32][0] = mu32;
    fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_33][0] = mu33;

    fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_02][0] = wmu02;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_03][0] = wmu03;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_11][0] = wmu11;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_12][0] = wmu12;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_20][0] = wmu20;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_21][0] = wmu21;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_30][0] = wmu30;

    fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_02][0] = nu02;
    fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_03][0] = nu03;
    fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_11][0] = nu11;
    fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_12][0] = nu12;
    fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_20][0] = nu20;
    fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_21][0] = nu21;
    fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_30][0] = nu30;

    fvals[(int)Nyxus::Feature2D::IMOM_NRM_00][0] = w00;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_01][0] = w01;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_02][0] = w02;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_03][0] = w03;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_10][0] = w10;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_11][0] = w11;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_12][0] = w12;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_13][0] = w13;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_20][0] = w20;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_21][0] = w21;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_22][0] = w22;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_23][0] = w23;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_30][0] = w30;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_31][0] = w31;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_32][0] = w32;
    fvals[(int)Nyxus::Feature2D::IMOM_NRM_33][0] = w33;

    fvals[(int)Nyxus::Feature2D::HU_M1][0] = hm1;
    fvals[(int)Nyxus::Feature2D::HU_M2][0] = hm2;
    fvals[(int)Nyxus::Feature2D::HU_M3][0] = hm3;
    fvals[(int)Nyxus::Feature2D::HU_M4][0] = hm4;
    fvals[(int)Nyxus::Feature2D::HU_M5][0] = hm5;
    fvals[(int)Nyxus::Feature2D::HU_M6][0] = hm6;
    fvals[(int)Nyxus::Feature2D::HU_M7][0] = hm7;

    fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_02][0] = wncm02;
    fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_03][0] = wncm03;
    fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_11][0] = wncm11;
    fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_12][0] = wncm12;
    fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_20][0] = wncm20;
    fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_21][0] = wncm21;
    fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_30][0] = wncm30;

    fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M1][0] = whm1;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M2][0] = whm2;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M3][0] = whm3;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M4][0] = whm4;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M5][0] = whm5;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M6][0] = whm6;
    fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M7][0] = whm7;
}

#ifdef USE_GPU

void Smoms2D_feature::gpu_process_all_rois(
    const std::vector<int>& Labels,
    std::unordered_map <int, LR>& RoiData,
    size_t batch_offset,
    size_t batch_len)
{
    for (auto i = 0; i < batch_len; i++)
    {
        size_t far_i = i + batch_offset;
        auto lab = Labels[far_i];
        LR& r = RoiData[lab];

        // Calculate features        
        Smoms2D_feature f;
        f.calculate_via_gpu(r, i);

        //---delayed until we process all the ROIs on GPU-side--->  imf.save_value (r.fvals);

        // Pull the result from GPU cache and save it
        if (!NyxusGpu::gpu_featurestatebuf.download())
        {
            std::cerr << "error in " << __FILE__ << ":" << __LINE__ << "\n";
            return;
        }

        save_values_from_gpu_buffer(RoiData, Labels, NyxusGpu::gpu_featurestatebuf, batch_offset, batch_len);
    }
}

void Smoms2D_feature::calculate_via_gpu(LR& r, size_t roi_idx)
{
    bool ok = NyxusGpu::GeoMoments2D_calculate(roi_idx, true);
    if (!ok)
        std::cerr << "Geometric moments: error calculating features on GPU\n";
}

void Smoms2D_feature::save_values_from_gpu_buffer(
    std::unordered_map <int, LR>& roidata,
    const std::vector<int>& roilabels,
    const GpuCache<gpureal>& intermediate_already_hostside,
    size_t batch_offset,
    size_t batch_len)
{
    for (size_t i = 0; i < batch_len; i++)
    {
        size_t roiidx = batch_offset + i;
        auto lbl = roilabels[roiidx];
        LR& roi = roidata[lbl];
        auto& fvals = roi.fvals;

        size_t offs = i * GpusideState::__COUNT__;
        const gpureal* ptrBuf = &intermediate_already_hostside.hobuffer[offs];

        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_00][0] = ptrBuf[GpusideState::RM00];
        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_01][0] = ptrBuf[GpusideState::RM01];
        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_02][0] = ptrBuf[GpusideState::RM02];
        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_03][0] = ptrBuf[GpusideState::RM03];
        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_10][0] = ptrBuf[GpusideState::RM10];
        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_11][0] = ptrBuf[GpusideState::RM11];
        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_12][0] = ptrBuf[GpusideState::RM12];
        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_13][0] = ptrBuf[GpusideState::RM13];
        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_20][0] = ptrBuf[GpusideState::RM20];
        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_21][0] = ptrBuf[GpusideState::RM21];
        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_22][0] = ptrBuf[GpusideState::RM22];
        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_23][0] = ptrBuf[GpusideState::RM23];
        fvals[(int)Nyxus::Feature2D::SPAT_MOMENT_30][0] = ptrBuf[GpusideState::RM30];

        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_00][0] = ptrBuf[GpusideState::CM00];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_01][0] = ptrBuf[GpusideState::CM01];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_02][0] = ptrBuf[GpusideState::CM02];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_03][0] = ptrBuf[GpusideState::CM03];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_10][0] = ptrBuf[GpusideState::CM10];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_11][0] = ptrBuf[GpusideState::CM11];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_12][0] = ptrBuf[GpusideState::CM12];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_13][0] = ptrBuf[GpusideState::CM13];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_20][0] = ptrBuf[GpusideState::CM20];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_21][0] = ptrBuf[GpusideState::CM21];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_22][0] = ptrBuf[GpusideState::CM22];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_23][0] = ptrBuf[GpusideState::CM23];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_30][0] = ptrBuf[GpusideState::CM30];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_31][0] = ptrBuf[GpusideState::CM31];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_32][0] = ptrBuf[GpusideState::CM32];
        fvals[(int)Nyxus::Feature2D::CENTRAL_MOMENT_33][0] = ptrBuf[GpusideState::CM33];

        fvals[(int)Nyxus::Feature2D::IMOM_NRM_00][0] = ptrBuf[GpusideState::W00];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_01][0] = ptrBuf[GpusideState::W01];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_02][0] = ptrBuf[GpusideState::W02];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_03][0] = ptrBuf[GpusideState::W03];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_10][0] = ptrBuf[GpusideState::W10];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_11][0] = ptrBuf[GpusideState::W11];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_12][0] = ptrBuf[GpusideState::W12];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_13][0] = ptrBuf[GpusideState::W13];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_20][0] = ptrBuf[GpusideState::W20];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_21][0] = ptrBuf[GpusideState::W21];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_22][0] = ptrBuf[GpusideState::W22];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_23][0] = ptrBuf[GpusideState::W23];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_30][0] = ptrBuf[GpusideState::W30];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_31][0] = ptrBuf[GpusideState::W31];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_32][0] = ptrBuf[GpusideState::W32];
        fvals[(int)Nyxus::Feature2D::IMOM_NRM_33][0] = ptrBuf[GpusideState::W33];

        fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_02][0] = ptrBuf[GpusideState::NU02];
        fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_03][0] = ptrBuf[GpusideState::NU03];
        fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_11][0] = ptrBuf[GpusideState::NU11];
        fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_12][0] = ptrBuf[GpusideState::NU12];
        fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_20][0] = ptrBuf[GpusideState::NU20];
        fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_21][0] = ptrBuf[GpusideState::NU21];
        fvals[(int)Nyxus::Feature2D::NORM_CENTRAL_MOMENT_30][0] = ptrBuf[GpusideState::NU30];

        fvals[(int)Nyxus::Feature2D::HU_M1][0] = ptrBuf[GpusideState::H1];
        fvals[(int)Nyxus::Feature2D::HU_M2][0] = ptrBuf[GpusideState::H2];
        fvals[(int)Nyxus::Feature2D::HU_M3][0] = ptrBuf[GpusideState::H3];
        fvals[(int)Nyxus::Feature2D::HU_M4][0] = ptrBuf[GpusideState::H4];
        fvals[(int)Nyxus::Feature2D::HU_M5][0] = ptrBuf[GpusideState::H5];
        fvals[(int)Nyxus::Feature2D::HU_M6][0] = ptrBuf[GpusideState::H6];
        fvals[(int)Nyxus::Feature2D::HU_M7][0] = ptrBuf[GpusideState::H7];

        fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_00][0] = ptrBuf[GpusideState::WRM00];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_01][0] = ptrBuf[GpusideState::WRM01];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_02][0] = ptrBuf[GpusideState::WRM02];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_03][0] = ptrBuf[GpusideState::WRM03];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_10][0] = ptrBuf[GpusideState::WRM10];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_11][0] = ptrBuf[GpusideState::WRM11];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_12][0] = ptrBuf[GpusideState::WRM12];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_20][0] = ptrBuf[GpusideState::WRM20];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_21][0] = ptrBuf[GpusideState::WRM21];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_30][0] = ptrBuf[GpusideState::WRM30];

        fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_02][0] = ptrBuf[GpusideState::WCM02];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_03][0] = ptrBuf[GpusideState::WCM03];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_11][0] = ptrBuf[GpusideState::WCM11];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_12][0] = ptrBuf[GpusideState::WCM12];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_20][0] = ptrBuf[GpusideState::WCM20];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_21][0] = ptrBuf[GpusideState::WCM21];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_30][0] = ptrBuf[GpusideState::WCM30];

        fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_02][0] = ptrBuf[GpusideState::WNU02];
        fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_03][0] = ptrBuf[GpusideState::WNU03];
        fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_11][0] = ptrBuf[GpusideState::WNU11];
        fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_12][0] = ptrBuf[GpusideState::WNU12];
        fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_20][0] = ptrBuf[GpusideState::WNU20];
        fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_21][0] = ptrBuf[GpusideState::WNU21];
        fvals[(int)Nyxus::Feature2D::WT_NORM_CTR_MOM_30][0] = ptrBuf[GpusideState::WNU30];

        fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M1][0] = ptrBuf[GpusideState::WH1];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M2][0] = ptrBuf[GpusideState::WH2];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M3][0] = ptrBuf[GpusideState::WH3];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M4][0] = ptrBuf[GpusideState::WH4];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M5][0] = ptrBuf[GpusideState::WH5];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M6][0] = ptrBuf[GpusideState::WH6];
        fvals[(int)Nyxus::Feature2D::WEIGHTED_HU_M7][0] = ptrBuf[GpusideState::WH7];
    }
}

#endif


