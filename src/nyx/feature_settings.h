#pragma once

#include <vector>

// feature settings
union FeatureSetting
{
	bool bval;
	int ival;
	double rval;
};

typedef std::vector<FeatureSetting> Fsettings;

// Settings that each feature faily may consume

enum class NyxSetting : int 
{
	SOFTNAN = 0,
	TINY,
	SINGLEROI,
	GREYDEPTH,
	PIXELSIZEUM,
	PIXELDISTANCE,
	XYRES,
	USEGPU,
	VERBOSLVL,
	IBSI,
	// GLCM
	GLCM_GREYDEPTH,
	GLCM_OFFSET,
	GLCM_NUMANG,
	GLCM_SPARSEINTENS,
	// GLDM
	GLDM_GREYDEPTH,
	// NGTDM
	NGTDM_GREYDEPTH,
	NGTDM_RADIUS,
	// GLRLM
	GLRLM_GREYDEPTH,
	// GLSZM
	GLSZM_GREYDEPTH,
	// Floating-point image quantization params (--fpimg*). Used by the IH family to
	// undo the load-time float->uint quantization and bin in the original float domain.
	// These MIRROR image_loader.cpp: when fp options are active the loader rescales over
	// [FPIMG_MIN, FPIMG_MAX] with target dynamic range FPIMG_TARGET_DR; otherwise it uses
	// the per-slide pre-ROI [min,max].
	FPIMG_TARGET_DR,
	FPIMG_MIN,
	FPIMG_MAX,
	FPIMG_ACTIVE,	// bool: fp options were supplied (loader uses FPIMG_MIN/MAX, not pre-ROI range)
	//
	__COUNT__
};

// common settings
#define STNGS_MISSING(obj) (obj.size() < (int)NyxSetting::__COUNT__)
#define STNGS_NAN(obj) (obj[(int)NyxSetting::SOFTNAN].rval)
#define STNGS_TINY(obj) (obj[(int)NyxSetting::TINY].rval)
#define STNGS_NGREYS(obj) (obj[(int)NyxSetting::GREYDEPTH].ival)
#define STNGS_IBSI(obj) (obj[(int)NyxSetting::IBSI].bval)
#define STNGS_USEGPU(obj) (obj[(int)NyxSetting::USEGPU].bval)
#define STNGS_SINGLEROI(obj) (obj[(int)NyxSetting::SINGLEROI].bval)
#define STNGS_PIXELDISTANCE(obj) (obj[(int)NyxSetting::PIXELDISTANCE].ival)
#define STNGS_VERBOSLVL(obj) (obj[(int)NyxSetting::VERBOSLVL].ival)
#define STNGS_FPIMG_DR(obj) (obj[(int)NyxSetting::FPIMG_TARGET_DR].rval)
#define STNGS_FPIMG_MIN(obj) (obj[(int)NyxSetting::FPIMG_MIN].rval)
#define STNGS_FPIMG_MAX(obj) (obj[(int)NyxSetting::FPIMG_MAX].rval)
#define STNGS_FPIMG_ACTIVE(obj) (obj[(int)NyxSetting::FPIMG_ACTIVE].bval)

// feature-specific settings
#define STNGS_GLCM_GREYDEPTH(obj) (obj[(int)NyxSetting::GLCM_GREYDEPTH].ival)
#define STNGS_GLCM_OFFSET(obj) (obj[(int)NyxSetting::GLCM_OFFSET].ival)
#define STNGS_GLCM_NUMANG(obj) (obj[(int)NyxSetting::GLCM_NUMANG].ival)
#define STNGS_GLCM_SPARSEINTENS(obj) (obj[(int)NyxSetting::GLCM_SPARSEINTENS].ival)
#define STNGS_GLDM_GREYDEPTH(obj) (obj[(int)NyxSetting::GLDM_GREYDEPTH].ival)
#define STNGS_GLRLM_GREYDEPTH(obj) (obj[(int)NyxSetting::GLRLM_GREYDEPTH].ival)
#define STNGS_GLSZM_GREYDEPTH(obj) (obj[(int)NyxSetting::GLSZM_GREYDEPTH].ival)
#define STNGS_NGTDM_GREYDEPTH(obj) (obj[(int)NyxSetting::NGTDM_GREYDEPTH].ival)
#define STNGS_NGTDM_RADIUS(obj) (obj[(int)NyxSetting::NGTDM_RADIUS].ival)


