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


