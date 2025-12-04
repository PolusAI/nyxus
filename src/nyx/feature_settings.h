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
	//
	__COUNT__
};

#define STNGS_NAN(obj) (obj[(int)NyxSetting::SOFTNAN].rval)
#define STNGS_TINY(obj) (obj[(int)NyxSetting::TINY].rval)
#define STNGS_NGREYS(obj) (obj[(int)NyxSetting::GREYDEPTH].ival)
#define STNGS_IBSI(obj) (obj[(int)NyxSetting::IBSI].bval)
#define STNGS_USEGPU(obj) (obj[(int)NyxSetting::USEGPU].bval)
#define STNGS_SINGLEROI(obj) (obj[(int)NyxSetting::SINGLEROI].bval)
#define STNGS_PIXELDISTANCE(obj) (obj[(int)NyxSetting::PIXELDISTANCE].ival)
#define STNGS_VERBOSLVL(obj) (obj[(int)NyxSetting::VERBOSLVL].ival)

#define STNGS_GLCM_GREYDEPTH(obj) (obj[(int)NyxSetting::GLCM_GREYDEPTH].ival)

