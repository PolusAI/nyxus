#pragma once

// Shared fixture for the 3D GLRLM tests: the settings a 3D GLRLM assertion runs on.

#include "test_main_nyxus.h"                  // Fsettings, NyxSetting and run_3d_feature_on_volume, with the roi_cache graph

// D3_GLRLM_feature::calculate() reads SOFTNAN, IBSI and GLRLM_GREYDEPTH; 'grey_depth' is inert for
// this family and passed only so a caller can state the profile it reproduces. 'glrlm_grey_depth'
// selects the binning by its sign: negative is a PyRadiomics-style bin count, positive a MATLAB-style
// level count, and 0 no binning at all.
static Fsettings make_glrlm3d_settings (int grey_depth, int glrlm_grey_depth)
{
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = grey_depth;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GLRLM_GREYDEPTH].ival = glrlm_grey_depth;
    return s;
}
