#include "dataset.h"

void Dataset::update_dataset_props_extrema()
{
	dataset_max_combined_roicloud_len = 0;
	dataset_max_n_rois = 0;
	dataset_max_roi_area = 0;
	dataset_max_roi_w = 0;
	dataset_max_roi_h = 0;
	dataset_max_roi_d = 0;

	for (SlideProps& p : dataset_props)
	{
		size_t sup_s_n = p.n_rois * p.max_roi_area;
		dataset_max_combined_roicloud_len = (std::max) (dataset_max_combined_roicloud_len, sup_s_n);
		dataset_max_n_rois = (std::max) (dataset_max_n_rois, p.n_rois);
		dataset_max_roi_area = (std::max) (dataset_max_roi_area, p.max_roi_area);
		dataset_max_roi_w = (std::max) (dataset_max_roi_w, p.max_roi_w);
		dataset_max_roi_h = (std::max) (dataset_max_roi_h, p.max_roi_h);
		dataset_max_roi_d = (std::max) (dataset_max_roi_d, p.max_roi_d);
	}
}

void Dataset::reset_dataset_props()
{
	dataset_props.clear();
}
