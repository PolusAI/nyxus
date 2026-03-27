#ifdef USE_GPU

#include <unordered_map>
#include "environment.h"
#include "cache.h"
#include "gpu/geomoments.cuh"
#include "helpers/helpers.h"
#include "roi_cache.h"

#define OK(x) if (x == false) \
{ \
	std::cerr << "gpu cache related error in " << __FILE__ << ":" << __LINE__; \
	return false; \
} \

#define OKV(x) if (x == false) \
{ \
	std::cerr << "gpu cache related error in " << __FILE__ << ":" << __LINE__; \
	return; \
} \

template<>
bool GpuCache<Pixel2>::clear()
{
	if (hobuffer)
	{
		std::free(hobuffer); // delete hobuffer;
		hobuffer = nullptr;

		// device side buffer
		if (!GpusideCache::gpu_delete(devbuffer))
			return false;
		devbuffer = nullptr;
	}

	if (ho_lengths)
	{
		std::free(ho_lengths); // delete ho_lengths;
		ho_lengths = nullptr;
	}

	if (ho_offsets)
	{
		std::free(ho_offsets); // delete ho_offsets;
		ho_offsets = nullptr;

		// device side buffer
		if (!GpusideCache::gpu_delete(dev_offsets))
			return false;
		dev_offsets = nullptr;
	}

	return true;
}

template<>
bool GpuCache<gpureal>::clear()
{
	if (hobuffer)
	{
		std::free(hobuffer); // delete hobuffer;
		hobuffer = nullptr;

		// device side buffer
		if (!GpusideCache::gpu_delete(devbuffer))
			return false;
		devbuffer = nullptr;
	}

	if (ho_lengths)
	{
		std::free(ho_lengths); // delete ho_lengths;
		ho_lengths = nullptr;
	}

	if (ho_offsets)
	{
		std::free(ho_offsets); // delete ho_offsets;
		ho_offsets = nullptr;

		// device side buffer
		if (!GpusideCache::gpu_delete(dev_offsets))
			return false;
		dev_offsets = nullptr;
	}

	return true;
}

template<>
bool GpuCache<size_t>::clear()
{
	throw std::runtime_error("unimplemented GpuCache/numeric/clear");
	return true;
}

template<>
bool GpuCache<Pixel2>::alloc (size_t total_len__, size_t num_rois__)
{
	total_len = total_len__;
	num_rois = num_rois__;

	// allocate on device
	size_t szb = total_len * sizeof(devbuffer[0]);
	if (! GpusideCache::allocate_on_device ((void**)&devbuffer, szb))
		return false;

	szb = num_rois * sizeof(dev_offsets[0]);
	if (! GpusideCache::allocate_on_device ((void**)&dev_offsets, szb))
		return false;

	// allocate on host	
	
	hobuffer = (Pixel2*) std::malloc(total_len * sizeof(hobuffer[0])); 
	if (!hobuffer) return false;
	
	ho_lengths = (size_t*) std::malloc(num_rois * sizeof(ho_lengths[0])); 
	if (!ho_lengths) return false;

	ho_offsets = (size_t*) std::malloc(num_rois * sizeof(ho_offsets[0])); 
	
	return true;
}

template<>
size_t GpuCache<Pixel2>::szb(size_t total_len, size_t num_rois)
{
	// see implementation of alloc() !
	size_t s1 = total_len * sizeof(devbuffer[0]);
	size_t s2 = num_rois * sizeof(dev_offsets[0]);
	return s1 + s2;
}

template<>
bool GpuCache<gpureal>::alloc(size_t roi_buf_len, size_t num_rois__)
{
	num_rois = num_rois__;
	total_len = roi_buf_len * num_rois;

	// allocate on host	
	hobuffer = (gpureal*) std::malloc(total_len * sizeof(hobuffer[0])); 

	//		not allocating ho_lengths
	//		not allocating ho_offsets

	// allocate on device
	size_t szb = total_len * sizeof(devbuffer[0]);
	if (!GpusideCache::allocate_on_device((void**)&devbuffer, szb))
		return false;

	//		not allocating dev_offsets

	return true;
}

template<>
size_t GpuCache<gpureal>::szb(size_t total_len, size_t num_rois)
{
	size_t szb = total_len * sizeof(devbuffer[0]);
	return szb;
}

template<>
bool GpuCache<Pixel2>::upload()
{
	size_t szb = total_len * sizeof(devbuffer[0]);
	if (!GpusideCache::upload_on_device((void*)devbuffer, (void*)hobuffer, szb))
		return false;

	szb = num_rois * sizeof(dev_offsets[0]);
	if (!GpusideCache::upload_on_device((void*)dev_offsets, (void*)ho_offsets, szb))
		return false;

	return true;
}

template<>
bool GpuCache<gpureal>::download()
{
	size_t szb = total_len * sizeof(devbuffer[0]);
	if (!GpusideCache::download_on_host ((void*)hobuffer, (void*)devbuffer, szb))
		return false;

	return true;
}

template<>
bool GpuCache<cufftDoubleComplex>::clear()
{
	if (hobuffer)
	{
		std::free(hobuffer); 
		hobuffer = nullptr;

		// device side buffer
		if (!GpusideCache::gpu_delete(devbuffer))
			return false;
		devbuffer = nullptr;
	}

	if (ho_lengths)
	{
		std::free(ho_lengths); 
		ho_lengths = nullptr;
	}

	if (ho_offsets)
	{
		std::free(ho_offsets); 
		ho_offsets = nullptr;

		// device side buffer
		if (!GpusideCache::gpu_delete(dev_offsets))
			return false;
		dev_offsets = nullptr;
	}

	return true;
}

template<>
bool GpuCache<cufftDoubleComplex>::alloc(size_t roi_buf_len, size_t num_rois__)
{
	num_rois = num_rois__;
	total_len = roi_buf_len * num_rois;

	// allocate on host	
	hobuffer = (cufftDoubleComplex*)std::malloc(total_len * sizeof(hobuffer[0])); 

	//		not allocating ho_lengths
	//		not allocating ho_offsets

	// allocate on device
	size_t szb = total_len * sizeof(devbuffer[0]);
	if (!GpusideCache::allocate_on_device((void**)&devbuffer, szb))
		return false;

	//		not allocating dev_offsets

	return true;
}

template<>
bool GpuCache<cufftDoubleComplex>::download()
{
	size_t szb = total_len * sizeof(devbuffer[0]);
	if (!GpusideCache::download_on_host((void*)hobuffer, (void*)devbuffer, szb))
		return false;

	return true;
}

template<>
bool GpuCache<PixIntens>::clear()
{
	if (hobuffer)
	{
		std::free(hobuffer); 
		hobuffer = nullptr;

		// device side buffer
		if (!GpusideCache::gpu_delete(devbuffer))
			return false;
		devbuffer = nullptr;
	}

	if (ho_lengths)
	{
		std::free(ho_lengths); 
		ho_lengths = nullptr;
	}

	if (ho_offsets)
	{
		std::free(ho_offsets); 
		ho_offsets = nullptr;

		// device side buffer
		if (!GpusideCache::gpu_delete(dev_offsets))
			return false;
		dev_offsets = nullptr;
	}

	return true;
}

template<>
bool GpuCache<PixIntens>::alloc(size_t roi_buf_len, size_t num_rois__)
{
	num_rois = num_rois__;
	total_len = roi_buf_len * num_rois;

	// allocate on host	
	hobuffer = (PixIntens*) std::malloc(total_len * sizeof(hobuffer[0])); 

	//		not allocating ho_lengths
	//		not allocating ho_offsets

	// allocate on device
	size_t szb = total_len * sizeof(devbuffer[0]);
	if (!GpusideCache::allocate_on_device((void**)&devbuffer, szb))
		return false;

	return true;
}

//********* float

template<>
bool GpuCache<float>::clear()
{
	if (hobuffer)
	{
		std::free(hobuffer);
		hobuffer = nullptr;

		// device side buffer
		if (!GpusideCache::gpu_delete(devbuffer))
			return false;
		devbuffer = nullptr;
	}

	if (ho_lengths)
	{
		std::free(ho_lengths); 
		ho_lengths = nullptr;
	}

	if (ho_offsets)
	{
		std::free(ho_offsets); 
		ho_offsets = nullptr;

		// device side buffer
		if (!GpusideCache::gpu_delete(dev_offsets))
			return false;
		dev_offsets = nullptr;
	}

	return true;
}

template<>
bool GpuCache<float>::alloc(size_t roi_buf_len, size_t num_rois__)
{
	num_rois = num_rois__;
	total_len = roi_buf_len * num_rois;

	// allocate on host	
	hobuffer = (float*)std::malloc(total_len * sizeof(hobuffer[0])); 

	//		not allocating ho_lengths 
	//		not allocating ho_offsets 

	// allocate on device
	size_t szb = total_len * sizeof(devbuffer[0]);
	if (!GpusideCache::allocate_on_device((void**)&devbuffer, szb))
		return false;

	return true;
}

template<>
bool GpuCache<float>::download()
{
	size_t szb = total_len * sizeof(devbuffer[0]);
	if (!GpusideCache::download_on_host((void*)hobuffer, (void*)devbuffer, szb))
		return false;

	return true;
}

#endif // USE_GPU
