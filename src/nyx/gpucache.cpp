#ifdef USE_GPU

#include <unordered_map>
#include "environment.h"
#include "cache.h"
#include "gpu/geomoments.cuh"
#include "helpers/helpers.h"
#include "roi_cache.h"

#ifdef NYXUS_GPU_TEST_HOOKS
bool GpusideCache::test_fail_before_gabor = false;
bool GpusideCache::test_fail_next_host_alloc = false;
#endif

namespace
{
	// A host allocation, which a test build can make fail once (GpusideCache::test_fail_next_host_alloc)
	void* host_alloc (size_t szb)
	{
#ifdef NYXUS_GPU_TEST_HOOKS
		if (GpusideCache::test_fail_next_host_alloc)
		{
			GpusideCache::test_fail_next_host_alloc = false;
			return nullptr;
		}
#endif
		return std::malloc (szb ? szb : 1);	// a null result means a failure, also for 0 bytes
	}

	// Frees each buffer of any GpuCache that is set, on its own, and nulls it, going on after a failed
	// free: an allocation that failed part way can leave a device buffer set while its host
	// counterpart is not, or the other way round. Leaves the cache empty, with no capacity.
	template <class T>
	bool clear_cache (GpuCache<T>& c)
	{
		std::free(c.hobuffer);
		c.hobuffer = nullptr;
		std::free(c.ho_lengths);
		c.ho_lengths = nullptr;
		std::free(c.ho_offsets);
		c.ho_offsets = nullptr;
		c.total_len = c.num_rois = 0;	// nothing is held: a capacity check refuses any item

		bool ok = true;
		if (c.devbuffer)
		{
			ok = GpusideCache::gpu_delete(c.devbuffer) && ok;
			c.devbuffer = nullptr;
		}
		if (c.dev_offsets)
		{
			ok = GpusideCache::gpu_delete(c.dev_offsets) && ok;
			c.dev_offsets = nullptr;
		}
		return ok;
	}

	// The layout GpuCache<Pixel2> and GpuCache<size_t> share: a solid buffer of 'total_len' items on
	// the host and the device, and per ROI a length and an offset into it, the offsets mirrored on the
	// device. A failure frees what was allocated.
	template <class T>
	bool alloc_with_offsets (GpuCache<T>& c, size_t total_len, size_t num_rois)
	{
		c.total_len = total_len;
		c.num_rois = num_rois;
		c.hobuffer = (T*) host_alloc(total_len * sizeof(T));
		c.ho_lengths = (size_t*) host_alloc(num_rois * sizeof(size_t));
		c.ho_offsets = (size_t*) host_alloc(num_rois * sizeof(size_t));
		if (c.hobuffer && c.ho_lengths && c.ho_offsets
			&& GpusideCache::allocate_on_device((void**)&c.devbuffer, total_len * sizeof(T))
			&& GpusideCache::allocate_on_device((void**)&c.dev_offsets, num_rois * sizeof(size_t)))
			return true;
		clear_cache(c);
		return false;
	}

	// The layout of the other caches: a solid buffer of roi_buf_len items per ROI on the host and the
	// device, with no per-ROI lengths or offsets. A failure frees what was allocated.
	template <class T>
	bool alloc_plain (GpuCache<T>& c, size_t roi_buf_len, size_t num_rois)
	{
		c.num_rois = num_rois;
		c.total_len = roi_buf_len * num_rois;
		c.hobuffer = (T*) host_alloc(c.total_len * sizeof(T));
		if (c.hobuffer && GpusideCache::allocate_on_device((void**)&c.devbuffer, c.total_len * sizeof(T)))
			return true;
		clear_cache(c);
		return false;
	}

	template <class T>
	size_t szb_with_offsets (size_t total_len, size_t num_rois)
	{
		// see alloc_with_offsets() !
		return total_len * sizeof(T) + num_rois * sizeof(size_t);
	}

	// uploads the first n_items of the buffer (at most its total_len) and every ROI's offset
	template <class T>
	bool upload_with_offsets (GpuCache<T>& c, size_t n_items)
	{
		if (n_items > c.total_len)
			return false;
		if (n_items && !GpusideCache::upload_on_device((void*)c.devbuffer, (void*)c.hobuffer, n_items * sizeof(T)))
			return false;
		if (!GpusideCache::upload_on_device((void*)c.dev_offsets, (void*)c.ho_offsets, c.num_rois * sizeof(size_t)))
			return false;
		return true;
	}
}

template<>
bool GpuCache<Pixel2>::clear()
{
	return clear_cache(*this);
}

template<>
bool GpuCache<gpureal>::clear()
{
	return clear_cache(*this);
}

template<>
bool GpuCache<size_t>::clear()
{
	return clear_cache(*this);
}

template<>
bool GpuCache<size_t>::alloc (size_t total_len__, size_t num_rois__)
{
	return alloc_with_offsets(*this, total_len__, num_rois__);
}

template<>
size_t GpuCache<size_t>::szb (size_t total_len, size_t num_rois)
{
	return szb_with_offsets<size_t>(total_len, num_rois);
}

template<>
bool GpuCache<size_t>::upload()
{
	return upload_with_offsets(*this, total_len);
}

template<>
bool GpuCache<size_t>::upload (size_t n_items)
{
	return upload_with_offsets(*this, n_items);
}

template<>
bool GpuCache<Pixel2>::alloc (size_t total_len__, size_t num_rois__)
{
	return alloc_with_offsets(*this, total_len__, num_rois__);
}

template<>
size_t GpuCache<Pixel2>::szb(size_t total_len, size_t num_rois)
{
	return szb_with_offsets<Pixel2>(total_len, num_rois);
}

template<>
bool GpuCache<gpureal>::alloc (size_t roi_buf_len, size_t num_rois__)
{
	return alloc_plain(*this, roi_buf_len, num_rois__);
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
	return upload_with_offsets(*this, total_len);
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
	return clear_cache(*this);
}

template<>
bool GpuCache<cufftDoubleComplex>::alloc (size_t roi_buf_len, size_t num_rois__)
{
	return alloc_plain(*this, roi_buf_len, num_rois__);
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
	return clear_cache(*this);
}

template<>
bool GpuCache<PixIntens>::alloc (size_t roi_buf_len, size_t num_rois__)
{
	return alloc_plain(*this, roi_buf_len, num_rois__);
}

//********* float

template<>
bool GpuCache<float>::clear()
{
	return clear_cache(*this);
}

template<>
bool GpuCache<float>::alloc (size_t roi_buf_len, size_t num_rois__)
{
	return alloc_plain(*this, roi_buf_len, num_rois__);
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
