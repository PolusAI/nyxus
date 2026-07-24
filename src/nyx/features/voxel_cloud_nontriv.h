#pragma once

#include <string>
#include <vector>
#include "pixel.h"

/// @brief Writeable out-of-memory 3D voxel cloud.
///
/// The 3D analog of OutOfRamPixelCloud: it stores Pixel3 records (x,y,z,inten) on disk
/// so an oversized volumetric ROI can be featurized without holding the whole voxel cube
/// in RAM. Records are appended in ascending-Z order by the streaming population scan, so
/// they land on disk already Z-sorted; a small in-RAM per-plane index (first-record offset
/// and count per Z) lets a consumer seek to any Z-slab and read just that slab's voxels.
/// The whole-cloud linear accessors (size/get_at/iterator) serve the Z-agnostic intensity
/// and histogram consumers; the slab accessors serve the sliding-window consumers (surface,
/// and later the 3D texture features).
class OutOfRamVoxelCloud
{
public:
	OutOfRamVoxelCloud();
	OutOfRamVoxelCloud (const OutOfRamVoxelCloud&) = delete;
	~OutOfRamVoxelCloud();

	// Lifecycle. init() creates the backing file under get_temp_dir_path().
	void init (unsigned int roi_label, const std::string& name);
	void close();
	void clear();

	// Population (called in ascending-Z order). begin_slab(z) marks where plane z's records
	// start; add_voxel() appends one record; end_slab(z) finalizes the plane's count.
	void begin_slab (size_t z);
	void add_voxel (const Pixel3& v);
	void end_slab (size_t z);

	// Whole-cloud linear access (intensity, histogram)
	size_t size() const;
	Pixel3 get_at (size_t idx) const;
	Pixel3 operator[] (size_t idx) const { return get_at(idx); }

	// Slab access (surface, later texture): a Z-plane or a [z0,z1] window
	size_t depth() const { return slab_first.size(); }
	void read_slab (size_t z, std::vector<Pixel3>& out) const;
	void read_slab_window (size_t z0, size_t z1, std::vector<Pixel3>& out) const;

	struct iterator
	{
	public:
		iterator (const OutOfRamVoxelCloud& obj, std::size_t idx) : m_object(obj), m_index(idx) {}
		Pixel3 operator * () const { return m_object.get_at(m_index); }
		bool operator == (iterator const& it) const { return (&m_object == &it.m_object) && (m_index == it.m_index); }
		bool operator != (iterator const& it) const { return (&m_object != &it.m_object) || (m_index != it.m_index); }
		iterator& operator ++ () { ++m_index; return *this; }
	private:
		const OutOfRamVoxelCloud& m_object;
		std::size_t m_index;
	};
	iterator begin() const { return iterator(*this, 0); }
	iterator end() const { return iterator(*this, size()); }

private:
	size_t n_items = 0;
	std::string filepath;
	FILE* pF = nullptr;
	// One on-disk record = x,y,z (Pixel3's Point3i fields) + intensity
	size_t item_size = sizeof(Pixel3::x) + sizeof(Pixel3::y) + sizeof(Pixel3::z) + sizeof(Pixel3::inten);
	// Per-Z index built during population: first record index and record count of each plane
	std::vector<size_t> slab_first, slab_count;
	void check_io_ok() const;
	void read_range (size_t first, size_t count, std::vector<Pixel3>& out) const;

	// Block read cache so linear consumers (get_at/operator[]/iterator, histogram) do one bulk
	// fread per block instead of an fseek + 4 freads per voxel -- the latter made the out-of-core
	// intensity/histogram passes ~25x slower than the in-RAM path. One block resident (bounded).
	// constexpr (implicitly inline in C++17+) so ODR-uses like (std::min)(BLK_RECORDS, ...),
	// which bind it to a const&, don't need a separate out-of-line definition -- GCC errors
	// on the missing definition at link time where MSVC silently tolerates it.
	static constexpr size_t BLK_RECORDS = 65536;
	mutable std::vector<char> blk_buf;
	mutable long long blk_first = -1;	// index of first cached record; -1 = cache empty
	mutable size_t blk_count = 0;
	void invalidate_cache() const { blk_first = -1; blk_count = 0; }
	Pixel3 decode_record (const char* rec) const;
};
