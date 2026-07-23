#include <errno.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <thread>
#include "../environment.h"
#include "../helpers/fsystem.h"
#include "../helpers/helpers.h"
#include "voxel_cloud_nontriv.h"

OutOfRamVoxelCloud::OutOfRamVoxelCloud() {}

OutOfRamVoxelCloud::~OutOfRamVoxelCloud()
{
	if (pF)
	{
		fclose(pF);
		pF = nullptr;
		fs::remove(filepath);
	}
}

void OutOfRamVoxelCloud::check_io_ok() const
{
	if (!pF)
		throw (std::runtime_error("ERROR in OutOfRamVoxelCloud - file might not be open with init()"));
}

Pixel3 OutOfRamVoxelCloud::decode_record (const char* rec) const
{
	Pixel3 vx;
	std::memcpy (&vx.x, rec, sizeof(vx.x)); rec += sizeof(vx.x);
	std::memcpy (&vx.y, rec, sizeof(vx.y)); rec += sizeof(vx.y);
	std::memcpy (&vx.z, rec, sizeof(vx.z)); rec += sizeof(vx.z);
	std::memcpy (&vx.inten, rec, sizeof(vx.inten));
	return vx;
}

void OutOfRamVoxelCloud::init (unsigned int _roi_label, const std::string& name)
{
	n_items = 0;
	slab_first.clear();
	slab_count.clear();
	invalidate_cache();

	auto tid = std::this_thread::get_id();

	std::stringstream ssPath;
	ssPath << Nyxus::get_temp_dir_path() << name << "_roi" << _roi_label << "_tid" << tid << ".vox.nyxus";
	filepath = ssPath.str();

	errno = 0;
	pF = fopen(filepath.c_str(), "w+b");
	if (!pF)
		throw (std::runtime_error("Error creating file " + filepath + " Error code:" + std::to_string(errno)));

	if (std::setvbuf(pF, nullptr, _IOFBF, 32768) != 0)
		std::cout << "setvbuf failed\n";
}

void OutOfRamVoxelCloud::close()
{
	if (pF)
	{
		fclose(pF);
		pF = nullptr;
	}
}

void OutOfRamVoxelCloud::clear()
{
	if (pF)
	{
		close();
		fs::remove (filepath);
		n_items = 0;
	}
	slab_first.clear();
	slab_count.clear();
	invalidate_cache();
}

void OutOfRamVoxelCloud::begin_slab (size_t z)
{
	// Planes arrive in ascending order; grow the index so slab_first[z]/slab_count[z] are addressable
	if (slab_first.size() <= z)
	{
		slab_first.resize (z + 1, n_items);
		slab_count.resize (z + 1, 0);
	}
	slab_first[z] = n_items;
	slab_count[z] = 0;
}

void OutOfRamVoxelCloud::add_voxel (const Pixel3& v)
{
	check_io_ok();

	fwrite ((const void*) &(v.x), sizeof(v.x), 1, pF);
	fwrite ((const void*) &(v.y), sizeof(v.y), 1, pF);
	fwrite ((const void*) &(v.z), sizeof(v.z), 1, pF);
	fwrite ((const void*) &(v.inten), sizeof(v.inten), 1, pF);

	n_items++;
	if (!slab_count.empty())
		slab_count.back()++;

	// A write past the cached block would make it stale; the population phase never interleaves
	// reads, so this just holds the cache empty until the first read.
	invalidate_cache();
}

void OutOfRamVoxelCloud::end_slab (size_t /*z*/) {}

size_t OutOfRamVoxelCloud::size() const
{
	return n_items;
}

Pixel3 OutOfRamVoxelCloud::get_at (size_t idx) const
{
	check_io_ok();

	// Serve from the cached block; on a miss, bulk-read the block containing idx with one fread
	// (sequential consumers then hit the cache BLK_RECORDS times before the next miss).
	if (blk_first < 0 || idx < (size_t) blk_first || idx >= (size_t) blk_first + blk_count)
	{
		size_t start = (idx / BLK_RECORDS) * BLK_RECORDS;
		size_t want = (std::min) (BLK_RECORDS, n_items - start);
		blk_buf.resize (want * item_size);
		fseek (pF, (long) (start * item_size), SEEK_SET);
		size_t got = fread (blk_buf.data(), item_size, want, pF);
		blk_first = (long long) start;
		blk_count = got;
	}
	return decode_record (blk_buf.data() + (idx - (size_t) blk_first) * item_size);
}

void OutOfRamVoxelCloud::read_range (size_t first, size_t count, std::vector<Pixel3>& out) const
{
	check_io_ok();

	out.clear();
	out.reserve (count);
	if (count == 0)
		return;

	// One bulk fread of the whole range, then decode in RAM (a slab is bounded by the plane bbox)
	std::vector<char> buf (count * item_size);
	fseek (pF, (long) (first * item_size), SEEK_SET);
	size_t got = fread (buf.data(), item_size, count, pF);
	for (size_t i = 0; i < got; i++)
		out.push_back (decode_record (buf.data() + i * item_size));
}

void OutOfRamVoxelCloud::read_slab (size_t z, std::vector<Pixel3>& out) const
{
	if (z >= slab_first.size())
	{
		out.clear();
		return;
	}
	read_range (slab_first[z], slab_count[z], out);
}

void OutOfRamVoxelCloud::read_slab_window (size_t z0, size_t z1, std::vector<Pixel3>& out) const
{
	// Contiguous on disk because records are Z-sorted: one range spans planes z0..z1
	if (slab_first.empty() || z0 >= slab_first.size())
	{
		out.clear();
		return;
	}
	size_t zi = (z1 < slab_first.size()) ? z1 : slab_first.size() - 1;
	size_t first = slab_first[z0];
	size_t count = 0;
	for (size_t z = z0; z <= zi; z++)
		count += slab_count[z];
	read_range (first, count, out);
}
