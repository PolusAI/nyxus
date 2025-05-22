#pragma once

#if __has_include(<filesystem>)
	#include <filesystem>
	namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
	#include <experimental/filesystem> 
	namespace fs = std::experimental::filesystem;
#else
	error "Missing the <filesystem> header."
#endif

namespace Nyxus
{
	// returns big file extension e.g. ".nii.gz"
	inline std::string get_big_extension (const std::string & fpath)
	{
		auto smallExt = fs::path(fpath).extension();
		auto st = fs::path(fpath).stem();
		std::string ext = fs::path(st).has_extension() ? fs::path(st).extension().string() + smallExt.string() : smallExt.string();
		return ext;
	}

}