#pragma once

#ifdef __APPLE__
    #define uint64 uint64_hack_
    #define int64 int64_hack_
    #include <tiffio.h>
    #undef uint64
    #undef int64
#else
    #include <tiffio.h>
#endif

namespace Nyxus
{
	/// @brief Closes a TIFF handle that a constructor opened but did not get to own.
	///
	/// A destructor does not run for an object whose constructor threw, so a TIFF* opened at the
	/// top of a loader's constructor is leaked by every throw below it -- and those constructors
	/// throw on exactly the inputs a caller is most likely to hand them: a non-grayscale file, an
	/// unsupported bit depth, a directory that will not read. The loader's own ~Loader() closes
	/// the handle on the paths where construction succeeded, and this closes it on the paths
	/// where it did not.
	///
	/// Declare it directly after the open and dismiss it once the constructor can no longer
	/// throw; the handle is then the loader's to close, as before.
	class TiffHandleGuard
	{
	public:
		explicit TiffHandleGuard (TIFF*& handle) : h_(handle) {}

		~TiffHandleGuard()
		{
			if (armed_ && h_)
			{
				TIFFClose (h_);
				h_ = nullptr;	// so a destructor reached by any other route cannot close it twice
			}
		}

		void dismiss() { armed_ = false; }

		TiffHandleGuard (const TiffHandleGuard&) = delete;
		TiffHandleGuard& operator= (const TiffHandleGuard&) = delete;

	private:
		TIFF*& h_;
		bool armed_ = true;
	};
}
