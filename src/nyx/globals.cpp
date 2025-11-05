#include "environment.h"
#include "globals.h"
#include "helpers/timing.h"

namespace Nyxus
{
	// Nested ROI
	std::unordered_map <std::string, NestableRois> nestedRoiData;

	// Objects that are used by allocateTrivialRoisBuffers() and then by the GPU platform code 
	// to transfer image matrices of all the image's ROIs
	PixIntens* ImageMatrixBuffer = nullptr;	// Solid buffer of all the image matrices in the image
	size_t imageMatrixBufferLen = 0;		// Combined size of all ROIs' image matrices in the image
	size_t largest_roi_imatr_buf_len = 0;

	// Shows a message in CLI or Python terminal 
	void sureprint (const std::string& msg, bool send_to_stderr)
	{
#ifdef WITH_PYTHON_H
		pybind11::print(msg);
#else
		if (send_to_stderr)
			std::cerr << msg;
		else
			std::cout << msg;
#endif
	}
}
