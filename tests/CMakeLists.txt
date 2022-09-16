cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

message(STATUS "Build tests")
#==== Compiler Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

option(USEGPU "Use GPU" OFF)
find_package(CUDA)

if(CUDA_FOUND)
	set(USEGPU ON)
	if (CUDA_VERSION_MAJOR STREQUAL "10")
		set(CUDA_ARCH_LIST "35;37;50;72;75")
	elseif (CUDA_VERSION_MAJOR STREQUAL "11")
		if(CUDA_VERSION_MINOR STREQUAL "0")
			set(CUDA_ARCH_LIST "35;37;50;72;75")
		elseif (CUDA_VERSION_MINOR STREQUAL "1" OR CUDA_VERSION_MINOR STREQUAL "2")
			set(CUDA_ARCH_LIST "35;37;50;72;75;80;86")
		endif()
	endif()
else()
	set(USEGPU OFF) 
endif(CUDA_FOUND)


set(TEST_SRC
	testAll.cc
	testDummy.h
	test_pixel_intensity_features.h
	test_data.h
	test_gabor.cc
	test_gabor.h
	test_download_data.cc
	test_download_data.h
	../src/nyx/features/basic_morphology.cpp
	../src/nyx/features/caliper_feret.cpp
	../src/nyx/features/caliper_martin.cpp
	../src/nyx/features/caliper_nassenstein.cpp
	../src/nyx/features/chords.cpp
	../src/nyx/features/chords_nontriv.cpp
	../src/nyx/features/circle.cpp
	../src/nyx/features/contour.cpp
	../src/nyx/features/convex_hull_nontriv.cpp
	../src/nyx/features/ellipse_fitting.cpp
	../src/nyx/features/erosion_pixels.cpp
	../src/nyx/features/euler_number.cpp
	../src/nyx/features/extrema.cpp
	../src/nyx/features/fractal_dim.cpp
	../src/nyx/features/gabor.cpp
	../src/nyx/features/gabor_nontriv.cpp
	../src/nyx/features/geo_len_thickness.cpp
	../src/nyx/features/glcm.cpp
	../src/nyx/features/glcm_nontriv.cpp
	../src/nyx/features/gldm.cpp
	../src/nyx/features/glrlm.cpp
	../src/nyx/features/glszm.cpp
	../src/nyx/features/hexagonality_polygonality.cpp
	../src/nyx/features/image_matrix.cpp
	../src/nyx/features/image_matrix_nontriv.cpp
	../src/nyx/features/image_moments.cpp
	../src/nyx/features/image_moments_nontriv.cpp
	../src/nyx/features/intensity.cpp
	../src/nyx/features/neighbors.cpp
	../src/nyx/features/ngtdm.cpp
	../src/nyx/features/radial_distribution.cpp
	../src/nyx/features/roi_label.cpp
	../src/nyx/features/roi_radius.cpp
	../src/nyx/features/rotation.cpp
	../src/nyx/features/specfunc.cpp
	../src/nyx/features/zernike.cpp
	../src/nyx/features/zernike_nontriv.cpp
	../src/nyx/common_stats.cpp
	../src/nyx/dirs_and_files.cpp
	../src/nyx/environment.cpp
	../src/nyx/environment_basic.cpp
	../src/nyx/feature_method.cpp
	../src/nyx/feature_mgr.cpp
	../src/nyx/feature_mgr_init.cpp
	../src/nyx/features_calc_workflow.cpp
	../src/nyx/featureset.cpp
	../src/nyx/globals.cpp
	../src/nyx/image_loader.cpp
	../src/nyx/output_2_buffer.cpp
	../src/nyx/output_2_csv.cpp
	../src/nyx/parallel.cpp
	../src/nyx/phase1.cpp
	../src/nyx/phase2.cpp
	../src/nyx/phase3.cpp
	../src/nyx/reduce_by_feature.cpp
	../src/nyx/reduce_trivial_rois.cpp
	../src/nyx/roi_cache.cpp
	../src/nyx/roi_cache_basic.cpp
	../src/nyx/scan_fastloader_way.cpp
	../src/nyx/pixel_feed.cpp
)

add_executable(runAllTests ${TEST_SRC})

if(USEGPU)

	set(GPU_SOURCE_FILES
		../src/nyx/gpu/gpu_helpers.cu
		../src/nyx/gpu/image_moments.cu
        ../src/nyx/gpu/gabor.cu
	)

	target_sources(runAllTests PRIVATE ${GPU_SOURCE_FILES})
	
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		target_compile_options(runAllTests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fPIC>)
	endif()
	target_link_libraries(runAllTests PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
	set_target_properties(runAllTests PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	list(APPEND runAllTests_LIBRARIES "stdc++fs" )
endif()

find_package(TIFF REQUIRED)
if (TIFF_FOUND)
	list(APPEND runAllTests_LIBRARIES ${TIFF_LIBRARIES} )
	include_directories (${TIFF_INCLUDE_DIR})
else ()
	message(FATAL_ERROR "libtiff library has not been found, please install it.")
endif (TIFF_FOUND)

target_include_directories (runAllTests PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_directories (runAllTests PUBLIC ${GTEST_LIBRARY_PATH})

target_link_libraries (runAllTests PUBLIC gtest ${runAllTests_LIBRARIES})
