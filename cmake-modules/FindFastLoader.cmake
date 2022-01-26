# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


# - Find FastLoader includes and required compiler flags and library dependencies
# Dependencies: C++17 support and threading library

# This file must be on the client project's CMake module path.
# The preferred way is to copy this file to a folder in the project directory
# and add the folder path to the CMAKE_MODULE_PATH variable.
# (see https://gitlab.kitware.com/cmake/community/-/wikis/doc/tutorials/How-To-Find-Libraries)
#
# The FastLoader_CXX_FLAGS should be added to the CMAKE_CXX_FLAGS
#
# This module defines
#  FastLoader_INCLUDE_DIR
#  FastLoader_LIBRARIES
#  FastLoader_CXX_FLAGS
#  FastLoader_FOUND
#

SET(FastLoader_FOUND ON)

# Ensure C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Check FastLoader dependencies
if (FastLoader_FIND_QUIETLY)
	find_package(Hedgehog QUIET)
	if (Hedgehog_FOUND)
		set(FastLoader_LIBRARIES "${FastLoader_LIBRARIES} ${Hedgehog_LIBRARIES}")
		set(FastLoader_CXX_FLAGS "${FastLoader_CXX_FLAGS} ${Hedgehog_CXX_FLAGS}")
		set(FastLoader_INCLUDE_DIR "${FastLoader_INCLUDE_DIR}" "${Hedgehog_INCLUDE_DIR}")
	else ()
		message(STATUS "Hedgehog library has not been found, please install it to use FastLoader.")
		SET(FastLoader_FOUND OFF)
	endif (Hedgehog_FOUND)

	find_package(TIFF QUIET)
	if (TIFF_FOUND)
		set(FastLoader_LIBRARIES "${FastLoader_LIBRARIES} ${TIFF_LIBRARIES}")
		set(FastLoader_CXX_FLAGS "${FastLoader_CXX_FLAGS} ${TIFF_CXX_FLAGS}")
		set(FastLoader_INCLUDE_DIR "${FastLoader_INCLUDE_DIR}" "${TIFF_INCLUDE_DIR}")
	else ()
		message(STATUS "libtiff library has not been found, please install it to use FastLoader.")
		SET(FastLoader_FOUND OFF)
	endif (TIFF_FOUND)
else ()
	find_package(Hedgehog REQUIRED)
	if (Hedgehog_FOUND)
		set(FastLoader_LIBRARIES "${FastLoader_LIBRARIES} ${Hedgehog_LIBRARIES}")
		set(FastLoader_CXX_FLAGS "${FastLoader_CXX_FLAGS} ${Hedgehog_CXX_FLAGS}")
		set(FastLoader_INCLUDE_DIR "${FastLoader_INCLUDE_DIR}" "${Hedgehog_INCLUDE_DIR}")
	else ()
		message(FATAL_ERROR "Hedgehog library has not been found, please install it to use FastLoader.")
		SET(FastLoader_FOUND OFF)
	endif (Hedgehog_FOUND)

	find_package(TIFF REQUIRED)
	if (TIFF_FOUND)
		set(FastLoader_LIBRARIES "${FastLoader_LIBRARIES} ${TIFF_LIBRARIES}")
		set(FastLoader_CXX_FLAGS "${FastLoader_CXX_FLAGS} ${TIFF_CXX_FLAGS}")
		set(FastLoader_INCLUDE_DIR "${FastLoader_INCLUDE_DIR}" "${TIFF_INCLUDE_DIR}")
	else ()
		message(FATAL_ERROR "libtiff library has not been found, please install it to use FastLoader.")
		SET(FastLoader_FOUND OFF)
	endif (TIFF_FOUND)
endif (FastLoader_FIND_QUIETLY)

# Check include files
FIND_PATH(FastLoader_base_INCLUDE_DIR fast_loader/fast_loader.h
		lib/fastloader
		/usr/include
		/usr/local/include
		)

IF (NOT FastLoader_base_INCLUDE_DIR)
	SET(FastLoader_FOUND OFF)
	MESSAGE(STATUS "Could not find FastLoader includes. FastLoader_FOUND now off")
ELSE ()
	set(FastLoader_INCLUDE_DIR "${FastLoader_INCLUDE_DIR}" "${FastLoader_base_INCLUDE_DIR}")
ENDIF ()

IF (FastLoader_FOUND)
	IF (NOT FastLoader_FIND_QUIETLY)
		MESSAGE(STATUS "Found FastLoader include: ${FastLoader_INCLUDE_DIR}")
	ENDIF (NOT FastLoader_FIND_QUIETLY)
ELSE (FastLoader_FOUND)
	IF (FastLoader_FIND_REQUIRED)
		MESSAGE(FATAL_ERROR "Could not find FastLoader header files")
	ENDIF (FastLoader_FIND_REQUIRED)
ENDIF (FastLoader_FOUND)

string(STRIP "${FastLoader_LIBRARIES}" FastLoader_LIBRARIES)
string(STRIP "${FastLoader_CXX_FLAGS}" FastLoader_CXX_FLAGS)


# The customization of include_dir will is an advanced option
MARK_AS_ADVANCED(FastLoader_INCLUDE_DIR)