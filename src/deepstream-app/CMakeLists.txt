################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

cmake_minimum_required(VERSION 3.16)
project(deepstream-app LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default CUDA version
if(NOT DEFINED CUDA_VER)
    set(CUDA_VER "12.6")
endif()

if("${CUDA_VER}" STREQUAL "")
    message(FATAL_ERROR "CUDA_VER is not set")
endif()

# DeepStream version
set(NVDS_VERSION "8.0")

# Installation directories
if(NOT DEFINED LIB_INSTALL_DIR)
    set(LIB_INSTALL_DIR "/opt/nvidia/deepstream/deepstream/lib/")
endif()

if(NOT DEFINED APP_INSTALL_DIR)
    set(APP_INSTALL_DIR "/opt/nvidia/deepstream/deepstream/bin/")
endif()

# Detect target device
execute_process(
    COMMAND gcc -dumpmachine
    OUTPUT_VARIABLE TARGET_DEVICE_FULL
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX REPLACE "-.*" "" TARGET_DEVICE ${TARGET_DEVICE_FULL})

# Platform specific flags
if(TARGET_DEVICE STREQUAL "aarch64")
    add_compile_definitions(PLATFORM_TEGRA)
endif()

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(X11 REQUIRED x11)
pkg_check_modules(JSON_GLIB REQUIRED json-glib-1.0)
pkg_check_modules(GSTAPP REQUIRED gstreamer-app-1.0)

# Find CUDA using modern CMake approach
enable_language(CUDA)
find_package(CUDAToolkit ${CUDA_VER} REQUIRED)

# Source files
file(GLOB DEEPSTREAM_APP_SOURCES
    "*.c"
    "*.cpp"
)

file(GLOB APPS_COMMON_SOURCES
    "../../apps-common/src/*.c"
)

file(GLOB APPS_COMMON_YAML_SOURCES
    "../../apps-common/src/deepstream-yaml/*.cpp"
)

set(ALL_SOURCES
    ${DEEPSTREAM_APP_SOURCES}
    ${APPS_COMMON_SOURCES}
    ${APPS_COMMON_YAML_SOURCES}
)

# Include directories
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../apps-common/includes
    /opt/nvidia/deepstream/deepstream/sources/includes
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Create executable
add_executable(deepstream-app ${ALL_SOURCES})

# Set include directories
target_include_directories(deepstream-app PRIVATE
    ${INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
    ${X11_INCLUDE_DIRS}
    ${JSON_GLIB_INCLUDE_DIRS}
    ${GSTAPP_INCLUDE_DIRS}
)

# Compile definitions
target_compile_definitions(deepstream-app PRIVATE
    DS_VERSION_MINOR=1
    DS_VERSION_MAJOR=5
)

# Optional JPEG save feature (uncomment if needed)
# target_compile_definitions(deepstream-app PRIVATE ENABLE_JPEG_SAVE)

# Compiler flags
target_compile_options(deepstream-app PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GSTREAMER_VIDEO_CFLAGS_OTHER}
    ${X11_CFLAGS_OTHER}
    ${JSON_GLIB_CFLAGS_OTHER}
    ${GSTAPP_CFLAGS_OTHER}
    -g
)

# Link directories
target_link_directories(deepstream-app PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}
    ${LIB_INSTALL_DIR}
    ${GSTREAMER_LIBRARY_DIRS}
    ${GSTREAMER_VIDEO_LIBRARY_DIRS}
    ${X11_LIBRARY_DIRS}
    ${JSON_GLIB_LIBRARY_DIRS}
    ${GSTAPP_LIBRARY_DIRS}
)

# Link libraries
target_link_libraries(deepstream-app PRIVATE
    # CUDA libraries (using modern CMake targets)
    CUDA::cudart
    CUDA::cuda_driver
    
    # DeepStream libraries
    nvdsgst_meta
    nvds_meta
    nvdsgst_helper
    nvdsgst_customhelper
    nvds_batch_jpegenc
    nvbufsurface
    nvbufsurftransform
    nvdsgst_smartrecord
    nvds_utils
    nvds_msgbroker
    
    # System libraries
    m
    yaml-cpp
    gstrtspserver-1.0
    dl
    
    # Package libraries
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    ${X11_LIBRARIES}
    ${JSON_GLIB_LIBRARIES}
    ${GSTAPP_LIBRARIES}
)

# Set RPATH for runtime library loading
set_target_properties(deepstream-app PROPERTIES
    INSTALL_RPATH ${LIB_INSTALL_DIR}
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Installation
install(TARGETS deepstream-app
    RUNTIME DESTINATION ${APP_INSTALL_DIR}
)

# Custom target for clean (equivalent to 'make clean')
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/deepstream-app
    COMMENT "Cleaning all build artifacts"
)

# Print configuration information
message(STATUS "DeepStream App Configuration:")
message(STATUS "  CUDA Version: ${CUDA_VER}")
message(STATUS "  NVDS Version: ${NVDS_VERSION}")
message(STATUS "  Target Device: ${TARGET_DEVICE}")
message(STATUS "  Library Install Dir: ${LIB_INSTALL_DIR}")
message(STATUS "  App Install Dir: ${APP_INSTALL_DIR}")