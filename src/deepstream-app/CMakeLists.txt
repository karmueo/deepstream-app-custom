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
project(deepstream_app_custom VERSION 1.0.0 LANGUAGES C CXX)
get_filename_component(REPOSITORY_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# The runtime configuration contains absolute paths below this prefix. Keep the
# CMake install and CPack layouts aligned; DESTDIR remains available for staged
# package verification.
set(RUNTIME_INSTALL_PREFIX "/opt/deepstream-app-custom")
set(CMAKE_INSTALL_PREFIX "${RUNTIME_INSTALL_PREFIX}" CACHE PATH
    "Runtime installation prefix" FORCE)
# 设置 CUDA 编译器路径
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

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
set(NVDS_VERSION "7.1")

# Installation directories
if(NOT DEFINED LIB_INSTALL_DIR)
    set(LIB_INSTALL_DIR "/opt/nvidia/deepstream/deepstream/lib/")
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
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT TARGET_DEVICE STREQUAL "aarch64")
    message(FATAL_ERROR "Release packages are supported only for aarch64 Jetson")
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
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND
   (NOT CUDAToolkit_VERSION_MAJOR EQUAL 12 OR
    NOT CUDAToolkit_VERSION_MINOR EQUAL 6))
    message(FATAL_ERROR "Release packages are supported only with CUDA 12.6")
endif()

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
    ${CMAKE_CURRENT_SOURCE_DIR}/../face_detect_plugin/includes
    ${CMAKE_CURRENT_SOURCE_DIR}/../emotieff_plugin/includes
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
    DS_VERSION_MAJOR=7
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
    -fstack-protector-strong
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Release>:-D_FORTIFY_SOURCE=2>
    "-ffile-prefix-map=${REPOSITORY_ROOT}=."
)

target_link_options(deepstream-app PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
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
    OUTPUT_NAME "deepstream-app-custom.bin"
    POSITION_INDEPENDENT_CODE TRUE
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../gst-plugins;${LIB_INSTALL_DIR}"
    BUILD_RPATH "${LIB_INSTALL_DIR}"
)

# Release builds retain a detached debug file internally and ship a stripped ELF.
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_OBJCOPY AND CMAKE_STRIP)
    add_custom_command(TARGET deepstream-app POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} --only-keep-debug
                $<TARGET_FILE:deepstream-app> $<TARGET_FILE:deepstream-app>.debug
        COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:deepstream-app>
        COMMAND ${CMAKE_OBJCOPY}
                --add-gnu-debuglink=$<TARGET_FILE:deepstream-app>.debug
                $<TARGET_FILE:deepstream-app>
        COMMENT "Separating debug symbols and stripping release executable"
    )
endif()

# Installation/package layout under /opt/deepstream-app-custom.
set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)
install(TARGETS deepstream-app RUNTIME DESTINATION bin)
install(PROGRAMS
        "${REPOSITORY_ROOT}/start_rgb_app.sh"
        "${REPOSITORY_ROOT}/start_rgb_drm_app.sh"
        DESTINATION .)

set(APP_CONFIG_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/configs/yml/app_config.yml")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${APP_CONFIG_SOURCE}")
file(READ "${APP_CONFIG_SOURCE}"
    _app_runtime_config_content)
string(FIND "${_app_runtime_config_content}"
    "config-file: ../../../face_detect_plugin/config_face_detect.yml"
    _face_config_path_offset)
if(_face_config_path_offset EQUAL -1)
    message(FATAL_ERROR "Cannot locate development face-detect config path in app_config.yml")
endif()
string(REPLACE
    "config-file: ../../../face_detect_plugin/config_face_detect.yml"
    "config-file: ${RUNTIME_INSTALL_PREFIX}/face_detect_plugin/config_face_detect.yml"
    _app_runtime_config_content "${_app_runtime_config_content}")
string(FIND "${_app_runtime_config_content}"
    "config-file: ../../../emotieff_plugin/configs/config_emotieff.yml"
    _emotion_config_path_offset)
if(_emotion_config_path_offset EQUAL -1)
    message(FATAL_ERROR "Cannot locate development emotieff config path in app_config.yml")
endif()
string(REPLACE
    "config-file: ../../../emotieff_plugin/configs/config_emotieff.yml"
    "config-file: ${RUNTIME_INSTALL_PREFIX}/emotieff_plugin/configs/config_emotieff.yml"
    _app_runtime_config_content "${_app_runtime_config_content}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/app_config.yml"
    "${_app_runtime_config_content}")
set(APP_SOURCE_CSV "${CMAKE_CURRENT_SOURCE_DIR}/configs/yml/file_sources.csv")
get_filename_component(APP_TEST_VIDEO
    "${CMAKE_CURRENT_SOURCE_DIR}/../../test/test.mp4" ABSOLUTE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${APP_SOURCE_CSV}")
file(READ "${APP_SOURCE_CSV}" _app_runtime_sources_content)
set(_development_test_video_uri "file://../../../../test/test.mp4")
string(FIND "${_app_runtime_sources_content}"
    "${_development_test_video_uri}"
    _test_video_uri_offset)
if(_test_video_uri_offset EQUAL -1)
    message(FATAL_ERROR "Cannot locate development test video URI in file_sources.csv")
endif()
string(REPLACE
    "${_development_test_video_uri}"
    "file://${RUNTIME_INSTALL_PREFIX}/test/test.mp4"
    _app_runtime_sources_content "${_app_runtime_sources_content}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/file_sources.csv"
    "${_app_runtime_sources_content}")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/generated/app_config.yml"
    DESTINATION configs/yml)
install(FILES
    configs/yml/config_infer_primary_yolo_352_rgb.yml
    "${CMAKE_CURRENT_BINARY_DIR}/generated/file_sources.csv"
    DESTINATION configs/yml
)
install(FILES "${APP_TEST_VIDEO}" DESTINATION test)

set(FACE_DETECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../face_detect_plugin")
set(FACE_DETECT_PLUGIN_LIBRARY
    "${FACE_DETECT_ROOT}/build/lib/libgstfacedetect.so"
    CACHE FILEPATH "Built face detection GStreamer plugin to package")
set(FACE_DETECT_MODEL_FILES
    "${FACE_DETECT_ROOT}/model/det_500m.onnx"
    "${FACE_DETECT_ROOT}/model/w600k_mbf.onnx"
    "${FACE_DETECT_ROOT}/model/detector.engine"
    "${FACE_DETECT_ROOT}/model/recognizer.engine"
    "${FACE_DETECT_ROOT}/model/detector.engine.json"
    "${FACE_DETECT_ROOT}/model/recognizer.engine.json")
install(FILES "${FACE_DETECT_PLUGIN_LIBRARY}" DESTINATION gst-plugins)
install(FILES "${FACE_DETECT_ROOT}/config_face_detect.yml"
    DESTINATION face_detect_plugin)
install(FILES "${FACE_DETECT_ROOT}/data/gallery.sqlite"
    DESTINATION face_detect_plugin/data)
install(FILES ${FACE_DETECT_MODEL_FILES}
    DESTINATION face_detect_plugin/model)

set(EMOTIEFF_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../emotieff_plugin")
set(EMOTIEFF_PLUGIN_LIBRARY
    "${EMOTIEFF_ROOT}/build/lib/libgstemotieff.so"
    CACHE FILEPATH "Built EmotiEff GStreamer plugin to package")
set(EMOTIEFF_ENGINE_BUILDER
    "${EMOTIEFF_ROOT}/build/emotieff-build-engine"
    CACHE FILEPATH "Built EmotiEff engine converter to package")
set(EMOTIEFF_MODEL_FILES
    "${EMOTIEFF_ROOT}/model/enet_b0_8_best_vgaf.onnx"
    "${EMOTIEFF_ROOT}/model/mbf_va_mtl.onnx"
    "${EMOTIEFF_ROOT}/model/enet_b0_8_best_vgaf.engine"
    "${EMOTIEFF_ROOT}/model/mbf_va_mtl.engine"
    "${EMOTIEFF_ROOT}/model/enet_b0_8_best_vgaf.engine.json"
    "${EMOTIEFF_ROOT}/model/mbf_va_mtl.engine.json"
    "${EMOTIEFF_ROOT}/model/enet_b0_8_best_vgaf.engine.layers.json"
    "${EMOTIEFF_ROOT}/model/mbf_va_mtl.engine.layers.json")
install(FILES "${EMOTIEFF_PLUGIN_LIBRARY}" DESTINATION gst-plugins)
install(PROGRAMS "${EMOTIEFF_ENGINE_BUILDER}" DESTINATION bin)
install(FILES "${EMOTIEFF_ROOT}/configs/config_emotieff.yml"
    DESTINATION emotieff_plugin/configs)
install(PROGRAMS "${EMOTIEFF_ROOT}/model/prepare_models.sh"
    "${EMOTIEFF_ROOT}/model/convert2trt.sh" DESTINATION emotieff_plugin/model)
install(FILES ${EMOTIEFF_MODEL_FILES} DESTINATION emotieff_plugin/model)
install(FILES "${EMOTIEFF_ROOT}/includes/emotieff_metadata.h"
    DESTINATION include/emotieff_plugin)

install(FILES
    configs/config_preprocess_rgb_352_primary.txt
    configs/labels.txt
    DESTINATION configs
)

# The SOT source configuration is also used by the development installation
# under the DeepStream tree, where its model paths are valid. Generate a
# package-specific copy so the /opt/deepstream-app-custom runtime remains
# self-contained when copied to another Jetson.
set(SOT_CONFIG_SOURCE "${REPOSITORY_ROOT}/sot_plugin/config_sot.yml")
set(SOT_RUNTIME_CONFIG
    "${CMAKE_CURRENT_BINARY_DIR}/generated/config_sot.yml")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
if(NOT EXISTS "${SOT_CONFIG_SOURCE}")
    message(FATAL_ERROR "Missing SOT configuration: ${SOT_CONFIG_SOURCE}")
endif()

file(READ "${SOT_CONFIG_SOURCE}" _sot_runtime_config_content)
set(_sot_legacy_model_dir
    "/opt/nvidia/deepstream/deepstream/sot_plugin/models")
set(_sot_runtime_model_dir "${RUNTIME_INSTALL_PREFIX}/models")
foreach(_sot_engine IN ITEMS
        nanotrack_head_fp16.engine
        nanotrack_backbone_fp16.engine
        nanotrack_backbone_search_fp16.engine)
    string(REPLACE
        "${_sot_legacy_model_dir}/${_sot_engine}"
        "${_sot_runtime_model_dir}/${_sot_engine}"
        _sot_runtime_config_content
        "${_sot_runtime_config_content}")
    string(FIND "${_sot_runtime_config_content}"
        "${_sot_runtime_model_dir}/${_sot_engine}"
        _sot_runtime_path_offset)
    if(_sot_runtime_path_offset EQUAL -1)
        message(FATAL_ERROR
            "Unable to configure packaged SOT engine path for ${_sot_engine}")
    endif()
endforeach()
file(WRITE "${SOT_RUNTIME_CONFIG}" "${_sot_runtime_config_content}")

install(FILES "${SOT_RUNTIME_CONFIG}"
        DESTINATION configs
        RENAME config_sot.yml)
install(FILES
    models/yolo26n_rgb_352_uav_no-p2_b4.engine
    models/yolo26n_rgb_352_uav_no-p2_b4.onnx
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/models/nanotrack_head_fp16.engine"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/models/nanotrack_head.onnx"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/models/nanotrack_backbone_fp16.engine"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/models/nanotrack_backbone.onnx"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/models/nanotrack_backbone_search_fp16.engine"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/models/nanotrack_backbone_search.onnx"
    DESTINATION models
)
install(PROGRAMS models/convert2trt.sh DESTINATION models)

set(PACKAGED_CUSTOM_PREPROCESS_LIBRARY
    "${CMAKE_CURRENT_SOURCE_DIR}/../nvdspreprocess_lib/libcustom2d_preprocess.so"
    CACHE FILEPATH "Built custom DeepStream preprocess library to package")
set(SOT_LIBRARY
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/lib/libsot.so"
    CACHE FILEPATH "Built SOT tracker library to package")
set(YOLO_CUSTOM_LIBRARY
    "${CMAKE_CURRENT_SOURCE_DIR}/../../DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"
    CACHE FILEPATH "Built DeepStream YOLO parser library to package")
set(_RUNTIME_REQUIRED_INPUTS
    "${YOLO_CUSTOM_LIBRARY}"
    "${PACKAGED_CUSTOM_PREPROCESS_LIBRARY}"
    "${SOT_LIBRARY}"
    "${FACE_DETECT_PLUGIN_LIBRARY}"
    "${FACE_DETECT_ROOT}/config_face_detect.yml"
    "${FACE_DETECT_ROOT}/data/gallery.sqlite"
    ${FACE_DETECT_MODEL_FILES}
    "${EMOTIEFF_PLUGIN_LIBRARY}"
    "${EMOTIEFF_ENGINE_BUILDER}"
    "${EMOTIEFF_ROOT}/configs/config_emotieff.yml"
    ${EMOTIEFF_MODEL_FILES}
    "${APP_TEST_VIDEO}"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/yolo26n_rgb_352_uav_no-p2_b4.engine"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/yolo26n_rgb_352_uav_no-p2_b4.onnx"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/convert2trt.sh"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/models/nanotrack_head_fp16.engine"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/models/nanotrack_head.onnx"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/models/nanotrack_backbone_fp16.engine"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/models/nanotrack_backbone.onnx"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/models/nanotrack_backbone_search_fp16.engine"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../sot_plugin/models/nanotrack_backbone_search.onnx"
)
foreach(_runtime_input IN LISTS _RUNTIME_REQUIRED_INPUTS)
    if(NOT EXISTS "${_runtime_input}")
        message(FATAL_ERROR
            "Missing runtime input: ${_runtime_input}. Build all project runtime libraries and models before configuring the main application.")
    endif()
endforeach()

if(NOT CMAKE_READELF)
    find_program(CMAKE_READELF readelf REQUIRED)
endif()
execute_process(
    COMMAND "${CMAKE_READELF}" -d "${SOT_LIBRARY}"
    RESULT_VARIABLE _SOT_READELF_RESULT
    OUTPUT_VARIABLE _SOT_DYNAMIC_SECTION
    ERROR_VARIABLE _SOT_READELF_ERROR
)
if(NOT _SOT_READELF_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Unable to inspect SOT_LIBRARY: ${_SOT_READELF_ERROR}")
endif()
foreach(_opencv_soname IN ITEMS
        "libopencv_core.so.4.5d" "libopencv_imgproc.so.4.5d")
    string(FIND "${_SOT_DYNAMIC_SECTION}" "${_opencv_soname}"
        _opencv_soname_offset)
    if(_opencv_soname_offset EQUAL -1)
        message(FATAL_ERROR
            "SOT_LIBRARY must link against Ubuntu 22.04 ${_opencv_soname}. Rebuild sot_plugin with its default SOT_OPENCV_* settings.")
    endif()
endforeach()

install(FILES
    "${YOLO_CUSTOM_LIBRARY}"
    "${PACKAGED_CUSTOM_PREPROCESS_LIBRARY}"
    DESTINATION lib
)
install(FILES "${SOT_LIBRARY}" DESTINATION lib RENAME libsot.so)
# Libraries above are prebuilt inputs, so CMake's target stripping does not
# apply to them. Strip only their installed Release copies.
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_STRIP)
    install(CODE "
        set(_runtime_root \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\")
        file(GLOB _runtime_libraries
            \"\${_runtime_root}/lib/*.so\"
            \"\${_runtime_root}/gst-plugins/*.so\")
        foreach(_library IN LISTS _runtime_libraries)
            execute_process(
                COMMAND \"${CMAKE_STRIP}\" --strip-unneeded \"\${_library}\"
                COMMAND_ERROR_IS_FATAL ANY)
        endforeach()
    ")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(_DEBIAN_CONTROL_DIR "${CMAKE_CURRENT_BINARY_DIR}/debian-control")
    file(MAKE_DIRECTORY "${_DEBIAN_CONTROL_DIR}")
    file(WRITE "${_DEBIAN_CONTROL_DIR}/postinst"
        "#!/bin/sh\nset -e\n\ninstall -d -m 0755 /etc/deepstream-app-custom\n\nexit 0\n")
    file(WRITE "${_DEBIAN_CONTROL_DIR}/conffiles"
        "${RUNTIME_INSTALL_PREFIX}/configs/yml/app_config.yml\n"
        "${RUNTIME_INSTALL_PREFIX}/configs/yml/config_infer_primary_yolo_352_rgb.yml\n"
        "${RUNTIME_INSTALL_PREFIX}/configs/yml/file_sources.csv\n"
        "${RUNTIME_INSTALL_PREFIX}/face_detect_plugin/config_face_detect.yml\n"
        "${RUNTIME_INSTALL_PREFIX}/emotieff_plugin/configs/config_emotieff.yml\n"
        "${RUNTIME_INSTALL_PREFIX}/configs/config_preprocess_rgb_352_primary.txt\n"
        "${RUNTIME_INSTALL_PREFIX}/configs/config_sot.yml\n")

    set(CPACK_GENERATOR "DEB")
    set(CPACK_PACKAGE_NAME "deepstream-app-custom")
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
    set(CPACK_PACKAGE_CONTACT "DeepStream Application Maintainer")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
        "DeepStream application for Jetson Orin Nano")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
    set(CPACK_DEBIAN_PACKAGE_SECTION "video")
    set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS
        "deepstream-7.1, nvidia-l4t-core (>= 36.4), nvidia-l4t-core (<< 36.5), libjson-glib-1.0-0, libopencv-core4.5d, libopencv-imgproc4.5d")
    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
        "${_DEBIAN_CONTROL_DIR}/postinst"
        "${_DEBIAN_CONTROL_DIR}/conffiles")
    set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${RUNTIME_INSTALL_PREFIX}")
    set(CPACK_STRIP_FILES TRUE)
    include(CPack)
else()
    # A build directory may previously have been configured for Release.
    # Remove stale CPack metadata so a subsequent development configure cannot
    # accidentally reproduce a distributable package.
    file(REMOVE
        "${CMAKE_BINARY_DIR}/CPackConfig.cmake"
        "${CMAKE_BINARY_DIR}/CPackSourceConfig.cmake")
    file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/_CPack_Packages")
endif()

# Custom target for clean (equivalent to 'make clean')
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMAND ${CMAKE_COMMAND} -E remove -f
            ${CMAKE_BINARY_DIR}/deepstream-app-custom.bin.debug
    COMMENT "Cleaning all build artifacts"
)

# Print configuration information
message(STATUS "DeepStream App Configuration:")
message(STATUS "  CUDA Version: ${CUDA_VER}")
message(STATUS "  NVDS Version: ${NVDS_VERSION}")
message(STATUS "  Target Device: ${TARGET_DEVICE}")
message(STATUS "  Library Install Dir: ${LIB_INSTALL_DIR}")
message(STATUS "  Runtime Install Prefix: ${CMAKE_INSTALL_PREFIX}")
