cmake_minimum_required(VERSION 3.18)
project(emotieff_plugin LANGUAGES C CXX)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX /opt/deepstream-app-custom CACHE PATH "Installation root" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
pkg_check_modules(JSON REQUIRED IMPORTED_TARGET json-glib-1.0)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)
find_path(TRT_INCLUDE NvInfer.h PATH_SUFFIXES aarch64-linux-gnu REQUIRED)
find_library(TRT_LIBRARY nvinfer REQUIRED)
find_library(TRT_PARSER nvonnxparser REQUIRED)

set(_opencv_prepared_dir "${CMAKE_BINARY_DIR}/opencv45/usr/include/opencv4")
set(_opencv_default_dir "${_opencv_prepared_dir}")
if(EXISTS "/usr/include/opencv4/opencv2/core/version.hpp")
    file(STRINGS "/usr/include/opencv4/opencv2/core/version.hpp" _opencv_system_minor
         REGEX "^#define[ \t]+CV_VERSION_MINOR[ \t]+[0-9]+")
    if(_opencv_system_minor MATCHES "[ \t]5$")
        set(_opencv_default_dir "/usr/include/opencv4")
    endif()
endif()
set(EMOTIEFF_OPENCV_INCLUDE_DIR "${_opencv_default_dir}" CACHE PATH
    "OpenCV 4.5 headers matching the application ABI")
if(EMOTIEFF_OPENCV_INCLUDE_DIR STREQUAL _opencv_prepared_dir AND
   NOT EXISTS "${EMOTIEFF_OPENCV_INCLUDE_DIR}/opencv2/core/version.hpp")
    execute_process(COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/prepare_opencv_headers.sh"
                            "${CMAKE_BINARY_DIR}"
                    RESULT_VARIABLE _opencv_result ERROR_VARIABLE _opencv_error)
    if(NOT _opencv_result EQUAL 0)
        message(FATAL_ERROR "Could not prepare OpenCV 4.5 headers: ${_opencv_error}")
    endif()
endif()
file(STRINGS "${EMOTIEFF_OPENCV_INCLUDE_DIR}/opencv2/core/version.hpp" OPENCV_MINOR
     REGEX "^#define[ \t]+CV_VERSION_MINOR[ \t]+[0-9]+")
if(NOT OPENCV_MINOR MATCHES "[ \t]5$")
    message(FATAL_ERROR "OpenCV 4.5 headers are required")
endif()
foreach(component core imgproc imgcodecs)
    find_library(OPENCV_${component} NAMES libopencv_${component}.so.4.5d
                 PATHS /usr/lib/aarch64-linux-gnu NO_DEFAULT_PATH REQUIRED)
    list(APPEND OPENCV_LIBS ${OPENCV_${component}})
endforeach()

set(DS_ROOT /opt/nvidia/deepstream/deepstream CACHE PATH "DeepStream root")
set(EMOTIEFF_INCLUDES includes ../face_detect_plugin/includes
    ${EMOTIEFF_OPENCV_INCLUDE_DIR} ${TRT_INCLUDE}
    ${DS_ROOT}/sources/includes ${CUDAToolkit_INCLUDE_DIRS})

add_library(emotieff_core STATIC src/math.cpp src/engine.cpp src/config.cpp)
target_include_directories(emotieff_core PUBLIC ${EMOTIEFF_INCLUDES})
target_compile_options(emotieff_core PRIVATE -Wall -Wextra)
target_link_libraries(emotieff_core PUBLIC ${OPENCV_LIBS} ${TRT_LIBRARY}
    CUDA::cudart PkgConfig::JSON yaml-cpp)

add_library(gstemotieff SHARED src/gstemotieff.cpp src/metadata.cpp)
target_include_directories(gstemotieff PRIVATE ${EMOTIEFF_INCLUDES})
target_compile_definitions(gstemotieff PRIVATE PACKAGE="deepstream-app-custom"
    VERSION="1.0" LICENSE="Proprietary" BINARY_PACKAGE="deepstream-app-custom"
    URL="https://example.invalid")
target_link_directories(gstemotieff PRIVATE ${DS_ROOT}/lib)
target_link_libraries(gstemotieff PRIVATE emotieff_core PkgConfig::GST
    nvds_meta nvdsgst_meta nvbufsurface nvbufsurftransform)
set_target_properties(gstemotieff PROPERTIES BUILD_RPATH "${DS_ROOT}/lib")

add_executable(emotieff-build-engine tools/build_engines.cpp)
target_link_libraries(emotieff-build-engine PRIVATE emotieff_core ${TRT_PARSER})
add_executable(emotieff-image-check tools/image_check.cpp)
target_link_libraries(emotieff-image-check PRIVATE emotieff_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(emotieff-core-tests tests/core_test.cpp)
    target_link_libraries(emotieff-core-tests PRIVATE emotieff_core)
    add_test(NAME emotieff-core-tests COMMAND emotieff-core-tests)
    add_executable(emotieff-meta-tests tests/meta_test.cpp src/metadata.cpp)
    target_include_directories(emotieff-meta-tests PRIVATE ${EMOTIEFF_INCLUDES})
    target_link_directories(emotieff-meta-tests PRIVATE ${DS_ROOT}/lib)
    target_link_libraries(emotieff-meta-tests PRIVATE PkgConfig::GST nvds_meta)
    add_test(NAME emotieff-meta-tests COMMAND emotieff-meta-tests)
    add_executable(emotieff-osd-tests tests/osd_test.cpp
        ../../apps-common/src/emotieff_osd.c)
    target_include_directories(emotieff-osd-tests PRIVATE ${EMOTIEFF_INCLUDES}
        ../../apps-common/includes)
    target_link_directories(emotieff-osd-tests PRIVATE ${DS_ROOT}/lib)
    target_link_libraries(emotieff-osd-tests PRIVATE PkgConfig::GST nvds_meta)
    add_test(NAME emotieff-osd-tests COMMAND emotieff-osd-tests)
endif()

install(TARGETS gstemotieff LIBRARY DESTINATION gst-plugins)
install(TARGETS emotieff-build-engine emotieff-image-check RUNTIME DESTINATION bin)
install(FILES configs/config_emotieff.yml DESTINATION emotieff_plugin/configs)
install(PROGRAMS model/prepare_models.sh model/convert2trt.sh DESTINATION emotieff_plugin/model)
install(FILES
    model/enet_b0_8_best_vgaf.onnx model/mbf_va_mtl.onnx
    model/enet_b0_8_best_vgaf.engine model/enet_b0_8_best_vgaf.engine.json
    model/enet_b0_8_best_vgaf.engine.layers.json
    model/mbf_va_mtl.engine model/mbf_va_mtl.engine.json
    model/mbf_va_mtl.engine.layers.json DESTINATION emotieff_plugin/model)
install(FILES includes/emotieff_metadata.h DESTINATION include/emotieff_plugin)
