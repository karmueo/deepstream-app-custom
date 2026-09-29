cmake_minimum_required(VERSION 3.18)
project(face_detect_plugin LANGUAGES CXX)
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
pkg_check_modules(SQLITE REQUIRED IMPORTED_TARGET sqlite3)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)
find_path(TRT_INCLUDE NvInfer.h PATH_SUFFIXES aarch64-linux-gnu REQUIRED)
find_library(TRT_LIBRARY nvinfer REQUIRED)
find_library(TRT_PARSER nvonnxparser REQUIRED)

# The host application and SOT already load the distribution OpenCV 4.5 ABI.
# NVIDIA's headers in /usr/include/opencv4 can be newer than that library.
set(_opencv_prepared_dir "${CMAKE_BINARY_DIR}/opencv45/usr/include/opencv4")
set(_opencv_default_dir "${_opencv_prepared_dir}")
if(EXISTS "/usr/include/opencv4/opencv2/core/version.hpp")
    file(STRINGS "/usr/include/opencv4/opencv2/core/version.hpp" _opencv_system_minor
         REGEX "^#define[ \t]+CV_VERSION_MINOR[ \t]+[0-9]+")
    if(_opencv_system_minor MATCHES "[ \t]5$")
        set(_opencv_default_dir "/usr/include/opencv4")
    endif()
endif()
set(FACE_OPENCV_INCLUDE_DIR "${_opencv_default_dir}" CACHE PATH
    "OpenCV 4.5 headers matching SOT and Ubuntu runtime libraries")
if(FACE_OPENCV_INCLUDE_DIR STREQUAL _opencv_prepared_dir AND
   NOT EXISTS "${FACE_OPENCV_INCLUDE_DIR}/opencv2/core/version.hpp")
    message(STATUS "Preparing OpenCV 4.5 headers in ${CMAKE_BINARY_DIR}")
    execute_process(
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/prepare_opencv_headers.sh"
                "${CMAKE_BINARY_DIR}"
        RESULT_VARIABLE _opencv_prepare_result
        OUTPUT_VARIABLE _opencv_prepare_output
        ERROR_VARIABLE _opencv_prepare_error)
    if(NOT _opencv_prepare_result EQUAL 0)
        message(FATAL_ERROR "Could not prepare OpenCV 4.5 headers: ${_opencv_prepare_error}\n"
                            "Run tools/prepare_opencv_headers.sh or set FACE_OPENCV_INCLUDE_DIR.")
    endif()
    message(STATUS "${_opencv_prepare_output}")
endif()
if(NOT EXISTS "${FACE_OPENCV_INCLUDE_DIR}/opencv2/core/version.hpp")
    message(FATAL_ERROR "OpenCV headers not found at ${FACE_OPENCV_INCLUDE_DIR}")
endif()
foreach(component core imgproc imgcodecs)
    find_library(OPENCV_${component} NAMES libopencv_${component}.so.4.5d
                 PATHS /usr/lib/aarch64-linux-gnu NO_DEFAULT_PATH REQUIRED)
    list(APPEND OPENCV_LIBS ${OPENCV_${component}})
endforeach()
file(STRINGS "${FACE_OPENCV_INCLUDE_DIR}/opencv2/core/version.hpp" OPENCV_MINOR
     REGEX "^#define[ \t]+CV_VERSION_MINOR[ \t]+[0-9]+")
if(NOT OPENCV_MINOR MATCHES "[ \t]5$")
    message(FATAL_ERROR "OpenCV 4.5 headers are required; found ${OPENCV_MINOR} at "
                        "${FACE_OPENCV_INCLUDE_DIR}")
endif()

set(DS_ROOT /opt/nvidia/deepstream/deepstream CACHE PATH "DeepStream root")
set(FACE_INCLUDES includes ${FACE_OPENCV_INCLUDE_DIR} ${TRT_INCLUDE}
    ${DS_ROOT}/sources/includes ${CUDAToolkit_INCLUDE_DIRS})

add_library(face_core STATIC src/math.cpp src/engine.cpp src/gallery.cpp src/config.cpp)
target_include_directories(face_core PUBLIC ${FACE_INCLUDES})
target_compile_options(face_core PRIVATE -Wall -Wextra)
target_link_libraries(face_core PUBLIC ${OPENCV_LIBS} ${TRT_LIBRARY}
    CUDA::cudart PkgConfig::JSON PkgConfig::SQLITE yaml-cpp)

add_library(gstfacedetect SHARED src/gstfacedetect.cpp src/metadata.cpp)
target_include_directories(gstfacedetect PRIVATE ${FACE_INCLUDES})
target_compile_definitions(gstfacedetect PRIVATE PACKAGE="deepstream-app-custom"
    VERSION="1.0" LICENSE="Proprietary" BINARY_PACKAGE="deepstream-app-custom"
    URL="https://example.invalid")
target_link_directories(gstfacedetect PRIVATE ${DS_ROOT}/lib)
target_link_libraries(gstfacedetect PRIVATE face_core PkgConfig::GST
    nvds_meta nvdsgst_meta nvbufsurface nvbufsurftransform)
set_target_properties(gstfacedetect PROPERTIES PREFIX "lib" BUILD_RPATH "${DS_ROOT}/lib")

add_executable(face-build-engines tools/build_engines.cpp)
target_link_libraries(face-build-engines PRIVATE face_core ${TRT_PARSER})
add_executable(face-gallery-import tools/gallery_import.cpp)
target_link_libraries(face-gallery-import PRIVATE face_core)
add_executable(face-image-check tools/image_check.cpp)
target_link_libraries(face-image-check PRIVATE face_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(face-core-tests tests/core_test.cpp)
    target_link_libraries(face-core-tests PRIVATE face_core)
    add_test(NAME face-core-tests COMMAND face-core-tests)
    add_executable(face-meta-tests tests/meta_test.cpp src/metadata.cpp)
    target_include_directories(face-meta-tests PRIVATE ${FACE_INCLUDES})
    target_link_directories(face-meta-tests PRIVATE ${DS_ROOT}/lib)
    target_link_libraries(face-meta-tests PRIVATE PkgConfig::GST nvds_meta)
    add_test(NAME face-meta-tests COMMAND face-meta-tests)
endif()

install(TARGETS gstfacedetect LIBRARY DESTINATION gst-plugins)
install(TARGETS face-build-engines face-gallery-import face-image-check RUNTIME DESTINATION bin)
install(FILES config_face_detect.yml DESTINATION face_detect_plugin)
install(DIRECTORY model/ DESTINATION face_detect_plugin/model
        PATTERN "*.onnx" EXCLUDE PATTERN "*.engine" EXCLUDE PATTERN "*.json" EXCLUDE)
install(DIRECTORY model/ DESTINATION face_detect_plugin/model FILES_MATCHING
        PATTERN "*.onnx" PATTERN "*.engine" PATTERN "*.engine.json")
install(FILES includes/face_metadata.h DESTINATION include/face_detect_plugin)
