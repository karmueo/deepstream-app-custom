cmake_minimum_required(VERSION 3.16)
project(gst_mynetwork LANGUAGES CXX)

# 可通过 -DCUDA_VER=... -DNVDS_VERSION=... 覆盖
if(NOT DEFINED CUDA_VER)
  set(CUDA_VER "12.6" CACHE STRING "CUDA version")
endif()
if(NOT DEFINED NVDS_VERSION)
  set(NVDS_VERSION "7.1" CACHE STRING "DeepStream major.minor")
endif()
set(DS_VERSION "${NVDS_VERSION}.0" CACHE STRING "DS_VERSION")

# install dirs (和原 Makefile 保持一致的默认值)
set(LIB_INSTALL_DIR "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib" CACHE PATH "Library install dir")
set(GST_INSTALL_DIR "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/gst-plugins/" CACHE PATH "GStreamer plugin dir")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
find_package(JsonCpp QUIET)
if(NOT JsonCpp_FOUND)
  message(STATUS "JsonCpp not found via CMake config, trying pkg-config...")
  pkg_check_modules(JSONCPP_PKG jsoncpp)
  if(JSONCPP_PKG_FOUND)
    set(JSONCPP_INCLUDE_DIRS ${JSONCPP_PKG_INCLUDE_DIRS})
    set(JSONCPP_LIBRARIES ${JSONCPP_PKG_LIBRARIES})
    message(STATUS "Found jsoncpp via pkg-config: ${JSONCPP_INCLUDE_DIRS} ${JSONCPP_LIBRARIES}")
  else()
    message(STATUS "jsoncpp not found via pkg-config. Will attempt to rely on system include/link names (jsoncpp)")
  endif()
endif()

add_library(gst_mynetwork SHARED gstmynetwork.cpp eo_protocol_parser.cpp)

target_compile_definitions(gst_mynetwork PRIVATE DS_VERSION=\"${DS_VERSION}\")
target_include_directories(gst_mynetwork PRIVATE
  /usr/local/cuda-${CUDA_VER}/include
  /opt/nvidia/deepstream/deepstream/sources/includes
  /usr/include/gstreamer-1.0
  ${GST_INCLUDE_DIRS}
  ${JSONCPP_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}
)
target_link_libraries(gst_mynetwork PRIVATE
  ${GST_LIBRARIES}
  -L/usr/local/cuda-${CUDA_VER}/lib64 -lcudart -ldl
  -lnppc -lnppig -lnpps -lnppicc -lnppidei
  -L${LIB_INSTALL_DIR} -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform
  $<$<BOOL:${JsonCpp_FOUND}>:JsonCpp::JsonCpp>
  $<$<BOOL:${JSONCPP_LIBRARIES}>:${JSONCPP_LIBRARIES}>
  $<$<NOT:$<BOOL:${JsonCpp_FOUND}>>:jsoncpp>
)
# 与原 Makefile 保持 rpath 到 LIB_INSTALL_DIR
target_link_options(gst_mynetwork PRIVATE "-Wl,-rpath,${LIB_INSTALL_DIR}")
target_link_options(gst_mynetwork PRIVATE "-Wl,-no-undefined")
# set_target_properties(gst_mynetwork PROPERTIES OUTPUT_NAME "gst_mynetwork" PREFIX "lib")
set_target_properties(gst_mynetwork
  PROPERTIES
    OUTPUT_NAME "udpmulticast_sink"
    POSITION_INDEPENDENT_CODE ON
    INSTALL_RPATH ${LIB_INSTALL_DIR}
)

# 导出 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 安装目标（需要 sudo）
install(TARGETS gst_mynetwork LIBRARY DESTINATION ${LIB_INSTALL_DIR})
install(TARGETS gst_mynetwork LIBRARY DESTINATION ${GST_INSTALL_DIR})

# 可选构建 receiver 子目录
option(BUILD_EO_RECEIVER "Build EO multicast receiver tool" ON)
if(BUILD_EO_RECEIVER)
  add_subdirectory(receiver)
endif()