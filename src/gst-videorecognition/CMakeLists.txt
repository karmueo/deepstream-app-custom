cmake_minimum_required(VERSION 3.12)
project(gst_videorecognition_plugin)

# 1. 基础配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# add_definitions(-DSAVE_IMAGES)

# 3. 查找依赖包
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED 
    gstreamer-1.0 
    gstreamer-base-1.0 
    gstreamer-video-1.0)
find_package(OpenCV)


# 4. 包含目录
include_directories(
    ${GST_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    "/usr/local/cuda/include"
    "/usr/local/cuda/targets/x86_64-linux/include"
    "/opt/nvidia/deepstream/deepstream/sources/includes"
    "/opt/nvidia/deepstream/deepstream/sources/includes/nvdsinferserver"
    "/opt/nvidia/deepstream/deepstream/sources/libs/nvdsinferserver"
    "/opt/nvidia/deepstream/deepstream/sources/libs/nvdsinferserver/tritonserver_include"
    "/opt/tritonclient/include"
)

# 5. 源文件设置
file(GLOB SRCS ${CMAKE_SOURCE_DIR}/*.cpp)
file(GLOB INCS ${CMAKE_SOURCE_DIR}/*.h)

# 6. 库路径设置
link_directories(
    "/usr/local/cuda/lib64"
    "/opt/nvidia/deepstream/deepstream/lib"
)

# 7. 构建共享库
add_library(gst_videorecognition SHARED ${SRCS} ${INCS})

target_compile_definitions(gst_videorecognition PRIVATE 
    DS_VERSION="7.1.0")

MESSAGE(STATUS ${OpenCV_LIBS})

# 8. 链接库
target_link_libraries(gst_videorecognition
    ${GST_LIBRARIES}
    ${OpenCV_LIBS}
    cudart
    dl
    nppc nppig npps nppicc nppidei
    nvdsgst_helper nvdsgst_meta nvds_meta 
    nvbufsurface nvbufsurftransform
    nvinfer
)

# 9. 安装规则
set(GST_PLUGINS_DIR "/opt/nvidia/deepstream/deepstream/lib/gst-plugins")
install(TARGETS gst_videorecognition
    LIBRARY DESTINATION ${GST_PLUGINS_DIR}
    COMPONENT runtime
)

# 10. 调试符号
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(gst_videorecognition PRIVATE -g)
endif()