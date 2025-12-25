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
include(CheckCXXSourceCompiles)


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

# 通过 __aarch64__ 宏判断 Jetson，使用 NVBUF_MEM_SURFACE_ARRAY
check_cxx_source_compiles("
#ifdef __aarch64__
int main() { return 0; }
#else
#error Not aarch64
#endif
" HAS_AARCH64)
if(HAS_AARCH64)
    target_compile_definitions(gst_videorecognition PRIVATE USE_NVBUF_MEM_SURFACE_ARRAY=1)
endif()

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

# 安装 models 目录到 deepstream-app-custom 下的 gst-videorecognition/models
set(GST_VIDEORECOGNITION_MODELS_DEST "/opt/nvidia/deepstream/deepstream/deepstream-app-custom/gst-videorecognition/models" CACHE PATH "Destination to install gst_videorecognition models")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
        message(STATUS "Installing gst_videorecognition models symlink from ${CMAKE_CURRENT_SOURCE_DIR}/models to ${GST_VIDEORECOGNITION_MODELS_DEST}")
        # 使用 install(CODE ...) 在安装时创建符号链接，而非复制目录
        # 这里在安装阶段会运行 CMake 代码：创建父目录、删除现有目标并创建符号链接
        set(_MODEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/models")
        set(_MODEL_DEST "${GST_VIDEORECOGNITION_MODELS_DEST}")
        install(CODE "\
            set(_dest_dir \"/opt/nvidia/deepstream/deepstream/deepstream-app-custom/gst-videorecognition\")\n\
            execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory \"\${_dest_dir}\" RESULT_VARIABLE _mkdir_ret)\n\
            if(NOT _mkdir_ret EQUAL 0)\n\
                message(FATAL_ERROR \"Failed to create directory '\${_dest_dir}' (code \${_mkdir_ret}). You may need root permissions: try 'sudo make install'\")\n\
            endif()\n\
            if(NOT EXISTS \"${_MODEL_SRC}\")\n\
                message(FATAL_ERROR \"Model source '${_MODEL_SRC}' does not exist at install time. Aborting symlink creation.\")\n\
            endif()\n\
            if(EXISTS \"${_MODEL_DEST}\")\n\
                file(REMOVE_RECURSE \"${_MODEL_DEST}\")\n\
            endif()\n\
            execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \"${_MODEL_SRC}\" \"${_MODEL_DEST}\" RESULT_VARIABLE _ln_ret)\n\
            if(NOT _ln_ret EQUAL 0)\n\
                message(FATAL_ERROR \"Failed to create symbolic link '${_MODEL_DEST}' -> '${_MODEL_SRC}' (code \${_ln_ret}). Check that source exists and you have permissions.\")\n\
            endif()\n\
            message(STATUS \"Created symlink: ${_MODEL_DEST} -> ${_MODEL_SRC}\")\n\
        ")
else()
    message(STATUS "No models directory found at ${CMAKE_CURRENT_SOURCE_DIR}/models; skipping models installation.")
endif()

# 10. 调试符号
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(gst_videorecognition PRIVATE -g)
endif()
