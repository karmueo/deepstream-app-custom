cmake_minimum_required(VERSION 3.16)
project(sot)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(DS_ROOT_DIR /opt/nvidia/deepstream/deepstream)
set(DS_LIBS_DIR ${DS_ROOT_DIR}/lib)
set(DS_INCLUDES_DIR ${DS_ROOT_DIR}/sources/includes)

find_package(GLib)
find_package(OpenCV)

include_directories(${GLIB2_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${DS_INCLUDES_DIR})
include_directories(${CMAKE_SOURCE_DIR}/includes)
include_directories(/usr/local/cuda-12.6/targets/x86_64-linux/include/)

file(GLOB TRACK ${CMAKE_SOURCE_DIR}/src/*.cpp)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC -g")
set(CMAKE_CUDA_FLAGS "--compiler-options -fPIC -g")

link_directories(${DS_LIBS_DIR})
add_library(${PROJECT_NAME} SHARED ${TRACK})

target_link_libraries(${PROJECT_NAME} nvds_meta ${OpenCV_LIBS})

# 添加安装
set(GST_PLUGINS_DIR "/opt/nvidia/deepstream/deepstream/lib")
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ${GST_PLUGINS_DIR}
        COMPONENT runtime
)