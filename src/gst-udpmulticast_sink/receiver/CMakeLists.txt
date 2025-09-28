cmake_minimum_required(VERSION 3.10)
project(eo_receiver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP_PKG jsoncpp)
if(JSONCPP_PKG_FOUND)
  set(JSONCPP_INCLUDE_DIRS ${JSONCPP_PKG_INCLUDE_DIRS})
  set(JSONCPP_LIBRARIES ${JSONCPP_PKG_LIBRARIES})
else()
  message(STATUS "jsoncpp pkg-config not found, will try link name jsoncpp")
endif()

add_executable(eo_receiver
  eo_receiver.cpp
  main.cpp
  ../eo_protocol_parser.cpp
)

target_include_directories(eo_receiver PRIVATE
  ${JSONCPP_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/..
)

target_link_libraries(eo_receiver PRIVATE
  $<$<BOOL:${JSONCPP_LIBRARIES}>:${JSONCPP_LIBRARIES}>
  $<$<NOT:$<BOOL:${JSONCPP_LIBRARIES}>>:jsoncpp>
)

install(TARGETS eo_receiver RUNTIME DESTINATION bin)
