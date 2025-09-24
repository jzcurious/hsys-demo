cmake_minimum_required(VERSION 3.24)
project(hs2025 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(work1)
add_subdirectory(work2)

include(cmake/misc/scc.cmake)
