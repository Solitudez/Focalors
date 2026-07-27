cmake_minimum_required(VERSION 3.10)
project(Focalors)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CONFIGURATION_TYPES
    Debug Hybrid PureMPI
    CACHE STRING "Available build configurations" FORCE)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
if(CMAKE_BUILD_TYPE STREQUAL "Hybrid")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -DHYBRID -DNDEBUG")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "PureMPI")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -DPUREMPI -DNDEBUG")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

cmake_policy(SET CMP0074 NEW)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/poisson_base/cmake" ${CMAKE_MODULE_PATH})
find_package(FFTW REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI)

# ------------------------ Solver ------------------------
file(GLOB_RECURSE SOLVER_HEADER_FILES
    "${CMAKE_SOURCE_DIR}/poisson_base/base/*.h" "${CMAKE_SOURCE_DIR}/poisson_base/base/*.hpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/pe/*.h"   "${CMAKE_SOURCE_DIR}/poisson_base/pe/*.hpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/instrumentor/*.h" "${CMAKE_SOURCE_DIR}/poisson_base/instrumentor/*.hpp"
    "${CMAKE_SOURCE_DIR}/ns/*.h"   "${CMAKE_SOURCE_DIR}/ns/*.hpp"
    "${CMAKE_SOURCE_DIR}/ibm_Uhlmann/*.h"   "${CMAKE_SOURCE_DIR}/ibm_Uhlmann/*.hpp"
    "${CMAKE_SOURCE_DIR}/ibm_MirrorPoint/*.h"   "${CMAKE_SOURCE_DIR}/ibm_MirrorPoint/*.hpp"
    "${CMAKE_SOURCE_DIR}/particle/*.h"   "${CMAKE_SOURCE_DIR}/particle/*.hpp"
    "${CMAKE_SOURCE_DIR}/io/*.h"   "${CMAKE_SOURCE_DIR}/io/*.hpp"
)
file(GLOB_RECURSE SOLVER_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/poisson_base/base/*.cpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/pe/*.cpp"
    "${CMAKE_SOURCE_DIR}/poisson_base/instrumentor/*.cpp"
    "${CMAKE_SOURCE_DIR}/ns/*.cpp"
    "${CMAKE_SOURCE_DIR}/ibm_Uhlmann/*.cpp"
    "${CMAKE_SOURCE_DIR}/ibm_MirrorPoint/*.cpp"
    "${CMAKE_SOURCE_DIR}/particle/*.cpp"
    "${CMAKE_SOURCE_DIR}/io/*.cpp"
)

add_library(Solver ${SOLVER_SOURCE_FILES} ${SOLVER_HEADER_FILES})
target_include_directories(Solver PUBLIC
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/poisson_base"
    ${FFTW_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(Solver PUBLIC
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,19.14>>:stdc++fs>
)
target_link_libraries(Solver PUBLIC
    ${FFTW_LIBRARIES}
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX)

# VTK
find_package(VTK COMPONENTS 
    CommonCore
    CommonMisc
    CommonDataModel
    IOXML
    IOParallelXML
    ParallelMPI
    FiltersProgrammable
)
if (NOT VTK_FOUND)
    message(FATAL_ERROR "Solver: Unable to find the VTK build folder.")
endif()
target_link_libraries(Solver PUBLIC ${VTK_LIBRARIES})
# vtk_module_autoinit is needed
vtk_module_autoinit(
    TARGETS Solver
    MODULES ${VTK_LIBRARIES}
)

# -----------------------------------------------------------------------------------------

add_subdirectory(case)
add_subdirectory(tools)

