cmake_minimum_required(VERSION 3.16)
project(grafakos-constant-verification)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

find_package(PkgConfig REQUIRED)
pkg_check_modules(MPFR REQUIRED IMPORTED_TARGET mpfr)

find_package(hwy CONFIG REQUIRED)

#[[
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_SYSTEM_GTEST OFF CACHE BOOL "" FORCE)
set(HWY_TEST_STANDALONE OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory(3rdParty/highway)
]]

foreach (_target main sinc_math main_omp_mpfr main_omp_simd)
    add_executable(${PROJECT_NAME}-${_target} "${_target}.cc"
            csvutil.h)

    target_link_libraries(${PROJECT_NAME}-${_target} PRIVATE
            OpenMP::OpenMP_CXX
            Eigen3::Eigen
            PkgConfig::MPFR m
            hwy::hwy)

endforeach ()
