cmake_minimum_required(VERSION 3.18)
project(parafoil_rl LANGUAGES CXX)

# ============================================================================
# Build configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build for maximum simulation speed during rollout collection
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# ============================================================================
# Find Python and pybind11
#
# Install pybind11 via pip: pip install pybind11
# Or conda: conda install pybind11
# ============================================================================
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Try to find pybind11 via cmake config (works with pip install)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
find_package(pybind11 REQUIRED HINTS "${pybind11_DIR}")

message(STATUS "Python:    ${Python3_EXECUTABLE}")
message(STATUS "pybind11:  ${pybind11_VERSION}")
message(STATUS "Build:     ${CMAKE_BUILD_TYPE}")

# ============================================================================
# Source files
#
# These paths assume your project layout is:
#   parafoil_rl/
#     cpp_env/        <- this CMakeLists.txt and new RL env files
#       ParafoilEnv.h
#       ParafoilEnv.cpp
#       parafoil_bindings.cpp
#     <root>/         <- existing plant files
#       nineDOF_Plant.h
#       nineDOF_Plant.cpp
#       nineDOF_Parameters.h
#       nineDOF_Transform.h   (if needed)
#
# Adjust PLANT_DIR if your original C++ files are elsewhere.
# ============================================================================
set(PLANT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")   # One level up from cpp_env/

# If your nineDOF_Plant.cpp and nineDOF_Parameters.h live alongside CMakeLists.txt,
# set PLANT_DIR to CMAKE_CURRENT_SOURCE_DIR instead.

pybind11_add_module(parafoil_cpp
    # New RL environment files
    ${CMAKE_CURRENT_SOURCE_DIR}/ParafoilEnv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parafoil_bindings.cpp

    # Existing plant files - adjust paths as needed
    ${PLANT_DIR}/nineDOF_Plant.cpp
)

target_include_directories(parafoil_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}       # ParafoilEnv.h
    ${PLANT_DIR}                      # nineDOF_Plant.h, nineDOF_Parameters.h
)

# ============================================================================
# Install: copies the .so into the python_env/ directory for easy import
# ============================================================================
install(TARGETS parafoil_cpp
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../python_env)

# ============================================================================
# Build instructions (run from the cpp_env/ directory):
#
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#   make install        # copies parafoil_cpp.so to python_env/
#
# Or from project root:
#   cmake -S cpp_env -B cpp_env/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build cpp_env/build --target install -j$(nproc)
# ============================================================================
