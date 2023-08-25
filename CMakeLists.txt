cmake_minimum_required(VERSION 3.9)

find_package(Enzyme REQUIRED CONFIG)

message("LLVM Version: " ${Enzyme_LLVM_VERSION_MAJOR})
message("Found LLVM at: " ${Enzyme_LLVM_BINARY_DIR})

# C and C++ Compilers can only be configured before project(...)
set(CMAKE_C_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang")
set(CMAKE_CXX_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang++")

project(EnzymeExample)

message("found dir ${Enzyme_DIR}")
message("found ${Enzyme_FOUND}")
get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
message("imported targets ${importTargetsAfter}")

add_executable(example
    multisource.c
    myblas.c
    myblas.h
)
target_link_libraries(example PUBLIC LLDEnzymeFlags)
