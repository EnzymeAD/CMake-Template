cmake_minimum_required(VERSION 3.9)

find_package(Enzyme REQUIRED CONFIG)

message("LLVM Version: " ${Enzyme_LLVM_VERSION_MAJOR})
message("Found LLVM at: " ${Enzyme_LLVM_BINARY_DIR})

# C and C++ Compilers can only be configured before project(...)
set(CMAKE_C_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang")
set(CMAKE_CXX_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang++")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --save-temps -Xclang -D -Xclang EIGEN_USE_BLAS -Xclang -no-opaque-pointers")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --save-temps -Xclang -D -Xclang EIGEN_USE_BLAS -Xclang -no-opaque-pointers")

project(EnzymeExample CXX)

set(BLA_VENDER OpenBLAS)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("OpenBLAS found.")
    #include_directories(/opt/OpenBLAS/include/)
    #target_link_libraries(main.exe ${BLAS_LIBRARIES})
endif(BLAS_FOUND)

message("found dir ${Enzyme_DIR}")
message("found ${Enzyme_FOUND}")
get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
message("imported targets ${importTargetsAfter}")

 
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 

add_executable(example
    multisource.cpp
    myblas.c
    myblas.h
)
target_link_libraries(example PUBLIC LLDEnzymeFlags Eigen3::Eigen ${BLAS_LIBRARIES})
