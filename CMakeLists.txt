cmake_minimum_required(VERSION 3.9)

find_package(Enzyme REQUIRED)

message("LLVM Version: " ${Enzyme_LLVM_VERSION_MAJOR})
message("Found LLVM at: " ${Enzyme_LLVM_BINARY_DIR})

# C and C++ Compilers can only be configured before project(...)
set(CMAKE_C_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang")
set(CMAKE_CXX_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang")

project(EnzymeExample)

find_package(Enzyme REQUIRED COMPONENTS ClangEnzyme)

get_target_property(ClangEnzyme ClangEnzyme-${Enzyme_LLVM_VERSION_MAJOR} LOCATION)

message("Found Enzyme at: " ${ClangEnzyme})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fuse-ld=lld -flto")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=lld -flto")
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld -flto -Wl,-mllvm=-load=${ClangEnzyme}")
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld -flto -Wl,-mllvm=-load=${ClangEnzyme}")

add_executable(example
    multisource.c
    myblas.c
    myblas.h
)
