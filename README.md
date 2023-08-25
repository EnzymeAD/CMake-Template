# CMake-Template

## Usage

### Install dependencies

- cmake
- make
- llvm
- enzyme

Using spack:
```
spack env activate .
spack install
```

Using homebrew:
```
brew bundle install
```

### Configure and build

Configure the CMake project using the version of Enzyme installed on the system:
```
mkdir build && cd build
cmake ..
make
```

Configure the CMake project using a custom Enzyme version:
```
mkdir build && cd build
cmake -DEnzyme_DIR=/path/to/Enzyme/enzyme/build 
make
```
