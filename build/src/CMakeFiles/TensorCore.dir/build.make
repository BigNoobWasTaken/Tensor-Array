# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/d/C++_ver2/TensorMain_wsl2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/d/C++_ver2/TensorMain_wsl2/build

# Include any dependencies generated for this target.
include src/CMakeFiles/TensorCore.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/TensorCore.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/TensorCore.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/TensorCore.dir/flags.make

src/CMakeFiles/TensorCore.dir/convolution.cc.o: src/CMakeFiles/TensorCore.dir/flags.make
src/CMakeFiles/TensorCore.dir/convolution.cc.o: ../src/convolution.cc
src/CMakeFiles/TensorCore.dir/convolution.cc.o: src/CMakeFiles/TensorCore.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/TensorCore.dir/convolution.cc.o"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/TensorCore.dir/convolution.cc.o -MF CMakeFiles/TensorCore.dir/convolution.cc.o.d -o CMakeFiles/TensorCore.dir/convolution.cc.o -c /mnt/d/C++_ver2/TensorMain_wsl2/src/convolution.cc

src/CMakeFiles/TensorCore.dir/convolution.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TensorCore.dir/convolution.cc.i"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/C++_ver2/TensorMain_wsl2/src/convolution.cc > CMakeFiles/TensorCore.dir/convolution.cc.i

src/CMakeFiles/TensorCore.dir/convolution.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TensorCore.dir/convolution.cc.s"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/C++_ver2/TensorMain_wsl2/src/convolution.cc -o CMakeFiles/TensorCore.dir/convolution.cc.s

src/CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.o: src/CMakeFiles/TensorCore.dir/flags.make
src/CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.o: ../src/core/data_type_wrapper.cc
src/CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.o: src/CMakeFiles/TensorCore.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.o"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.o -MF CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.o.d -o CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.o -c /mnt/d/C++_ver2/TensorMain_wsl2/src/core/data_type_wrapper.cc

src/CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.i"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/C++_ver2/TensorMain_wsl2/src/core/data_type_wrapper.cc > CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.i

src/CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.s"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/C++_ver2/TensorMain_wsl2/src/core/data_type_wrapper.cc -o CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.s

src/CMakeFiles/TensorCore.dir/core/tensor.cc.o: src/CMakeFiles/TensorCore.dir/flags.make
src/CMakeFiles/TensorCore.dir/core/tensor.cc.o: ../src/core/tensor.cc
src/CMakeFiles/TensorCore.dir/core/tensor.cc.o: src/CMakeFiles/TensorCore.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/TensorCore.dir/core/tensor.cc.o"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/TensorCore.dir/core/tensor.cc.o -MF CMakeFiles/TensorCore.dir/core/tensor.cc.o.d -o CMakeFiles/TensorCore.dir/core/tensor.cc.o -c /mnt/d/C++_ver2/TensorMain_wsl2/src/core/tensor.cc

src/CMakeFiles/TensorCore.dir/core/tensor.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TensorCore.dir/core/tensor.cc.i"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/C++_ver2/TensorMain_wsl2/src/core/tensor.cc > CMakeFiles/TensorCore.dir/core/tensor.cc.i

src/CMakeFiles/TensorCore.dir/core/tensor.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TensorCore.dir/core/tensor.cc.s"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/C++_ver2/TensorMain_wsl2/src/core/tensor.cc -o CMakeFiles/TensorCore.dir/core/tensor.cc.s

src/CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.o: src/CMakeFiles/TensorCore.dir/flags.make
src/CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.o: ../src/core/tensor_convolution.cc
src/CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.o: src/CMakeFiles/TensorCore.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.o"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.o -MF CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.o.d -o CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.o -c /mnt/d/C++_ver2/TensorMain_wsl2/src/core/tensor_convolution.cc

src/CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.i"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/C++_ver2/TensorMain_wsl2/src/core/tensor_convolution.cc > CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.i

src/CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.s"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/C++_ver2/TensorMain_wsl2/src/core/tensor_convolution.cc -o CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.s

src/CMakeFiles/TensorCore.dir/core/tensorbase.cc.o: src/CMakeFiles/TensorCore.dir/flags.make
src/CMakeFiles/TensorCore.dir/core/tensorbase.cc.o: ../src/core/tensorbase.cc
src/CMakeFiles/TensorCore.dir/core/tensorbase.cc.o: src/CMakeFiles/TensorCore.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/TensorCore.dir/core/tensorbase.cc.o"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/TensorCore.dir/core/tensorbase.cc.o -MF CMakeFiles/TensorCore.dir/core/tensorbase.cc.o.d -o CMakeFiles/TensorCore.dir/core/tensorbase.cc.o -c /mnt/d/C++_ver2/TensorMain_wsl2/src/core/tensorbase.cc

src/CMakeFiles/TensorCore.dir/core/tensorbase.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TensorCore.dir/core/tensorbase.cc.i"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/C++_ver2/TensorMain_wsl2/src/core/tensorbase.cc > CMakeFiles/TensorCore.dir/core/tensorbase.cc.i

src/CMakeFiles/TensorCore.dir/core/tensorbase.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TensorCore.dir/core/tensorbase.cc.s"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/C++_ver2/TensorMain_wsl2/src/core/tensorbase.cc -o CMakeFiles/TensorCore.dir/core/tensorbase.cc.s

src/CMakeFiles/TensorCore.dir/layer_impl.cc.o: src/CMakeFiles/TensorCore.dir/flags.make
src/CMakeFiles/TensorCore.dir/layer_impl.cc.o: ../src/layer_impl.cc
src/CMakeFiles/TensorCore.dir/layer_impl.cc.o: src/CMakeFiles/TensorCore.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/TensorCore.dir/layer_impl.cc.o"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/TensorCore.dir/layer_impl.cc.o -MF CMakeFiles/TensorCore.dir/layer_impl.cc.o.d -o CMakeFiles/TensorCore.dir/layer_impl.cc.o -c /mnt/d/C++_ver2/TensorMain_wsl2/src/layer_impl.cc

src/CMakeFiles/TensorCore.dir/layer_impl.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TensorCore.dir/layer_impl.cc.i"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/C++_ver2/TensorMain_wsl2/src/layer_impl.cc > CMakeFiles/TensorCore.dir/layer_impl.cc.i

src/CMakeFiles/TensorCore.dir/layer_impl.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TensorCore.dir/layer_impl.cc.s"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/C++_ver2/TensorMain_wsl2/src/layer_impl.cc -o CMakeFiles/TensorCore.dir/layer_impl.cc.s

src/CMakeFiles/TensorCore.dir/layer_utility.cc.o: src/CMakeFiles/TensorCore.dir/flags.make
src/CMakeFiles/TensorCore.dir/layer_utility.cc.o: ../src/layer_utility.cc
src/CMakeFiles/TensorCore.dir/layer_utility.cc.o: src/CMakeFiles/TensorCore.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/CMakeFiles/TensorCore.dir/layer_utility.cc.o"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/TensorCore.dir/layer_utility.cc.o -MF CMakeFiles/TensorCore.dir/layer_utility.cc.o.d -o CMakeFiles/TensorCore.dir/layer_utility.cc.o -c /mnt/d/C++_ver2/TensorMain_wsl2/src/layer_utility.cc

src/CMakeFiles/TensorCore.dir/layer_utility.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TensorCore.dir/layer_utility.cc.i"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/C++_ver2/TensorMain_wsl2/src/layer_utility.cc > CMakeFiles/TensorCore.dir/layer_utility.cc.i

src/CMakeFiles/TensorCore.dir/layer_utility.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TensorCore.dir/layer_utility.cc.s"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/C++_ver2/TensorMain_wsl2/src/layer_utility.cc -o CMakeFiles/TensorCore.dir/layer_utility.cc.s

src/CMakeFiles/TensorCore.dir/linear.cc.o: src/CMakeFiles/TensorCore.dir/flags.make
src/CMakeFiles/TensorCore.dir/linear.cc.o: ../src/linear.cc
src/CMakeFiles/TensorCore.dir/linear.cc.o: src/CMakeFiles/TensorCore.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/CMakeFiles/TensorCore.dir/linear.cc.o"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/TensorCore.dir/linear.cc.o -MF CMakeFiles/TensorCore.dir/linear.cc.o.d -o CMakeFiles/TensorCore.dir/linear.cc.o -c /mnt/d/C++_ver2/TensorMain_wsl2/src/linear.cc

src/CMakeFiles/TensorCore.dir/linear.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TensorCore.dir/linear.cc.i"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/C++_ver2/TensorMain_wsl2/src/linear.cc > CMakeFiles/TensorCore.dir/linear.cc.i

src/CMakeFiles/TensorCore.dir/linear.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TensorCore.dir/linear.cc.s"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/C++_ver2/TensorMain_wsl2/src/linear.cc -o CMakeFiles/TensorCore.dir/linear.cc.s

src/CMakeFiles/TensorCore.dir/normalization.cc.o: src/CMakeFiles/TensorCore.dir/flags.make
src/CMakeFiles/TensorCore.dir/normalization.cc.o: ../src/normalization.cc
src/CMakeFiles/TensorCore.dir/normalization.cc.o: src/CMakeFiles/TensorCore.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/CMakeFiles/TensorCore.dir/normalization.cc.o"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/TensorCore.dir/normalization.cc.o -MF CMakeFiles/TensorCore.dir/normalization.cc.o.d -o CMakeFiles/TensorCore.dir/normalization.cc.o -c /mnt/d/C++_ver2/TensorMain_wsl2/src/normalization.cc

src/CMakeFiles/TensorCore.dir/normalization.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TensorCore.dir/normalization.cc.i"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/C++_ver2/TensorMain_wsl2/src/normalization.cc > CMakeFiles/TensorCore.dir/normalization.cc.i

src/CMakeFiles/TensorCore.dir/normalization.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TensorCore.dir/normalization.cc.s"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/C++_ver2/TensorMain_wsl2/src/normalization.cc -o CMakeFiles/TensorCore.dir/normalization.cc.s

src/CMakeFiles/TensorCore.dir/recurrent.cc.o: src/CMakeFiles/TensorCore.dir/flags.make
src/CMakeFiles/TensorCore.dir/recurrent.cc.o: ../src/recurrent.cc
src/CMakeFiles/TensorCore.dir/recurrent.cc.o: src/CMakeFiles/TensorCore.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/CMakeFiles/TensorCore.dir/recurrent.cc.o"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/TensorCore.dir/recurrent.cc.o -MF CMakeFiles/TensorCore.dir/recurrent.cc.o.d -o CMakeFiles/TensorCore.dir/recurrent.cc.o -c /mnt/d/C++_ver2/TensorMain_wsl2/src/recurrent.cc

src/CMakeFiles/TensorCore.dir/recurrent.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TensorCore.dir/recurrent.cc.i"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/C++_ver2/TensorMain_wsl2/src/recurrent.cc > CMakeFiles/TensorCore.dir/recurrent.cc.i

src/CMakeFiles/TensorCore.dir/recurrent.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TensorCore.dir/recurrent.cc.s"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/C++_ver2/TensorMain_wsl2/src/recurrent.cc -o CMakeFiles/TensorCore.dir/recurrent.cc.s

src/CMakeFiles/TensorCore.dir/sequential.cc.o: src/CMakeFiles/TensorCore.dir/flags.make
src/CMakeFiles/TensorCore.dir/sequential.cc.o: ../src/sequential.cc
src/CMakeFiles/TensorCore.dir/sequential.cc.o: src/CMakeFiles/TensorCore.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object src/CMakeFiles/TensorCore.dir/sequential.cc.o"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/TensorCore.dir/sequential.cc.o -MF CMakeFiles/TensorCore.dir/sequential.cc.o.d -o CMakeFiles/TensorCore.dir/sequential.cc.o -c /mnt/d/C++_ver2/TensorMain_wsl2/src/sequential.cc

src/CMakeFiles/TensorCore.dir/sequential.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TensorCore.dir/sequential.cc.i"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/C++_ver2/TensorMain_wsl2/src/sequential.cc > CMakeFiles/TensorCore.dir/sequential.cc.i

src/CMakeFiles/TensorCore.dir/sequential.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TensorCore.dir/sequential.cc.s"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/C++_ver2/TensorMain_wsl2/src/sequential.cc -o CMakeFiles/TensorCore.dir/sequential.cc.s

src/CMakeFiles/TensorCore.dir/transformer.cc.o: src/CMakeFiles/TensorCore.dir/flags.make
src/CMakeFiles/TensorCore.dir/transformer.cc.o: ../src/transformer.cc
src/CMakeFiles/TensorCore.dir/transformer.cc.o: src/CMakeFiles/TensorCore.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object src/CMakeFiles/TensorCore.dir/transformer.cc.o"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/TensorCore.dir/transformer.cc.o -MF CMakeFiles/TensorCore.dir/transformer.cc.o.d -o CMakeFiles/TensorCore.dir/transformer.cc.o -c /mnt/d/C++_ver2/TensorMain_wsl2/src/transformer.cc

src/CMakeFiles/TensorCore.dir/transformer.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TensorCore.dir/transformer.cc.i"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/C++_ver2/TensorMain_wsl2/src/transformer.cc > CMakeFiles/TensorCore.dir/transformer.cc.i

src/CMakeFiles/TensorCore.dir/transformer.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TensorCore.dir/transformer.cc.s"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/C++_ver2/TensorMain_wsl2/src/transformer.cc -o CMakeFiles/TensorCore.dir/transformer.cc.s

# Object files for target TensorCore
TensorCore_OBJECTS = \
"CMakeFiles/TensorCore.dir/convolution.cc.o" \
"CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.o" \
"CMakeFiles/TensorCore.dir/core/tensor.cc.o" \
"CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.o" \
"CMakeFiles/TensorCore.dir/core/tensorbase.cc.o" \
"CMakeFiles/TensorCore.dir/layer_impl.cc.o" \
"CMakeFiles/TensorCore.dir/layer_utility.cc.o" \
"CMakeFiles/TensorCore.dir/linear.cc.o" \
"CMakeFiles/TensorCore.dir/normalization.cc.o" \
"CMakeFiles/TensorCore.dir/recurrent.cc.o" \
"CMakeFiles/TensorCore.dir/sequential.cc.o" \
"CMakeFiles/TensorCore.dir/transformer.cc.o"

# External object files for target TensorCore
TensorCore_EXTERNAL_OBJECTS =

src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/convolution.cc.o
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/core/data_type_wrapper.cc.o
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/core/tensor.cc.o
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/core/tensor_convolution.cc.o
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/core/tensorbase.cc.o
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/layer_impl.cc.o
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/layer_utility.cc.o
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/linear.cc.o
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/normalization.cc.o
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/recurrent.cc.o
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/sequential.cc.o
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/transformer.cc.o
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/build.make
src/libTensorCore.so: src/CMakeFiles/TensorCore.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/C++_ver2/TensorMain_wsl2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Linking CXX shared library libTensorCore.so"
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TensorCore.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/TensorCore.dir/build: src/libTensorCore.so
.PHONY : src/CMakeFiles/TensorCore.dir/build

src/CMakeFiles/TensorCore.dir/clean:
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build/src && $(CMAKE_COMMAND) -P CMakeFiles/TensorCore.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/TensorCore.dir/clean

src/CMakeFiles/TensorCore.dir/depend:
	cd /mnt/d/C++_ver2/TensorMain_wsl2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/C++_ver2/TensorMain_wsl2 /mnt/d/C++_ver2/TensorMain_wsl2/src /mnt/d/C++_ver2/TensorMain_wsl2/build /mnt/d/C++_ver2/TensorMain_wsl2/build/src /mnt/d/C++_ver2/TensorMain_wsl2/build/src/CMakeFiles/TensorCore.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/TensorCore.dir/depend

