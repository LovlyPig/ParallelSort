cmake_minimum_required(VERSION 3.18)
set(CMAKE_BUILD_TYPE Debug)

project(ParallelProgramming LANGUAGES CXX CUDA)
# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --extended-lambda -lineinfo")

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
	"${PROJECT_SOURCE_DIR}/src/*.cpp"
	"${PROJECT_SOURCE_DIR}/src/*.cu"
)

add_executable(tester ${SRC_FILES})
target_include_directories(tester PUBLIC ${PROJECT_SOURCE_DIR}/include)


