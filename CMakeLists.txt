cmake_minimum_required(VERSION 3.24)

project(breeze LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS)

add_library(breeze
  src/tensor.cu
  src/ops.cu
)

target_include_directories(breeze
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(breeze
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wall>
)

include(GNUInstallDirs)
install(TARGETS breeze
  EXPORT breeze-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT breeze-targets
  FILE breeze-config.cmake
  NAMESPACE breeze::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/breeze
)

enable_testing()
find_package(GTest REQUIRED)

add_subdirectory(tests)

add_executable(demo demo/main.cpp)
target_link_libraries(demo PRIVATE breeze)

