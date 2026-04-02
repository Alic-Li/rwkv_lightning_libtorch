cmake_minimum_required(VERSION 3.24)

project(rwkv_lightning_libtorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(GNUInstallDirs)

set(Torch_DIR "" CACHE PATH "Path to the LibTorch CMake package")
if(NOT Torch_DIR AND DEFINED ENV{Torch_DIR})
  set(Torch_DIR "$ENV{Torch_DIR}" CACHE PATH "Path to the LibTorch CMake package" FORCE)
endif()
if(NOT Torch_DIR)
  message(FATAL_ERROR "Torch_DIR is required. Pass -DTorch_DIR=/path/to/torch/share/cmake/Torch")
endif()

option(RWKV_BUNDLE_RUNTIME "Create a portable runtime bundle with Torch shared libraries" ON)
set(RWKV_BUNDLE_DIR "${CMAKE_BINARY_DIR}/dist" CACHE PATH "Output directory for the portable runtime bundle")

set(RWKV_BACKEND "cuda" CACHE STRING "Backend to build: cuda or hip")
set_property(CACHE RWKV_BACKEND PROPERTY STRINGS cuda hip)
set(TORCH_CUDA_ARCH_LIST "8.0" CACHE STRING "CUDA arch list for LibTorch build")

if(MSVC)
    add_definitions(-DNOMINMAX)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4819>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

if(RWKV_BACKEND STREQUAL "hip" AND NOT TARGET roc::hipsparselt)
  # Some ROCm PyTorch wheels export torch_hip_library with a hard dependency on
  # roc::hipsparselt even when hipsparselt is not installed. Provide a no-op
  # imported target so CMake generation can proceed on systems without that
  # optional package.
  add_library(roc::hipsparselt INTERFACE IMPORTED)
endif()

find_package(Torch REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Drogon CONFIG QUIET)
if(NOT Drogon_FOUND)
  find_package(Drogon QUIET)
endif()

get_filename_component(TORCH_CMAKE_DIR "${Torch_DIR}" REALPATH)
get_filename_component(TORCH_ROOT "${TORCH_CMAKE_DIR}/../../.." REALPATH)
get_filename_component(TORCH_SITE_PACKAGES_ROOT "${TORCH_ROOT}/.." REALPATH)
set(TORCH_RUNTIME_DIR "${TORCH_ROOT}/lib")
set(RWKV_BUILD_RPATH "${TORCH_RUNTIME_DIR}")
set(RWKV_BUNDLE_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/BundleTorchRuntime.cmake")

if(RWKV_BACKEND STREQUAL "cuda")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)

  if(MSVC)
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/Zc:preprocessor -Xcompiler=/wd4819")
  endif()

  set(
    RWKV_BACKEND_SOURCES
    src/infer/cuda/rwkv7_state_fwd_fp16.cpp
    src/infer/cuda/rwkv7_state_fwd_fp16.cu
    src/infer/cuda/sampling.cu
  )
  set(RWKV_BACKEND_INCLUDE_DIR src/infer/cuda)
  set(RWKV_BACKEND_LINK_LIBS CUDA::cudart)
elseif(RWKV_BACKEND STREQUAL "hip")
  enable_language(HIP)
  set(
    RWKV_BACKEND_SOURCES
    src/infer/hip/rwkv7_state_fwd_fp16_op.hip
    src/infer/hip/rwkv7_state_fwd_fp16.hip
    src/infer/hip/sampling.hip
  )
  set(RWKV_BACKEND_INCLUDE_DIR src/infer/hip)
  set(RWKV_BACKEND_LINK_LIBS "")
else()
  message(FATAL_ERROR "Unsupported RWKV_BACKEND='${RWKV_BACKEND}'. Use cuda or hip.")
endif()

add_library(
  rwkv_infer_core
  src/infer/rwkv_model.cpp
  src/model_load/safetensors_loader.cpp
  src/infer/utils/tokenizer.cpp
  ${RWKV_BACKEND_SOURCES}
)

target_include_directories(
  rwkv_infer_core
  PUBLIC
    src
    src/infer
    src/infer/utils
    ${RWKV_BACKEND_INCLUDE_DIR}
)

target_compile_definitions(rwkv_infer_core PUBLIC _N_=64)

target_compile_options(
  rwkv_infer_core
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
)

target_link_libraries(rwkv_infer_core PUBLIC ${TORCH_LIBRARIES} ${RWKV_BACKEND_LINK_LIBS})

add_library(
  rwkv_backend_support
  src/model_load/model_loader.cpp
  src/infer/inference_engine.cpp
  src/state_manager/state_pool.cpp
)

target_link_libraries(rwkv_backend_support PUBLIC rwkv_infer_core SQLite3::SQLite3)
target_include_directories(rwkv_backend_support PUBLIC src src/infer src/infer/utils)
target_compile_options(rwkv_backend_support PRIVATE -O3)

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE rwkv_infer_core)

if(Drogon_FOUND)
  add_executable(
    rwkv_lightning
    src/rwkv_lightning.cpp
    src/API_servers/api_service.cpp
  )
  target_link_libraries(rwkv_lightning PRIVATE rwkv_backend_support Drogon::Drogon)
  target_include_directories(rwkv_lightning PRIVATE src src/API_servers src/model_load src/state_manager src/infer src/infer/utils)
  target_compile_options(rwkv_lightning PRIVATE -O3)
else()
  message(WARNING "Drogon not found. 'rwkv_lightning' server target will not be built in this environment.")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  file(GLOB TORCH_NVIDIA_RUNTIME_DIRS LIST_DIRECTORIES true "${TORCH_SITE_PACKAGES_ROOT}/nvidia/*/lib")
  list(APPEND RWKV_BUILD_RPATH ${TORCH_NVIDIA_RUNTIME_DIRS})
  list(REMOVE_DUPLICATES RWKV_BUILD_RPATH)
  set(RWKV_INSTALL_RPATH "\$ORIGIN/../torch/lib" "\$ORIGIN/../third_party/lib")
  foreach(TORCH_NVIDIA_RUNTIME_DIR IN LISTS TORCH_NVIDIA_RUNTIME_DIRS)
    get_filename_component(TORCH_NVIDIA_PACKAGE_DIR "${TORCH_NVIDIA_RUNTIME_DIR}" DIRECTORY)
    get_filename_component(TORCH_NVIDIA_PACKAGE_NAME "${TORCH_NVIDIA_PACKAGE_DIR}" NAME)
    list(APPEND RWKV_INSTALL_RPATH "\$ORIGIN/../nvidia/${TORCH_NVIDIA_PACKAGE_NAME}/lib")
  endforeach()
  list(REMOVE_DUPLICATES RWKV_INSTALL_RPATH)

  set_target_properties(
    benchmark
    PROPERTIES
      BUILD_RPATH "${RWKV_BUILD_RPATH}"
      INSTALL_RPATH "${RWKV_INSTALL_RPATH}"
  )
  target_link_options(benchmark PRIVATE "-Wl,--disable-new-dtags")
  if(TARGET rwkv_lightning)
    set_target_properties(
      rwkv_lightning
      PROPERTIES
        BUILD_RPATH "${RWKV_BUILD_RPATH}"
        INSTALL_RPATH "${RWKV_INSTALL_RPATH}"
    )
    target_link_options(rwkv_lightning PRIVATE "-Wl,--disable-new-dtags")
  endif()
endif()

install(TARGETS benchmark RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(TARGET rwkv_lightning)
  install(TARGETS rwkv_lightning RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(RWKV_BUNDLE_RUNTIME AND EXISTS "${TORCH_RUNTIME_DIR}")
  set(RWKV_BUNDLE_TARGET_PATHS "\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/benchmark${CMAKE_EXECUTABLE_SUFFIX}")
  if(TARGET rwkv_lightning)
    list(APPEND RWKV_BUNDLE_TARGET_PATHS "\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/rwkv_lightning${CMAKE_EXECUTABLE_SUFFIX}")
  endif()
  string(REPLACE ";" "|" RWKV_BUNDLE_TARGET_PATHS_JOINED "${RWKV_BUNDLE_TARGET_PATHS}")

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BundleTorchRuntime.cmake.in"
    "${RWKV_BUNDLE_SCRIPT}"
    @ONLY
  )

  install(SCRIPT "${RWKV_BUNDLE_SCRIPT}")
endif()

add_custom_target(
  bundle
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${RWKV_BUNDLE_DIR}"
  COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --prefix "${RWKV_BUNDLE_DIR}"
  DEPENDS benchmark
  COMMENT "Installing portable runtime bundle to ${RWKV_BUNDLE_DIR}"
)

if(TARGET rwkv_lightning)
  add_dependencies(bundle rwkv_lightning)
endif()
