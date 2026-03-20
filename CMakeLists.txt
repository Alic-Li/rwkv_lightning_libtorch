cmake_minimum_required(VERSION 3.24)

project(rwkv_lightning_libtorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(
  Torch_DIR
  "/home/alic-li/python_env/py312/lib/python3.12/site-packages/torch/share/cmake/Torch"
  CACHE PATH
  "Path to the LibTorch CMake package"
)

set(RWKV_BACKEND "cuda" CACHE STRING "Backend to build: cuda or hip")
set_property(CACHE RWKV_BACKEND PROPERTY STRINGS cuda hip)
set(TORCH_CUDA_ARCH_LIST "8.0" CACHE STRING "CUDA arch list for LibTorch build")

find_package(Torch REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Drogon CONFIG QUIET)
if(NOT Drogon_FOUND)
  find_package(Drogon QUIET)
endif()

if(RWKV_BACKEND STREQUAL "cuda")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)

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

target_link_libraries(rwkv_backend_support PUBLIC rwkv_infer_core SQLite::SQLite3)
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
  set(TORCH_RPATH "/home/alic-li/python_env/py312/lib/python3.12/site-packages/torch/lib")
  target_link_options(rwkv_infer_core INTERFACE "-Wl,-rpath,${TORCH_RPATH}")
  target_link_options(rwkv_backend_support INTERFACE "-Wl,-rpath,${TORCH_RPATH}")
  target_link_options(benchmark PRIVATE "-Wl,-rpath,${TORCH_RPATH}")
  if(TARGET rwkv_lightning)
    target_link_options(rwkv_lightning PRIVATE "-Wl,-rpath,${TORCH_RPATH}")
  endif()
endif()
