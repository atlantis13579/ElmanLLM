cmake_minimum_required(VERSION 3.24)
project(Elman VERSION 0.1.0 LANGUAGES CXX)
get_filename_component(ELMAN_BUILD_VARIANT "${CMAKE_BINARY_DIR}" NAME)

option(ELMAN_ENABLE_CUDA "Build the native CUDA backend" OFF)
option(ELMAN_BUILD_TESTS "Build numerical and integration tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Binaries/${CMAKE_SYSTEM_NAME}/${ELMAN_BUILD_VARIANT}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Binaries/${CMAKE_SYSTEM_NAME}/${ELMAN_BUILD_VARIANT}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Binaries/${CMAKE_SYSTEM_NAME}/${ELMAN_BUILD_VARIANT}")
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()
if(ELMAN_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

file(GLOB CPU_OPERATORS CONFIGURE_DEPENDS "Src/Operators/CPU/*.cpp")
set(ELMAN_SOURCES
    Src/Backends/Cpu.cpp
    Src/Core/Autograd.cpp
    Src/Core/AutogradOps.cpp
    Src/Core/Backend.cpp
    Src/Core/Graph.cpp
    Src/Core/Tensor.cpp
    Src/Inference/Inference.cpp
    Src/IO/Checkpoint.cpp
    Src/IO/Json.cpp
    Src/Models/Model.cpp
    Src/Optimizer/Adam.cpp
    Src/Optimizer/AdamW.cpp
    Src/Optimizer/Optimizer.cpp
    Src/Optimizer/Sgd.cpp
    Src/Runtime/RunContext.cpp
    Src/Training/Trainer.cpp
    ${CPU_OPERATORS}
)
if(ELMAN_ENABLE_CUDA)
    file(GLOB CUDA_OPERATORS CONFIGURE_DEPENDS "Src/Operators/CUDA/*.cu")
    list(APPEND ELMAN_SOURCES Src/Backends/Cuda.cu ${CUDA_OPERATORS})
endif()

add_library(elman_core STATIC ${ELMAN_SOURCES})
target_include_directories(elman_core PUBLIC "${PROJECT_SOURCE_DIR}/Src")
if(ELMAN_ENABLE_CUDA)
    target_compile_definitions(elman_core PUBLIC ELMAN_HAS_CUDA)
endif()

add_executable(elman Src/Main.cpp)
target_link_libraries(elman PRIVATE elman_core)

if(ELMAN_BUILD_TESTS)
    enable_testing()
    add_executable(elman_tensor_tests Tests/TestMain.cpp)
    target_link_libraries(elman_tensor_tests PRIVATE elman_core)
    add_executable(elman_kernel_tests Tests/KernelReference.cpp)
    target_link_libraries(elman_kernel_tests PRIVATE elman_core)
    add_executable(elman_model_io_tests Tests/ModelIO.cpp)
    target_link_libraries(elman_model_io_tests PRIVATE elman_core)
    add_executable(elman_optimizer_tests Tests/OptimizerTest.cpp)
    target_link_libraries(elman_optimizer_tests PRIVATE elman_core)

    add_test(NAME tensor_cpu COMMAND elman_tensor_tests cpu)
    add_test(NAME kernels_cpu COMMAND elman_kernel_tests cpu)
    add_test(NAME model_io COMMAND elman_model_io_tests)
    add_test(NAME optimizer_cpu COMMAND elman_optimizer_tests cpu)
    if(ELMAN_ENABLE_CUDA)
        add_test(NAME tensor_cuda COMMAND elman_tensor_tests cuda)
        add_test(NAME kernels_cuda COMMAND elman_kernel_tests cuda)
        add_test(NAME optimizer_cuda COMMAND elman_optimizer_tests cuda)
        set_tests_properties(tensor_cuda kernels_cuda optimizer_cuda PROPERTIES SKIP_RETURN_CODE 77)
        foreach(model alexnet gpt2)
            if(EXISTS "${PROJECT_SOURCE_DIR}/Data/${model}_sample.csv")
                add_test(NAME ${model}_full_cuda
                    COMMAND "${CMAKE_COMMAND}"
                        "-DELMAN_EXECUTABLE=$<TARGET_FILE:elman>"
                        "-DELMAN_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
                        "-DELMAN_MODEL=${model}"
                        "-DELMAN_BACKEND=cuda"
                        -P "${PROJECT_SOURCE_DIR}/Tests/RunSmoke.cmake")
                set_tests_properties(${model}_full_cuda PROPERTIES LABELS full-model RUN_SERIAL TRUE TIMEOUT 1200)
            endif()
        endforeach()
    endif()
endif()
