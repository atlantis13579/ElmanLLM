cmake_minimum_required(VERSION 3.24)
project(Elman VERSION 0.1.0 LANGUAGES CXX)
get_filename_component(ELMAN_BUILD_VARIANT "${CMAKE_BINARY_DIR}" NAME)

option(ELMAN_ENABLE_CUDA "Build the native CUDA backend" OFF)
option(ELMAN_ENABLE_METAL "Build the Apple Metal backend using MPSGraph" ${APPLE})
option(ELMAN_BUILD_TESTS "Build numerical and integration tests" ON)
set(ELMAN_LLM_MODEL_DIR "" CACHE PATH "Optional model/tokenizer directory for the native chat test")
set(ELMAN_CUBLAS_ROOT "" CACHE PATH "Optional cuBLAS runtime and import-library root")
set(ELMAN_CUDNN_ROOT "" CACHE PATH "Optional cuDNN 9 root for CUDA convolution")
set(ELMAN_MKL_ROOT "" CACHE PATH "Optional oneMKL root for native CPU Float32 GEMM")
set(ELMAN_SLEEF_ROOT "" CACHE PATH "Optional static SLEEF root for AVX2 CPU Float32 math")

if(ELMAN_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "ELMAN_ENABLE_METAL requires the macOS SDK")
    endif()
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")
    elseif(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 15.0)
        message(FATAL_ERROR "ELMAN_ENABLE_METAL requires macOS 15.0 or newer")
    endif()
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Binaries/${CMAKE_SYSTEM_NAME}/${ELMAN_BUILD_VARIANT}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Lib")
if(CMAKE_CONFIGURATION_TYPES)
    set(ELMAN_TEST_RUNTIME_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>")
else()
    set(ELMAN_TEST_RUNTIME_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()
if(ELMAN_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

file(GLOB CPU_OPERATORS CONFIGURE_DEPENDS "Src/Operators/CPU/*.cpp")
set(ELMAN_SOURCES
    Src/Training/GraphTraining.cpp
    Src/Serving/GenerationScheduler.cpp
    Src/Backends/Cpu.cpp
    Src/Backends/CpuGemm.cpp
    Src/Backends/CpuMath.cpp
    Src/Backends/CpuParallel.cpp
    Src/Core/Autograd.cpp
    Src/Core/AutogradOps.cpp
    Src/Core/Autocast.cpp
    Src/Core/Backend.cpp
    Src/Core/AllocationPool.cpp
    Src/Core/Execution.cpp
    Src/Core/Device.cpp
    Src/Core/Memory.cpp
    Src/Core/DType.cpp
    Src/Core/IntegerPointwise.cpp
    Src/Core/Float64Pointwise.cpp
    Src/Core/LowPrecisionCast.cpp
    Src/Core/LowPrecisionPointwise.cpp
    Src/Core/LowPrecisionReduction.cpp
    Src/Core/LowPrecisionMatrix.cpp
    Src/Core/LowPrecisionSoftmax.cpp
    Src/Core/LowPrecisionNormalization.cpp
    Src/Core/LowPrecisionAttention.cpp
    Src/Core/LowPrecisionAttentionExecution.cpp
    Src/Core/LowPrecisionFused.cpp
    Src/Core/LowPrecisionRandom.cpp
    Src/Core/LowPrecisionCategorical.cpp
    Src/Core/LowPrecisionOptimizer.cpp
    Src/Core/FloatingDropout.cpp
    Src/Core/FloatingRandom.cpp
    Src/Core/LowPrecisionLoss.cpp
    Src/Core/LowPrecisionSpatial.cpp
    Src/Core/LowPrecisionIndexing.cpp
    Src/Core/Float64Reduction.cpp
    Src/Core/Float64Matrix.cpp
    Src/Core/Float64Softmax.cpp
    Src/Core/Float64Indexing.cpp
    Src/Core/Float64Spatial.cpp
    Src/Core/Float64Loss.cpp
    Src/Core/Float64Attention.cpp
    Src/Core/Float64Random.cpp
    Src/Core/Float64Categorical.cpp
    Src/Core/Float64Normalization.cpp
    Src/Core/IntegerMatrix.cpp
    Src/Core/PackedIntegerCast.cpp
    Src/Core/PackedIntegerExecution.cpp
    Src/Core/Quantization.cpp
    Src/Core/QuantizationDispatch.cpp
    Src/Core/StorageLayout.cpp
    Src/Core/Graph.cpp
    Src/Core/CommandGraph.cpp
    Src/Core/Tensor.cpp
    Src/Operators/inference_ops.cpp
    Src/Operators/flux2_ops.cpp
    Src/Operators/upsample.cpp
    Src/Inference/Diffusion/Ddim.cpp
    Src/Models/ClipText.cpp
    Src/Runtime/DiffusionCommand.cpp
    Src/Inference/Diffusion/StableDiffusion.cpp
    Src/Models/Diffusion/UNet.cpp
    Src/Models/Diffusion/Vae.cpp
    Src/Models/Diffusion/Flux2Text.cpp
    Src/Models/Diffusion/Flux2Transformer.cpp
    Src/Models/Diffusion/Flux2Vae.cpp
    Src/Inference/Diffusion/Flux2Sampler.cpp
    Src/Inference/Diffusion/Flux2Session.cpp
    Src/Layers/LayerNorm.cpp
    Src/Layers/Attention.cpp
    Src/Models/Diffusion/Blocks.cpp
    Src/IO/ComponentWeights.cpp
    Src/Distributed/ProcessGroup.cpp
    Src/Distributed/TensorParallel.cpp
    Src/Core/ComplexTensor.cpp
    Src/Core/ComplexPointwise.cpp
    Src/Core/ComplexAutograd.cpp
    Src/Reinforcement/PolicyValueInference.cpp
    Src/Inference/ModelEvaluation.cpp
    Src/Inference/GraphSession.cpp
    Src/Inference/GraphBuilder.cpp
    Src/Inference/Importers/SavedModel.cpp
    Src/Inference/Importers/TorchScript.cpp
    Src/Inference/Importers/Keras.cpp
    Src/Inference/Importers/TfLite.cpp
    Src/Inference/InferenceSession.cpp
    Src/Inference/LLM/CacheBlockPool.cpp
    Src/Inference/LLM/PagedKVCache.cpp
    Src/Inference/LLM/PagedAttention.cpp
    Src/Inference/LLM/LatentAttention.cpp
    Src/Inference/LLM/Experts.cpp
    Src/Inference/LLM/ProjectionWeight.cpp
    Src/Inference/LLM/DevicePlacement.cpp
    Src/Inference/LLM/Qwen2Graph.cpp
    Src/Inference/LLM/Qwen2PagedGraph.cpp
    Src/Inference/LLM/PrefixCache.cpp
    Src/Inference/LLM/Generation.cpp
    Src/Inference/LLM/MemoryPlan.cpp
    Src/Inference/LLM/Sampling.cpp
    Src/IO/Checkpoint.cpp
    Src/IO/AtomicFile.cpp
    Src/IO/Jpeg.cpp
    Src/IO/Png.cpp
    Src/IO/Json.cpp
    Src/IO/Safetensors.cpp
    Src/IO/InferenceWeights.cpp
    Src/IO/TensorArchive.cpp
    Src/IO/Gguf.cpp
    Src/IO/NumPy.cpp
    Src/IO/TorchWeights.cpp
    Src/IO/TensorFlowBundle.cpp
    Src/IO/TensorFlowProto.cpp
    Src/IO/Hdf5.cpp
    Src/IO/ZipArchive.cpp
    Src/Models/Model.cpp
    Src/Reinforcement/ContinuousActorCritic.cpp
    Src/Models/ModuleModel.cpp
    Src/Models/LLM/Decoder.cpp
    Src/Models/LLM/DeepSeek.cpp
    Src/Models/LLM/DecoderFactory.cpp
    Src/Models/LLM/Qwen2.cpp
    Src/Models/ResNet.cpp
    Src/Reinforcement/SpatialPolicyValue.cpp
    Src/Layers/Module.cpp
    Src/Layers/Sequential.cpp
    Src/Layers/Linear.cpp
    Src/Layers/Conv2d.cpp
    Src/Layers/GroupNorm.cpp
    Src/Layers/BatchNorm2d.cpp
    Src/Layers/ReLU.cpp
    Src/Layers/Dropout.cpp
    Src/Optimizer/Adam.cpp
    Src/Optimizer/AdamW.cpp
    Src/Optimizer/Optimizer.cpp
    Src/Optimizer/Double.cpp
    Src/Optimizer/Complex.cpp
    Src/Optimizer/Low.cpp
    Src/Optimizer/Sgd.cpp
    Src/Runtime/RunContext.cpp
    Src/Text/Tokenizer.cpp
    Src/Text/HuggingFaceTokenizer.cpp
    Src/Text/DeepSeekChat.cpp
    Src/Text/ClipTokenizer.cpp
    Src/Text/Unicode.cpp
    Src/Text/TokenStream.cpp
    Src/Training/DataLoader.cpp
    Src/Training/LossScaler.cpp
    Src/Reinforcement/Reinforcement.cpp
    Src/Reinforcement/PPO.cpp
    Src/Reinforcement/PpoRunner.cpp
    Src/Reinforcement/PolicyValueTraining.cpp
    Src/Training/Sampler.cpp
    Src/Training/Trainer.cpp
    ${CPU_OPERATORS}
)
if(ELMAN_ENABLE_CUDA)
    file(GLOB CUDA_OPERATORS CONFIGURE_DEPENDS "Src/Operators/CUDA/*.cu")
    list(APPEND ELMAN_SOURCES Src/Backends/Cuda.cu Src/Backends/CudaExecution.cu Src/Backends/CudaGraph.cu Src/Backends/CudaGemm.cu ${CUDA_OPERATORS})
    if(MSVC)
        # CCCL requires the conforming MSVC preprocessor in CUDA 13.
        set_source_files_properties(Src/Operators/CUDA/argsort_cuda.cu Src/Operators/CUDA/cumsum_cuda.cu
            PROPERTIES COMPILE_OPTIONS "-Xcompiler=/Zc:preprocessor")
        set_source_files_properties(Src/Operators/CUDA/complex_pointwise_cuda.cu
            PROPERTIES COMPILE_OPTIONS "-Xcompiler=/Zc:preprocessor")
    endif()
endif()

add_library(elman_core STATIC ${ELMAN_SOURCES})
find_package(Threads REQUIRED)
target_link_libraries(elman_core PRIVATE Threads::Threads)
target_include_directories(elman_core PUBLIC "${PROJECT_SOURCE_DIR}/Src")
if(ELMAN_ENABLE_METAL)
    file(GLOB METAL_OPERATORS CONFIGURE_DEPENDS "Src/Operators/Metal/*_metal.mm")
    target_sources(elman_core PRIVATE Src/Backends/Metal.mm Src/Backends/MPSGraph.mm ${METAL_OPERATORS})
    target_compile_options(elman_core PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    target_compile_definitions(elman_core PUBLIC ELMAN_HAS_METAL)
    foreach(framework Foundation Metal MetalPerformanceShaders MetalPerformanceShadersGraph)
        find_library(ELMAN_${framework}_FRAMEWORK ${framework} REQUIRED)
        target_link_libraries(elman_core PRIVATE "${ELMAN_${framework}_FRAMEWORK}")
    endforeach()
endif()
if(ELMAN_SLEEF_ROOT)
    if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$")
        message(FATAL_ERROR "ELMAN_SLEEF_ROOT currently supports x86-64 AVX2 targets only")
    endif()
    find_library(ELMAN_SLEEF_LIBRARY NAMES sleef HINTS "${ELMAN_SLEEF_ROOT}/lib" NO_DEFAULT_PATH NO_CACHE REQUIRED)
    find_path(ELMAN_SLEEF_INCLUDE sleef.h HINTS "${ELMAN_SLEEF_ROOT}/include" NO_DEFAULT_PATH NO_CACHE REQUIRED)
    target_sources(elman_core PRIVATE Src/Backends/CpuMathAvx2.cpp)
    set_source_files_properties(Src/Backends/CpuMathAvx2.cpp PROPERTIES
        INCLUDE_DIRECTORIES "${ELMAN_SLEEF_INCLUDE}" COMPILE_DEFINITIONS SLEEF_STATIC_LIBS)
    set_source_files_properties(Src/Backends/CpuMath.cpp PROPERTIES COMPILE_DEFINITIONS ELMAN_HAS_SLEEF)
    target_link_libraries(elman_core PRIVATE "${ELMAN_SLEEF_LIBRARY}")
    if(MSVC)
        set_source_files_properties(Src/Backends/CpuMathAvx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(Src/Backends/CpuMathAvx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
endif()
if(ELMAN_MKL_ROOT)
    find_library(ELMAN_MKL_LIBRARY NAMES mkl_rt HINTS "${ELMAN_MKL_ROOT}/lib" "${ELMAN_MKL_ROOT}/lib/intel64" NO_DEFAULT_PATH NO_CACHE REQUIRED)
    find_path(ELMAN_MKL_INCLUDE mkl.h HINTS "${ELMAN_MKL_ROOT}/include" NO_DEFAULT_PATH NO_CACHE REQUIRED)
    set_source_files_properties(Src/Backends/CpuGemm.cpp PROPERTIES
        INCLUDE_DIRECTORIES "${ELMAN_MKL_INCLUDE}" COMPILE_DEFINITIONS ELMAN_HAS_MKL)
    target_link_libraries(elman_core PRIVATE "${ELMAN_MKL_LIBRARY}")
    if(WIN32)
        file(GLOB ELMAN_MKL_RUNTIME "${ELMAN_MKL_ROOT}/bin/*.dll")
        if(NOT ELMAN_MKL_RUNTIME)
            message(FATAL_ERROR "ELMAN_MKL_ROOT requires runtime DLLs under bin")
        endif()
        add_custom_command(TARGET elman_core POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${ELMAN_TEST_RUNTIME_DIRECTORY}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ELMAN_MKL_RUNTIME} "${ELMAN_TEST_RUNTIME_DIRECTORY}" VERBATIM)
    endif()
endif()
if(WIN32)
    target_link_libraries(elman_core PRIVATE icu ws2_32)
elseif(APPLE)
    find_library(ELMAN_ICU_CORE icucore REQUIRED)
    find_library(ELMAN_COREFOUNDATION CoreFoundation REQUIRED)
    target_link_libraries(elman_core PRIVATE "${ELMAN_ICU_CORE}" "${ELMAN_COREFOUNDATION}")
else()
    find_package(ICU REQUIRED COMPONENTS uc i18n)
    target_link_libraries(elman_core PRIVATE ICU::uc ICU::i18n)
endif()
if(ELMAN_ENABLE_CUDA)
    target_compile_definitions(elman_core PUBLIC ELMAN_HAS_CUDA)
    if(ELMAN_CUDNN_ROOT)
        find_path(ELMAN_CUDNN_INCLUDE cudnn.h HINTS "${ELMAN_CUDNN_ROOT}" PATH_SUFFIXES include NO_DEFAULT_PATH NO_CACHE REQUIRED)
        find_library(ELMAN_CUDNN_LIBRARY NAMES cudnn HINTS "${ELMAN_CUDNN_ROOT}"
            PATH_SUFFIXES lib lib64 lib/x64 NO_DEFAULT_PATH NO_CACHE REQUIRED)
        target_sources(elman_core PRIVATE Src/Backends/CudaConvolution.cu)
        set_source_files_properties(Src/Backends/CudaConvolution.cu PROPERTIES INCLUDE_DIRECTORIES "${ELMAN_CUDNN_INCLUDE}")
        set_source_files_properties(Src/Backends/CudaConvolution.cu
            Src/Operators/CUDA/conv2d_cuda.cu Src/Operators/CUDA/conv2d_input_grad_cuda.cu
            Src/Operators/CUDA/conv2d_weight_grad_cuda.cu PROPERTIES COMPILE_DEFINITIONS ELMAN_HAS_CUDNN)
        target_link_libraries(elman_core PRIVATE "${ELMAN_CUDNN_LIBRARY}")
        if(WIN32)
            file(GLOB ELMAN_CUDNN_RUNTIME "${ELMAN_CUDNN_ROOT}/bin/cudnn*.dll" "${ELMAN_CUDNN_ROOT}/bin/x64/cudnn*.dll")
            if(NOT ELMAN_CUDNN_RUNTIME)
                message(FATAL_ERROR "ELMAN_CUDNN_ROOT requires cuDNN runtime DLLs under bin or bin/x64")
            endif()
            add_custom_command(TARGET elman_core POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${ELMAN_TEST_RUNTIME_DIRECTORY}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ELMAN_CUDNN_RUNTIME} "${ELMAN_TEST_RUNTIME_DIRECTORY}" VERBATIM)
        endif()
    endif()
    if(ELMAN_CUBLAS_ROOT)
        unset(ELMAN_CUBLAS_LIBRARY CACHE)
        unset(ELMAN_CUBLASLT_LIBRARY CACHE)
        find_library(ELMAN_CUBLAS_LIBRARY NAMES cublas HINTS "${ELMAN_CUBLAS_ROOT}/lib" "${ELMAN_CUBLAS_ROOT}/lib/x64" NO_DEFAULT_PATH NO_CACHE REQUIRED)
        find_library(ELMAN_CUBLASLT_LIBRARY NAMES cublasLt HINTS "${ELMAN_CUBLAS_ROOT}/lib" "${ELMAN_CUBLAS_ROOT}/lib/x64" NO_DEFAULT_PATH NO_CACHE REQUIRED)
        target_link_libraries(elman_core PRIVATE "${ELMAN_CUBLAS_LIBRARY}" "${ELMAN_CUBLASLT_LIBRARY}")
        if(WIN32)
            file(GLOB ELMAN_CUBLAS_RUNTIME "${ELMAN_CUBLAS_ROOT}/bin/cublas*.dll" "${ELMAN_CUBLAS_ROOT}/bin/x64/cublas*.dll")
            if(NOT ELMAN_CUBLAS_RUNTIME)
                message(FATAL_ERROR "ELMAN_CUBLAS_ROOT requires cuBLAS runtime DLLs under bin or bin/x64")
            endif()
            add_custom_command(TARGET elman_core POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${ELMAN_TEST_RUNTIME_DIRECTORY}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ELMAN_CUBLAS_RUNTIME} "${ELMAN_TEST_RUNTIME_DIRECTORY}" VERBATIM)
        endif()
    else()
        target_link_libraries(elman_core PRIVATE CUDA::cublas CUDA::cublasLt)
    endif()
endif()

add_library(elman_serving STATIC
    Src/Networking/TcpServer.cpp
    Src/Networking/HttpServer.cpp
    Src/Networking/HttpFiles.cpp
    Src/Serving/Server.cpp
    Src/Serving/GpuMonitor.cpp
    Src/Serving/Http/ChatService.cpp
    Src/Serving/Http/ChatResponse.cpp
    Src/Serving/Http/Http.cpp
    Src/Serving/Http/ImageService.cpp
    Src/Reinforcement/Messages.cpp
    Src/Reinforcement/PolicyValueServer.cpp)
target_link_libraries(elman_serving PUBLIC elman_core PRIVATE Threads::Threads)
if(WIN32)
    target_link_libraries(elman_serving PRIVATE ws2_32)
endif()
if(ELMAN_ENABLE_CUDA)
    find_path(ELMAN_NVML_INCLUDE_DIR nvml.h PATHS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH NO_CACHE)
    if(ELMAN_NVML_INCLUDE_DIR)
        target_compile_definitions(elman_serving PRIVATE ELMAN_HAS_NVML)
        target_include_directories(elman_serving PRIVATE "${ELMAN_NVML_INCLUDE_DIR}")
        target_link_libraries(elman_serving PRIVATE ${CMAKE_DL_LIBS})
    endif()
endif()

add_executable(elman Apps/Main.cpp)
target_link_libraries(elman PRIVATE elman_serving)
if(WIN32)
    foreach(runtime ELMAN_MKL_RUNTIME ELMAN_CUDNN_RUNTIME ELMAN_CUBLAS_RUNTIME)
        if(${runtime})
            add_custom_command(TARGET elman POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${${runtime}} "$<TARGET_FILE_DIR:elman>" VERBATIM)
        endif()
    endforeach()
    set(ELMAN_APP_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Binaries/Windows")
    set_target_properties(elman PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${ELMAN_APP_OUTPUT_DIRECTORY}")
    foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(elman PROPERTIES "RUNTIME_OUTPUT_DIRECTORY_${configuration}" "${ELMAN_APP_OUTPUT_DIRECTORY}")
    endforeach()
    target_link_libraries(elman_serving PRIVATE shell32)
endif()

if(ELMAN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif()
