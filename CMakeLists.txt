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
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Binaries/${CMAKE_SYSTEM_NAME}/${ELMAN_BUILD_VARIANT}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/Binaries/${CMAKE_SYSTEM_NAME}/${ELMAN_BUILD_VARIANT}")
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
    Src/NN/InferenceLayers.cpp
    Src/Models/Diffusion/Blocks.cpp
    Src/IO/ComponentWeights.cpp
    Src/Distributed/ProcessGroup.cpp
    Src/Distributed/TensorParallel.cpp
    Src/Core/ComplexTensor.cpp
    Src/Core/ComplexPointwise.cpp
    Src/Core/ComplexAutograd.cpp
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
    Src/Models/ModuleModel.cpp
    Src/Models/LLM/Decoder.cpp
    Src/Models/LLM/DeepSeek.cpp
    Src/Models/LLM/DecoderFactory.cpp
    Src/Models/LLM/Qwen2.cpp
    Src/Models/ResNet.cpp
    Src/NN/Module.cpp
    Src/NN/Layers.cpp
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
        add_custom_command(TARGET elman_core POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ELMAN_MKL_RUNTIME} "$<TARGET_FILE_DIR:elman_core>" VERBATIM)
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
            add_custom_command(TARGET elman_core POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ELMAN_CUDNN_RUNTIME} "$<TARGET_FILE_DIR:elman_core>" VERBATIM)
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
            add_custom_command(TARGET elman_core POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ELMAN_CUBLAS_RUNTIME} "$<TARGET_FILE_DIR:elman_core>" VERBATIM)
        endif()
    else()
        target_link_libraries(elman_core PRIVATE CUDA::cublas CUDA::cublasLt)
    endif()
endif()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    Apps/Web/Static/index.html Apps/Web/Static/style.css Apps/Web/Static/app.js
)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/Apps/Web/Static/index.html" ELMAN_WEB_INDEX)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/Apps/Web/Static/style.css" ELMAN_WEB_STYLE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/Apps/Web/Static/app.js" ELMAN_WEB_SCRIPT)
configure_file(Apps/Web/Assets.h.in Apps/Web/Assets.h @ONLY)

add_library(elman_app_services STATIC Apps/ChatService.cpp Apps/GpuMonitor.cpp Apps/ImageService.cpp)
target_include_directories(elman_app_services PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Apps")
target_link_libraries(elman_app_services PUBLIC elman_core PRIVATE Threads::Threads)
if(ELMAN_ENABLE_CUDA)
    find_path(ELMAN_NVML_INCLUDE_DIR nvml.h PATHS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH NO_CACHE)
    if(ELMAN_NVML_INCLUDE_DIR)
        target_compile_definitions(elman_app_services PRIVATE ELMAN_HAS_NVML)
        target_include_directories(elman_app_services PRIVATE "${ELMAN_NVML_INCLUDE_DIR}")
        target_link_libraries(elman_app_services PRIVATE ${CMAKE_DL_LIBS})
    endif()
endif()

add_executable(elman Apps/Main.cpp Apps/Web/ChatResponse.cpp Apps/Web/Http.cpp Apps/Web/Server.cpp)
target_include_directories(elman PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/Apps/Web")
target_link_libraries(elman PRIVATE elman_app_services Threads::Threads)
if(WIN32)
    target_link_libraries(elman PRIVATE ws2_32)
endif()

if(ELMAN_BUILD_TESTS)
    enable_testing()
    set(elman_test_output "${CMAKE_CURRENT_BINARY_DIR}/TestArtifacts")
    set(operator_fixtures "${PROJECT_SOURCE_DIR}/Tests/OperatorCases.json")

    function(add_elman_test_target target source)
        add_executable(${target} Tests/${source})
        target_link_libraries(${target} PRIVATE elman_core ${ARGN})
    endfunction()

    add_elman_test_target(elman_flux2_tests Flux2Test.cpp)
    add_elman_test_target(elman_operator_parity OperatorParity.cpp)
    add_elman_test_target(elman_tensor_tests Float32Test.cpp)
    add_elman_test_target(elman_kernel_tests KernelReference.cpp)
    add_elman_test_target(elman_model_io_tests ModelIO.cpp)
    add_elman_test_target(elman_optimizer_tests OptimizerTest.cpp)
    add_elman_test_target(elman_tokenizer_tests TokenizerTest.cpp)
    add_elman_test_target(elman_storage_tests StorageTest.cpp)
    add_elman_test_target(elman_chat_tests ChatTest.cpp elman_app_services)
    add_elman_test_target(elman_memory_tests MemoryTest.cpp Threads::Threads)
    add_elman_test_target(elman_cache_blocks CacheBlocksTest.cpp Threads::Threads)
    add_elman_test_target(elman_paged_cache PagedKVCacheTest.cpp Threads::Threads)
    add_elman_test_target(elman_shared_cache SharedCacheTest.cpp Threads::Threads)
    add_elman_test_target(elman_prefix_cache PrefixCacheTest.cpp Threads::Threads)
    add_elman_test_target(elman_indexed_copy IndexedCopyTest.cpp)
    add_elman_test_target(elman_pool_tests AllocationPoolTest.cpp Threads::Threads)
    add_elman_test_target(elman_view_tests ContiguousViewTest.cpp)
    add_elman_test_target(elman_attention_tests TiledAttentionTest.cpp)
    add_elman_test_target(elman_execution_tests ExecutionTest.cpp Threads::Threads)
    add_elman_test_target(elman_staging_tests StagingTest.cpp Threads::Threads)
    add_elman_test_target(elman_float8_tests Float8Test.cpp)
    add_elman_test_target(elman_storage_layout StorageLayoutTest.cpp)
    add_elman_test_target(elman_text_stream_tests TextStreamTest.cpp)
    add_elman_test_target(elman_float64_tests Float64Test.cpp)
    add_elman_test_target(elman_integer_tests IntegerTest.cpp)
    if(MSVC)
        target_compile_options(elman_text_stream_tests PRIVATE /utf-8)
    endif()

    set(cpu_cuda_test_backends cpu)
    set(gpu_test_backends)
    if(ELMAN_ENABLE_CUDA)
        list(APPEND cpu_cuda_test_backends cuda)
        list(APPEND gpu_test_backends cuda)
        target_sources(elman_execution_tests PRIVATE Tests/CudaExecutionProbe.cu)
        target_sources(elman_staging_tests PRIVATE Tests/CudaStagingProbe.cu)
    endif()
    if(ELMAN_ENABLE_METAL)
        list(APPEND gpu_test_backends metal)
    endif()

    foreach(backend cpu ${gpu_test_backends})
        add_test(NAME flux2_${backend} COMMAND elman_flux2_tests ${backend} "${elman_test_output}/Flux2/${backend}")
        set_tests_properties(flux2_${backend} PROPERTIES SKIP_RETURN_CODE 77 LABELS "diffusion;operator" TIMEOUT 180)
    endforeach()
    add_test(NAME operator_fixtures_cpu COMMAND elman_operator_parity "${operator_fixtures}" cpu)
    foreach(backend IN LISTS gpu_test_backends)
        add_test(NAME operators_cpu_${backend} COMMAND elman_operator_parity "${operator_fixtures}" ${backend})
        set_tests_properties(operators_cpu_${backend} PROPERTIES SKIP_RETURN_CODE 77 LABELS "${backend};operator" TIMEOUT 600)
    endforeach()

    foreach(backend IN LISTS cpu_cuda_test_backends)
        add_test(NAME memory_${backend} COMMAND elman_memory_tests ${backend})
        add_test(NAME prefix_cache_${backend} COMMAND elman_prefix_cache ${backend})
        add_test(NAME shared_cache_${backend} COMMAND elman_shared_cache ${backend})
        add_test(NAME paged_cache_${backend} COMMAND elman_paged_cache ${backend})
        add_test(NAME indexed_copy_${backend} COMMAND elman_indexed_copy ${backend})
        add_test(NAME allocation_pool_${backend} COMMAND elman_pool_tests ${backend})
        add_test(NAME contiguous_view_${backend} COMMAND elman_view_tests ${backend})
        add_test(NAME attention_memory_${backend} COMMAND elman_attention_tests ${backend})
        add_test(NAME execution_${backend} COMMAND elman_execution_tests ${backend})
        add_test(NAME host_staging_${backend} COMMAND elman_staging_tests ${backend})
        add_test(NAME storage_layout_${backend} COMMAND elman_storage_layout ${backend})
        add_test(NAME storage_${backend} COMMAND elman_storage_tests ${backend})
        set_tests_properties(prefix_cache_${backend} PROPERTIES LABELS "llm;memory;paged-cache;prefix" RUN_SERIAL TRUE TIMEOUT 180)
        set_tests_properties(shared_cache_${backend} PROPERTIES LABELS "llm;memory;paged-cache" RUN_SERIAL TRUE TIMEOUT 180)
        set_tests_properties(paged_cache_${backend} indexed_copy_${backend}
            PROPERTIES LABELS "llm;memory;paged-cache" RUN_SERIAL TRUE TIMEOUT 120)
        set_tests_properties(contiguous_view_${backend} PROPERTIES LABELS "llm;memory;views" RUN_SERIAL TRUE TIMEOUT 120)
        set_tests_properties(attention_memory_${backend} PROPERTIES LABELS "llm;memory;fusion" RUN_SERIAL TRUE TIMEOUT 120)
        set_tests_properties(execution_${backend} PROPERTIES TIMEOUT 60)
        set_tests_properties(host_staging_${backend} PROPERTIES LABELS "memory;execution" TIMEOUT 60)
        set_tests_properties(storage_layout_${backend} PROPERTIES LABELS "dtype;storage" TIMEOUT 60)

        add_test(NAME float64_${backend} COMMAND elman_float64_tests ${backend} "${elman_test_output}/Float64/${backend}")
        add_test(NAME integer_${backend} COMMAND elman_integer_tests ${backend})
        add_test(NAME float8_${backend} COMMAND elman_float8_tests ${backend} --suite "${elman_test_output}/Float8/${backend}")
        add_test(NAME float16_${backend} COMMAND elman_float8_tests ${backend} --float16)
        add_test(NAME floating_random_${backend} COMMAND elman_float8_tests ${backend} --floating-random)
        add_test(NAME tensor_${backend} COMMAND elman_tensor_tests ${backend})
        add_test(NAME kernels_${backend} COMMAND elman_kernel_tests ${backend})
        add_test(NAME optimizer_${backend} COMMAND elman_optimizer_tests ${backend})
        set_tests_properties(float64_${backend} integer_${backend} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
        set_tests_properties(float8_${backend} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300 RUN_SERIAL TRUE
            LABELS "dtype;float8;float16;float32;lowprecision;optimizer;checkpoint;random;dropout;attention;sampling")
        set_tests_properties(float16_${backend} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60 LABELS "dtype;float16")
        set_tests_properties(floating_random_${backend} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180 RUN_SERIAL TRUE
            LABELS "dtype;float32;float16;random")
        set_tests_properties(tensor_${backend} kernels_${backend} optimizer_${backend} PROPERTIES SKIP_RETURN_CODE 77)
        set_tests_properties(float64_${backend} PROPERTIES LABELS "dtype;float64")
        set_tests_properties(integer_${backend} PROPERTIES LABELS "dtype;integer")
        set_property(TEST tensor_${backend} kernels_${backend} optimizer_${backend} APPEND PROPERTY LABELS float32)
        set_property(TEST kernels_${backend} APPEND PROPERTY LABELS float16)
    endforeach()

    if(ELMAN_ENABLE_CUDA)
        set_tests_properties(memory_cuda paged_cache_cuda indexed_copy_cuda allocation_pool_cuda contiguous_view_cuda
            attention_memory_cuda execution_cuda host_staging_cuda storage_layout_cuda storage_cuda
            PROPERTIES SKIP_RETURN_CODE 77)
        set_tests_properties(host_staging_cuda storage_layout_cuda PROPERTIES RUN_SERIAL TRUE TIMEOUT 120)
        set_tests_properties(execution_cuda PROPERTIES RUN_SERIAL TRUE)
    endif()

    add_test(NAME cache_blocks COMMAND elman_cache_blocks)
    set_tests_properties(cache_blocks PROPERTIES LABELS "llm;memory;paged-cache" TIMEOUT 60)
    add_test(NAME text_stream COMMAND elman_text_stream_tests)
    add_test(NAME tokenizer COMMAND elman_tokenizer_tests)
    add_test(NAME model_io COMMAND elman_model_io_tests)
    if(ELMAN_LLM_MODEL_DIR)
        add_test(NAME chat_output COMMAND elman_chat_tests "${ELMAN_LLM_MODEL_DIR}")
        set_tests_properties(chat_output PROPERTIES LABELS "text" RUN_SERIAL TRUE TIMEOUT 60)
    endif()
endif()
