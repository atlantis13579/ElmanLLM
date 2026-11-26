cmake_minimum_required(VERSION 3.24)
project(Elman VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
get_filename_component(ELMAN_BUILD_VARIANT "${CMAKE_BINARY_DIR}" NAME)

option(ELMAN_ENABLE_CUDA "Build the native CUDA backend" OFF)
option(ELMAN_ENABLE_MKL "Accelerate CPU Float32 operations with oneMKL" OFF)
option(ELMAN_ENABLE_METAL "Build the Apple Metal backend using MPSGraph" ${APPLE})
foreach(legacy_option ELMAN_BUILD_TESTS ELMAN_CUBLAS_ROOT ELMAN_CUDNN_ROOT ELMAN_MKL_ROOT
        ELMAN_SLEEF_ROOT ELMAN_LLM_MODEL_DIR)
    unset(${legacy_option} CACHE)
endforeach()

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
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/MP8> $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()
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

file(GLOB_RECURSE ELMAN_SOURCES CONFIGURE_DEPENDS "Src/*.cpp")
file(GLOB_RECURSE ELMAN_HEADERS CONFIGURE_DEPENDS
    "Src/*.h" "Src/*.hpp" "Src/*.cuh" "Src/*.inc")
# Src/Tools holds standalone command line tools, each with its own entry point.
list(FILTER ELMAN_SOURCES EXCLUDE REGEX "/Src/Tools/")
if(ELMAN_ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_OPERATORS CONFIGURE_DEPENDS "Src/Kernals/CUDA/*.cu")
    list(APPEND ELMAN_SOURCES
        Src/Backends/Cuda.cu
        Src/Backends/CudaExecution.cu
        Src/Backends/CudaGraph.cu
        Src/Backends/CudaGemm.cu
        Src/Backends/CudaGemm_cuBLAS.cu
        ${CUDA_OPERATORS})
    if(MSVC)
        # CCCL requires the conforming MSVC preprocessor in CUDA 13.
        set_source_files_properties(Src/Kernals/CUDA/argsort_cuda.cu Src/Kernals/CUDA/cumsum_cuda.cu
            PROPERTIES COMPILE_OPTIONS "-Xcompiler=/Zc:preprocessor")
        set_source_files_properties(Src/Kernals/CUDA/complex_pointwise_cuda.cu
            PROPERTIES COMPILE_OPTIONS "-Xcompiler=/Zc:preprocessor")
    endif()
endif()

add_library(elman_core SHARED ${ELMAN_SOURCES} ${ELMAN_HEADERS})
add_library(Elman::Core ALIAS elman_core)
set_target_properties(elman_core PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON EXPORT_NAME Core)
find_package(Threads REQUIRED)
target_link_libraries(elman_core PRIVATE Threads::Threads)
target_include_directories(elman_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Elman>)
if(ELMAN_ENABLE_METAL)
    # Objective-C++ backend, fusion compiler and Metal kernels.
    file(GLOB_RECURSE METAL_SOURCES CONFIGURE_DEPENDS "Src/*.mm")
    target_sources(elman_core PRIVATE ${METAL_SOURCES})
    target_compile_options(elman_core PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    target_compile_definitions(elman_core PUBLIC ELMAN_HAS_METAL)
    foreach(framework Foundation Metal MetalPerformanceShaders MetalPerformanceShadersGraph)
        find_library(ELMAN_${framework}_FRAMEWORK ${framework} REQUIRED)
        target_link_libraries(elman_core PRIVATE "${ELMAN_${framework}_FRAMEWORK}")
    endforeach()
endif()
if(ELMAN_ENABLE_MKL)
    find_path(ELMAN_MKL_INCLUDE mkl.h HINTS "$ENV{MKLROOT}/include" "${ELMAN_MKL_INSTALL_ROOT}/include" NO_CACHE REQUIRED)
    get_filename_component(ELMAN_MKL_INSTALL_ROOT "${ELMAN_MKL_INCLUDE}" DIRECTORY)
    set(ELMAN_MKL_INSTALL_ROOT "${ELMAN_MKL_INSTALL_ROOT}" CACHE INTERNAL "Discovered oneMKL installation" FORCE)
    find_library(ELMAN_MKL_LIBRARY NAMES mkl_rt HINTS "${ELMAN_MKL_INSTALL_ROOT}/lib" "${ELMAN_MKL_INSTALL_ROOT}/lib/intel64" NO_CACHE REQUIRED)
    set_source_files_properties(Src/Backends/CpuGemm.cpp PROPERTIES
        INCLUDE_DIRECTORIES "${ELMAN_MKL_INCLUDE}" COMPILE_DEFINITIONS ELMAN_HAS_MKL)
    target_link_libraries(elman_core PRIVATE "${ELMAN_MKL_LIBRARY}")
    if(WIN32)
        # Match the Intel OpenMP layer selected by CpuGemm.cpp. Do not ship the
        # alternate threading layers, debug/stub OpenMP or allocator interposer.
        set(ELMAN_MKL_RUNTIME)
        foreach(component mkl_rt mkl_core mkl_intel_thread mkl_def mkl_vml_def libiomp5md)
            file(GLOB component_runtime CONFIGURE_DEPENDS
                "${ELMAN_MKL_INSTALL_ROOT}/bin/${component}.dll"
                "${ELMAN_MKL_INSTALL_ROOT}/bin/${component}.[0-9]*.dll")
            list(LENGTH component_runtime component_count)
            if(NOT component_count EQUAL 1)
                message(FATAL_ERROR "oneMKL requires exactly one ${component} runtime DLL under ${ELMAN_MKL_INSTALL_ROOT}/bin")
            endif()
            list(APPEND ELMAN_MKL_RUNTIME ${component_runtime})
        endforeach()
        # Dispatch happens at run time. Retain kernels for all supported CPUs,
        # including the VML conditional-reproducibility implementation.
        file(GLOB ELMAN_MKL_KERNEL_RUNTIME CONFIGURE_DEPENDS
            "${ELMAN_MKL_INSTALL_ROOT}/bin/mkl_avx*.dll"
            "${ELMAN_MKL_INSTALL_ROOT}/bin/mkl_mc*.dll"
            "${ELMAN_MKL_INSTALL_ROOT}/bin/mkl_vml_*.dll")
        list(APPEND ELMAN_MKL_RUNTIME ${ELMAN_MKL_KERNEL_RUNTIME})
        list(REMOVE_DUPLICATES ELMAN_MKL_RUNTIME)
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
    get_filename_component(ELMAN_CUDA_LIBRARY_DIR "${CUDA_cublas_LIBRARY}" DIRECTORY)
    find_path(ELMAN_CUDNN_INCLUDE cudnn.h HINTS ${CUDAToolkit_INCLUDE_DIRS} "$ENV{CUDNN_ROOT}/include" NO_CACHE)
    find_library(ELMAN_CUDNN_LIBRARY NAMES cudnn HINTS "${ELMAN_CUDA_LIBRARY_DIR}"
        "$ENV{CUDNN_ROOT}/lib" "$ENV{CUDNN_ROOT}/lib/x64" NO_CACHE)
    if(ELMAN_CUDNN_INCLUDE AND ELMAN_CUDNN_LIBRARY)
        target_sources(elman_core PRIVATE Src/Backends/CudaConvolution.cu)
        set_source_files_properties(Src/Backends/CudaConvolution.cu PROPERTIES INCLUDE_DIRECTORIES "${ELMAN_CUDNN_INCLUDE}")
        set_source_files_properties(Src/Backends/CudaConvolution.cu
            Src/Kernals/CUDA/conv2d_cuda.cu Src/Kernals/CUDA/conv2d_input_grad_cuda.cu
            Src/Kernals/CUDA/conv2d_weight_grad_cuda.cu PROPERTIES COMPILE_DEFINITIONS ELMAN_HAS_CUDNN)
        target_link_libraries(elman_core PRIVATE "${ELMAN_CUDNN_LIBRARY}")
        if(WIN32)
            file(GLOB ELMAN_CUDNN_RUNTIME "${CUDAToolkit_BIN_DIR}/cudnn*.dll"
                "${CUDAToolkit_BIN_DIR}/x64/cudnn*.dll"
                "$ENV{CUDNN_ROOT}/bin/cudnn*.dll" "$ENV{CUDNN_ROOT}/bin/x64/cudnn*.dll")
            if(NOT ELMAN_CUDNN_RUNTIME)
                message(FATAL_ERROR "cuDNN headers and library found, but no runtime DLL is available")
            endif()
            add_custom_command(TARGET elman_core POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${ELMAN_TEST_RUNTIME_DIRECTORY}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ELMAN_CUDNN_RUNTIME} "${ELMAN_TEST_RUNTIME_DIRECTORY}" VERBATIM)
        endif()
    endif()
    target_link_libraries(elman_core PRIVATE CUDA::cublas CUDA::cublasLt)
    if(WIN32)
        # Resolve cuBLAS from the installed CUDA Toolkit through PATH at run time.
        file(GLOB ELMAN_CUBLAS_RUNTIME "${CUDAToolkit_BIN_DIR}/cublas64_*.dll"
            "${CUDAToolkit_BIN_DIR}/x64/cublas64_*.dll")
        file(GLOB ELMAN_CUBLASLT_RUNTIME "${CUDAToolkit_BIN_DIR}/cublasLt64_*.dll"
            "${CUDAToolkit_BIN_DIR}/x64/cublasLt64_*.dll")
        if(NOT ELMAN_CUBLAS_RUNTIME OR NOT ELMAN_CUBLASLT_RUNTIME)
            message(FATAL_ERROR "CUDA requires cuBLAS and cuBLASLt runtime DLLs in the CUDA Toolkit")
        endif()
    endif()
endif()

add_library(elman_serving STATIC
    App/Serving/GenerationScheduler.cpp
    App/Networking/TcpServer.cpp
    App/Networking/HttpServer.cpp
    App/Networking/HttpFiles.cpp
    App/Serving/Server.cpp
    App/Serving/GpuMonitor.cpp
    App/Serving/Http/ChatService.cpp
    App/Serving/Http/ChatResponse.cpp
    App/Serving/Http/Http.cpp
    App/Serving/Http/ImageService.cpp
    App/Reinforcement/Messages.cpp
    App/Reinforcement/PolicyValueServer.cpp
    App/Reinforcement/RoboticsPpoClient.cpp)
target_include_directories(elman_serving PUBLIC "${PROJECT_SOURCE_DIR}/App")
target_link_libraries(elman_serving PUBLIC elman_core PRIVATE Threads::Threads)
if(WIN32)
    target_link_libraries(elman_serving PRIVATE ws2_32)
endif()
if(ELMAN_ENABLE_CUDA)
    target_link_libraries(elman_serving PRIVATE CUDA::cudart_static)
    find_path(ELMAN_NVML_INCLUDE_DIR nvml.h PATHS ${CUDAToolkit_INCLUDE_DIRS} NO_DEFAULT_PATH NO_CACHE)
    if(ELMAN_NVML_INCLUDE_DIR)
        target_compile_definitions(elman_serving PRIVATE ELMAN_HAS_NVML)
        target_include_directories(elman_serving PRIVATE "${ELMAN_NVML_INCLUDE_DIR}")
        target_link_libraries(elman_serving PRIVATE ${CMAKE_DL_LIBS})
    endif()
endif()

add_executable(elman App/Main.cpp
    App/Commands/Diffusion.cpp
    App/Commands/Generate.cpp
    App/Commands/InferTensors.cpp
    App/Commands/Sample.cpp
    App/Commands/Tokenizer.cpp
    App/Commands/Transcribe.cpp
    App/Commands/Train.cpp)
target_link_libraries(elman PRIVATE elman_serving)
set_target_properties(elman PROPERTIES BUILD_RPATH_USE_ORIGIN ON)
if(WIN32)
    foreach(runtime ELMAN_MKL_RUNTIME ELMAN_CUDNN_RUNTIME)
        if(${runtime})
            add_custom_command(TARGET elman POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${${runtime}} "$<TARGET_FILE_DIR:elman>" VERBATIM)
        endif()
    endforeach()
    target_link_libraries(elman_serving PRIVATE shell32)
endif()

# Standalone command line tools. They live outside the library because each one
# provides its own entry point.
add_executable(elman_import_weights Src/Tools/ImportWeights.cpp)
target_link_libraries(elman_import_weights PRIVATE elman_core)
add_executable(elman_quantize_weights Src/Tools/QuantizeWeights.cpp Src/Tools/ToolIO.cpp)
target_link_libraries(elman_quantize_weights PRIVATE elman_core)
if(NOT WIN32)
    target_link_libraries(elman_quantize_weights PRIVATE Threads::Threads)
endif()
add_executable(elman_launch_distributed Src/Tools/LaunchDistributed.cpp)
target_link_libraries(elman_launch_distributed PRIVATE elman_core)
option(ELMAN_ENABLE_ONNX "Build the ONNX graph import tool (requires the ONNX C++ library)" OFF)
if(ELMAN_ENABLE_ONNX)
    find_package(onnx CONFIG REQUIRED)
    add_executable(elman_import_onnx Src/Tools/ImportOnnx.cpp)
    target_compile_definitions(elman_import_onnx PRIVATE ELMAN_HAS_ONNX)
    target_link_libraries(elman_import_onnx PRIVATE elman_core onnx onnx_proto)
endif()

enable_testing()
add_subdirectory(Tests)

install(TARGETS elman_core elman EXPORT ElmanTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
set(ELMAN_RUNTIME_FILENAMES)
if(WIN32)
    foreach(runtime ELMAN_MKL_RUNTIME ELMAN_CUDNN_RUNTIME)
        if(${runtime})
            install(FILES ${${runtime}} DESTINATION ${CMAKE_INSTALL_BINDIR})
            foreach(library IN LISTS ${runtime})
                get_filename_component(filename "${library}" NAME)
                list(APPEND ELMAN_RUNTIME_FILENAMES "${filename}")
            endforeach()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES ELMAN_RUNTIME_FILENAMES)
    configure_file(CMake/PruneRuntime.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/PruneRuntime.cmake" @ONLY)
    add_custom_command(TARGET elman_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} "-DELMAN_RUNTIME_DIRECTORY=${ELMAN_TEST_RUNTIME_DIRECTORY}"
            -P "${CMAKE_CURRENT_BINARY_DIR}/PruneRuntime.cmake" VERBATIM)
    install(CODE "set(ELMAN_RUNTIME_DIRECTORY \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}\")
        include(\"${CMAKE_CURRENT_BINARY_DIR}/PruneRuntime.cmake\")")
endif()
if(ELMAN_ENABLE_MKL AND EXISTS "${ELMAN_MKL_INSTALL_ROOT}/licenses")
    install(DIRECTORY "${ELMAN_MKL_INSTALL_ROOT}/licenses/" DESTINATION ${CMAKE_INSTALL_DATADIR}/Elman/licenses/oneMKL)
endif()
install(DIRECTORY Src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Elman
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inc")
install(DIRECTORY App/Web/ DESTINATION ${CMAKE_INSTALL_DATADIR}/Elman/Web)
install(EXPORT ElmanTargets NAMESPACE Elman:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Elman)
configure_package_config_file(CMake/ElmanConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/ElmanConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Elman
    PATH_VARS CMAKE_INSTALL_BINDIR)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/ElmanConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ElmanConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/ElmanConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Elman)
