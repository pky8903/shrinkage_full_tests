cmake_minimum_required(VERSION 3.18)
project(shrinkage_full_tests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# ── External dependencies (override via -DCUFFTDX_DIR=... / -DCUTLASS_DIR=...) ──
set(CUFFTDX_DIR  "/home/kyp/Workspace/01.libraries/nvidia-mathdx-25.01.1/nvidia/mathdx/25.01"
    CACHE PATH "Path to NVIDIA MathDx (contains include/cufftdx.hpp)")
set(CUTLASS_DIR  "/home/kyp/Workspace/01.libraries/cutlass"
    CACHE PATH "Path to CUTLASS (contains include/cutlass/)")

find_package(CUDAToolkit REQUIRED)

# ── Shared sources ────────────────────────────────────────────────────────────
set(SHRINKAGE_PATTERN_SRC third_party/src/patternUtil/test_pattern.cu)
set(SHRINKAGE_CD_SRC
    third_party/src/patternUtil/test_pattern.cu
    third_party/src/metricUtil/cd_measure_util.cu)

# ── Executable definitions ─────────────────────────────────────────────────
#   shrinkage_adjoint_test           : adjoint/main_adjoint.cu
#   shrinkage_tan_tanadj_test        : tan_tanadj/main_tan_tanadj.cu
#   shrinkage_forward_training_test  : training/main_forward_training.cu  (needs cuDNN)
#   batch_demo                       : training/main_batch_demo.cu         (needs cuDNN)
#   compare_demo                     : training/main_compare.cu            (needs cuDNN)
#   cd_training_test                 : training/main_cd_training.cu        (needs cuDNN)
add_executable(shrinkage_adjoint_test          adjoint/main_adjoint.cu           ${SHRINKAGE_PATTERN_SRC})
add_executable(shrinkage_tan_tanadj_test       tan_tanadj/main_tan_tanadj.cu     ${SHRINKAGE_PATTERN_SRC})
add_executable(shrinkage_forward_training_test training/main_forward_training.cu ${SHRINKAGE_PATTERN_SRC})
add_executable(batch_demo                      training/main_batch_demo.cu       ${SHRINKAGE_PATTERN_SRC})
add_executable(compare_demo                    training/main_compare.cu          ${SHRINKAGE_PATTERN_SRC})
add_executable(cd_training_test                training/main_cd_training.cu      ${SHRINKAGE_CD_SRC})

set(SHRINKAGE_TARGETS
    shrinkage_adjoint_test
    shrinkage_tan_tanadj_test
    shrinkage_forward_training_test
    batch_demo
    compare_demo
    cd_training_test)

foreach(_tgt ${SHRINKAGE_TARGETS})
    target_include_directories(${_tgt} PRIVATE
        ${CUFFTDX_DIR}/include
        ${CUTLASS_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/include
        /usr/local/cuda/include
        /usr/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/adjoint
        ${CMAKE_CURRENT_SOURCE_DIR}/tan_tanadj
        ${CMAKE_CURRENT_SOURCE_DIR}/training
    )

    target_compile_definitions(${_tgt} PRIVATE
        API_TESTER_CUDA
        CUFFT_TARGET_ARCHS=860
    )

    target_compile_options(${_tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-extended-lambda
            --expt-relaxed-constexpr
        >
    )

    target_link_libraries(${_tgt} PRIVATE
        CUDA::cudart
        CUDA::cufft
        -lnvToolsExt
    )

    set_target_properties(${_tgt} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()

# cuDNN required only by the training/batch-demo/compare/cd targets.
target_link_libraries(shrinkage_forward_training_test PRIVATE cudnn)
target_link_libraries(batch_demo                      PRIVATE cudnn)
target_link_libraries(compare_demo                    PRIVATE cudnn)
target_link_libraries(cd_training_test                PRIVATE cudnn)
