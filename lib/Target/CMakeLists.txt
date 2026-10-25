add_llk_library(LLKTargetX86
    X86/TargetAVX2.cpp
)

add_llk_library(LLKTargetAArch64
    AArch64/TargetSVE.cpp
)

add_llk_library(LLKTargetMapping
    MappingTargets.cpp
    X86/Mapping/AVX2MappingTarget.cpp
    X86/Mapping/AVX2BundleLowering.cpp
    X86/Mapping/AVX2BackendLowering.cpp
    GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.cpp
    AArch64/Mapping/SVEMappingTarget.cpp
    AArch64/Mapping/SVEBundleLowering.cpp
    AArch64/Mapping/SVEBackendLowering.cpp
    AArch64/Mapping/SMEMappingTarget.cpp
    AArch64/Mapping/SMEBundleLowering.cpp
    AArch64/Mapping/SMEVectorLowering.cpp
    AArch64/Mapping/SMEArmLowering.cpp
    LINK_LIBRARIES LLKMapping LLKMachine MicroDialect LLK::MLIR
)
