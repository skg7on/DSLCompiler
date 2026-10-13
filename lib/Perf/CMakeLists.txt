add_llk_library(LLKPerf
    MicroOpDAGBuilders.cpp
    OpDAGBuilder.cpp
    TileInfo.cpp
    MicroDAG.cpp
    MicroCostModel.cpp
    SelectedKernelAnalysis.cpp
    MicroPerfReport.cpp
    SearchSpace.cpp
    Candidate.cpp
    CandidateGenerator.cpp
    Legality.cpp
    CandidateBinding.cpp
    ScheduleRecord.cpp
    TuningSession.cpp
    LINK_LIBRARIES MicroDialect LLKMachine LLKMapping LLK::MLIR LLVM
)
