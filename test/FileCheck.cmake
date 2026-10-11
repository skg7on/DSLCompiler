# IR round-trip, FileCheck, CLI, and verifier tests.
# FileCheck tests require llk-opt and must use the FileCheck built alongside
# the LLVM/MLIR packages consumed by this project. Clear the cached result on
# every configure so changing LLVM_PROJECT_BUILD_DIR cannot retain FileCheck
# from an older or unrelated LLVM build.
if(LLK_BUILD_TOOLS)
  unset(FILECHECK_BIN CACHE)
  unset(FILECHECK_BIN)
  find_program(FILECHECK_BIN NAMES FileCheck
      HINTS "${LLVM_TOOLS_BINARY_DIR}"
      NO_DEFAULT_PATH)
  if(NOT FILECHECK_BIN)
    message(FATAL_ERROR
        "FileCheck is required when LLK_BUILD_TOOLS=ON, but it was not found "
        "in LLVM_TOOLS_BINARY_DIR (${LLVM_TOOLS_BINARY_DIR}). Rebuild LLVM "
        "with -DLLVM_BUILD_UTILS=ON.")
  endif()

  function(add_llk_filecheck_test test_name input_file)
      # Use sh -c to properly handle input redirection (CMake's add_test
      # COMMAND mode does not support < redirection natively).
      # Execute the RUN line: llk-opt %s | llk-opt | FileCheck %s
      # This round-trips the IR through llk-opt twice to verify parse+print stability.
      # Arguments after the input file are placed on the first llk-opt
      # invocation only, so a test can select a pass; the second invocation
      # stays the bare parse/print round-trip.
      add_test(NAME ${test_name}
          COMMAND sh -c "$<TARGET_FILE:llk-opt> ${ARGN} ${CMAKE_SOURCE_DIR}/${input_file} | $<TARGET_FILE:llk-opt> | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/${input_file}"
      )
  endfunction()

  add_llk_filecheck_test(DialectOps test/Dialect/ops.mlir)
  add_llk_filecheck_test(MicroDialectOps test/Dialect/Micro/ops.mlir)
  add_llk_filecheck_test(MicroDialectTileOps test/Dialect/Micro/tile_ops.mlir)
  add_llk_filecheck_test(MicroDialectSearchSpace test/Dialect/Micro/search_space.mlir)
  add_llk_filecheck_test(MicroMappingSearchBinding test/Conversion/MicroMapping/search_binding.mlir)

  # MicroMappingMap: the --micro-map pass runs the mapping chain (extract ->
  # search -> bind) and writes the selected plan's metadata onto the kernel.
  # The pass options are quoted so sh hands llk-opt one argv token; the whole
  # option string is the user-facing contract from design §21.
  add_llk_filecheck_test(MicroMappingMap test/Conversion/MicroMapping/micro_map.mlir
      "\"--micro-map=target=x86-avx2 machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul mode=deterministic\"")

  # MicroMappingVerify: phase 2 of the layered verification (design §18.3).
  # The pipeline chains `--micro-map` (which binds a plan and records its ids)
  # into `--micro-verify-mapping` (which loads the same target and resolves every
  # recorded id). A successful verify leaves the IR byte-identical, so the
  # FileCheck assertions are the plan/mapping metadata MicroMappingMap checks --
  # a verifier that mutated would have to break them. Raw add_test (rather than
  # add_llk_filecheck_test) because two passes are chained with their own option
  # strings; the helper's ARGN list would hand sh a command separator.
  add_test(NAME MicroMappingVerify
      COMMAND sh -c "$<TARGET_FILE:llk-opt> \"--micro-map=target=x86-avx2 machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul mode=deterministic\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/verify_mapping.mlir | $<TARGET_FILE:llk-opt> \"--micro-verify-mapping=target=x86-avx2 machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul\" | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/verify_mapping.mlir"
  )

  # MicroMappingVerifyInvalid: the negative half of phase 2. Each chunk of the
  # fixture is mapped IR with one recorded id that does not resolve (or a module
  # with no kernel at all), and `--verify-diagnostics` asserts the stable §22.3
  # code *and* the specific detail on the diagnostic. The same test also pins
  # that a fully resolved chunk produces no diagnostic.
  add_test(NAME MicroMappingVerifyInvalid
      COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
              "--micro-verify-mapping=target=x86-avx2 machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul"
              ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/verify_mapping_invalid.mlir
  )

  # MicroMappingUnmaterialized: the canonical tile-transfer fixture is now a
  # *materialized* plan (task B4): the CLI supplies the canonical
  # PlanMaterializer, so the vector->tile-store edge becomes a typed
  # `micro.tile_async_copy` + `micro.wait` into the store's memory, and the
  # store's operand is rewired to the destination-memory tile. The fixture
  # runs with require-executable=1, so a warning about an unmaterialized
  # connection is itself a failure. stderr is merged because the pass's
  # diagnostics go there while FileCheck reads stdout. Raw add_test because
  # add_llk_filecheck_test() pipes stdout only and cannot set --check-prefix.
  add_test(NAME MicroMappingUnmaterialized
      COMMAND sh -c "$<TARGET_FILE:llk-opt> \"--micro-map=target=probe machine=${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/probe_machine.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/unmaterialized_rules.llkmap emitters=e1 mode=deterministic require-executable=1\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_map_unmaterialized.mlir 2>&1 | ${FILECHECK_BIN} --check-prefix=MAP ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_map_unmaterialized.mlir"
  )

  # MicroMappingRequireExecutable: the executable binding contract (design
  # §18.2) is satisfied by the canonical materializer -- the same plan is now
  # fully executable, so the backend-facing run succeeds and emits the
  # materialized tile movement rather than refusing an omitted decision. A
  # null materializer still fails Executable (covered by the unit tests in
  # MappingPlanBinderTest), but the CLI always supplies one.
  add_test(NAME MicroMappingRequireExecutable
      COMMAND sh -c "$<TARGET_FILE:llk-opt> \"--micro-map=target=probe machine=${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/probe_machine.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/unmaterialized_rules.llkmap emitters=e1 mode=deterministic require-executable=1\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_map_unmaterialized.mlir 2>&1 | grep -q 'micro.tile_async_copy'"
  )

  # MicroMappingBarrier: the storage plan (task B3) is now run on the selected
  # plan (task C1), so a movement that needs more than one transfer engine is
  # barrier-synchronized and the canonical materializer emits a `micro.barrier`.
  # The fixture routes SRAM -> L2 -> DRAM (two links, two engines) on
  # barrier_machine.yaml, so `requiresBarrier` is set and the barrier is
  # observable in the IR. Raw add_test because FileCheck needs --check-prefix and
  # stderr must be merged (the pass's diagnostics go to stderr).
  add_test(NAME MicroMappingBarrier
      COMMAND sh -c "$<TARGET_FILE:llk-opt> \"--micro-map=target=barrier machine=${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/barrier_machine.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/unmaterialized_rules.llkmap emitters=e1 mode=deterministic require-executable=1\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/barrier_sync.mlir 2>&1 | ${FILECHECK_BIN} --check-prefix=MAP ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/barrier_sync.mlir"
  )

  # MicroMappingBarrierStorage: the reported and verified half of C1 -- the
  # report's selectedState carries populated allocations, synchronization and the
  # plan-step DAG, and B6's barrier verification fires (deleting the barrier
  # makes verification fail). A script because it compares the IR, the report
  # file, and two verification outcomes.
  add_test(NAME MicroMappingBarrierStorage
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/barrier_storage.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingBarrierStorage
  )

  # MicroMappingReport: the `report=<path>` option's contract -- the report file
  # is produced, the mapped IR is byte-identical with and without it (the report
  # is metadata), and two separate llk-opt runs produce identical reports. A
  # script because FileCheck only speaks about one stdout stream, while this
  # compares two IR outputs and two report files.
  add_test(NAME MicroMappingReport
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_map_report.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingReport
  )

  # MicroMappingReportOnly: `report-only=1` runs the search and writes the
  # report but leaves the module untouched (design §21: "emitting a plan report
  # without modifying input IR"). A script because the assertions span two
  # files -- the report and the printed IR -- and one of them is a diff of the
  # output against the input kernel itself, byte for byte.
  add_test(NAME MicroMappingReportOnly
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_map_report_only.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingReportOnly
  )

  # MicroMappingCandidate: `candidate=<sym>` drives `--micro-map` (and, for the
  # replay half, `--micro-bind-plan`) from a persistent `micro.candidate`
  # (phase-4 task 4). The candidate's values pin rule parameters and its
  # `layout`-kind parameter -- resolved by kind, not by name -- selects the
  # bound layout, so the plan the pass binds is traceable to the search-space
  # point it came from. Because the plan id folds in the binding's hash, the
  # round trip needs the same `candidate=` on both runs (ruling S8); this test
  # proves it, and that the id is genuinely absent from a binding-free replay.
  # A script because the binding's hash is read back out of the report file and
  # compared against the bound IR, and because the rejection and namespace cases
  # are separate invocations whose diagnostics are asserted.
  add_test(NAME MicroMappingCandidate
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_map_candidate.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingCandidate
  )

  # MicroMappingBindPlanId: `--micro-bind-plan` accepts the id the report prints
  # (bare hex), plus 0x-hex and signed/unsigned decimal, and rejects anything
  # else. End-to-end because the id is obtained by running `--micro-map report=`
  # first -- the documented report -> bind workflow. A script because the id is
  # read out of a file and fed back into a second invocation.
  add_test(NAME MicroMappingBindPlanId
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_bind_plan_id.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingBindPlanId
  )

  # MicroMappingBindPlanMode: a plan id is a content hash, so reproducing one
  # means replaying the search -- same mode, beam-width, and top-k. This pins
  # that `--micro-bind-plan` takes those options and honours them: a beam id
  # round-trips in beam mode, and a deterministic id is absent from a beam
  # search with the same cap (the diagnostic names the mode it searched).
  add_test(NAME MicroMappingBindPlanMode
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_bind_plan_mode.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingBindPlanMode
  )

  # MicroMappingBindPlanRoundtrip: the strongest form of the report -> bind
  # contract -- binding the reported id reproduces the *same* plan, not merely
  # *a* plan, so the bound IR is byte-identical to the `--micro-map report=`
  # output. The id-parsing and mode tests above prove the id is accepted and the
  # search is replayed; this one proves the replayed search selects the plan the
  # report named. Three cases: deterministic and beam over the hand-authored
  # kernel, and exact over IR that `--llk-to-micro` produced, so the round-trip
  # is covered on compiler-generated IR too. A script because it compares two
  # IR streams the way FileCheck cannot.
  #
  # The ids here are all binding-free, so the replay passes no `candidate=`. A
  # plan id folds the search point in, so a binding-derived id needs the same
  # `candidate=` on both runs; MicroMappingCandidate covers that round trip
  # (ruling S8).
  add_test(NAME MicroMappingBindPlanRoundtrip
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_bind_plan_roundtrip.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingBindPlanRoundtrip
  )

  # MicroMappingMatmulE2E / MicroMappingSwiGLUE2E: the gap-#3 acceptance. Each
  # runs the two-pass pipeline `--llk-to-micro | --micro-map mode=exact` over
  # the LLK source, so the kernel being mapped is genuinely compiler-generated
  # (the input is the same LLK program as the matching LLKToMicro fixture), and
  # asserts the shipped AVX2 target binds a complete plan onto it: a
  # `micro.plan` on the kernel and a `micro.mapping` on every covered node,
  # including the `micro.vector` op variants the lowering emits and the
  # `micro.tile_async_copy`/`micro.tile_store` movement ops.
  add_test(NAME MicroMappingMatmulE2E
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/matmul_e2e.mlir | $<TARGET_FILE:llk-opt> \"--micro-map=target=x86-avx2 machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul mode=exact\" | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/matmul_e2e.mlir"
  )
  # MicroMappingMatmulTraverses / MicroMappingSwiGLUTraverses: the stage-C3
  # acceptance. The same compiler-generated kernel that the mapping tests above
  # map is now pushed through the whole bridge, and the conversion has to
  # finish -- `scf.for`, `linalg.matmul`, and the write-back as an
  # `tensor.insert_slice` threaded through the spatial loops -- with no residual
  # micro operation for a backend to trip over. Traversal is what this proves;
  # numerical correctness is C4/C8's.
  # Stage C8: the design's acceptance chains, driven through the tools' public
  # flags only. Whether a schedule round-trips, whether a plan report is
  # byte-identical across two searches, whether the performance model accepts
  # what the mapper wrote -- none of that is visible from inside one process,
  # which is why this is a runner and not a unit test.
  add_test(NAME MicroMappingAcceptancePipeline
      COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/acceptance_pipeline.py
          $<TARGET_FILE:llk-opt> $<TARGET_FILE:llk-compile> $<TARGET_FILE:micro-perf>
          ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/AcceptancePipeline
  )
  # Stage C10: the workflow and acceptance documents must point at paths that
  # exist and flags the tools accept, so an example a reader is told to run is
  # one the tools can actually run.
  add_test(NAME DocReferences
      COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Docs/check_doc_references.py
          ${CMAKE_SOURCE_DIR}
          $<TARGET_FILE:llk-opt> $<TARGET_FILE:llk-compile>
          $<TARGET_FILE:llk-tune> $<TARGET_FILE:micro-perf>
  )
  add_test(NAME WorkflowSmoke
      COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Docs/workflow_smoke.py
          ${CMAKE_SOURCE_DIR}/test/Docs/workflow_smoke_manifest.json
          ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
          $<TARGET_FILE:llk-opt> $<TARGET_FILE:llk-compile>
          $<TARGET_FILE:llk-tune> $<TARGET_FILE:micro-perf>
  )
  add_test(NAME WorkflowSmokeNegativeControls
      COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Docs/workflow_smoke_test.py
          ${CMAKE_SOURCE_DIR}/test/Docs/workflow_smoke_manifest.json
          ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/test/Docs/workflow_smoke.py
          $<TARGET_FILE:llk-opt> $<TARGET_FILE:llk-compile>
          $<TARGET_FILE:llk-tune> $<TARGET_FILE:micro-perf>
  )
  add_test(NAME DocReferencesUnit
      COMMAND python3 -m unittest discover
          -s ${CMAKE_SOURCE_DIR}/test/Docs -p test_*.py
  )
  # Run the community examples: help-name lint alone cannot validate pass
  # option values, file arguments, or the artifacts shown in a tutorial.
  add_test(NAME CommunityTutorials
      COMMAND python3 ${CMAKE_SOURCE_DIR}/docs/manual/source/tutorials/run_tutorials.py
          --build-dir $<TARGET_FILE_DIR:llk-opt>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/CommunityTutorials
  )
  # Stage C5: the shared mapped-compilation path, taken by the compiler. A
  # plan is searched, bound, verified, handed to the target's own emitters where
  # they implement it, lowered, bufferized and JIT-compiled -- and the compiler
  # reports what ran next to what was selected, so "the target lowered this"
  # and "the reference bridge carried it" stay separate claims.
  add_test(NAME MicroMappingCompileMatmulToExecutable
      COMMAND sh -c "$<TARGET_FILE:llk-compile> --mapping-target=x86-avx2 --mapping-root=${CMAKE_SOURCE_DIR} --mapping-mode=exact ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/matmul_e2e.mlir | grep -q 'Compilation successful'"
  )
  add_test(NAME MicroMappingCompileSwiGLUToExecutable
      COMMAND sh -c "$<TARGET_FILE:llk-compile> --mapping-target=x86-avx2 --mapping-root=${CMAKE_SOURCE_DIR} --mapping-mode=exact ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/swiglu_e2e.mlir | grep -q 'Compilation successful'"
  )
  # The inspection stops: a caller that wants the mapped or target-lowered IR
  # has no business running a JIT to get it.
  add_test(NAME MicroMappingCompileStopsAtMappedMicro
      COMMAND sh -c "$<TARGET_FILE:llk-compile> --mapping-target=x86-avx2 --mapping-root=${CMAKE_SOURCE_DIR} --mapping-mode=exact --mapping-stop=mapped-micro ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/matmul_e2e.mlir | grep -q 'micro.kernel'"
  )
  add_test(NAME MicroMappingCompileStopsAtTargetLowered
      COMMAND sh -c "$<TARGET_FILE:llk-compile> --mapping-target=x86-avx2 --mapping-root=${CMAKE_SOURCE_DIR} --mapping-mode=exact --mapping-stop=target-lowered ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/matmul_e2e.mlir | grep -q 'micro.kernel'"
  )
  add_test(NAME FusedSelectedLowering
      COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/fused_selected_lowering.py
          $<TARGET_FILE:llk-opt> $<TARGET_FILE:llk-compile> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/FusedSelectedLowering
  )
  add_test(NAME SelectedVectorWidth
      COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/selected_vector_width.py
          $<TARGET_FILE:llk-compile> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/SelectedVectorWidth
  )
  add_test(NAME SelectedContraction
      COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/selected_contraction.py
          $<TARGET_FILE:llk-opt> $<TARGET_FILE:llk-compile> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/SelectedContraction
  )
  add_test(NAME TailPadExport
      COMMAND python3 ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/tail_pad_export.py
          $<TARGET_FILE:llk-opt> $<TARGET_FILE:llk-compile> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/TailPadExport
  )
  # A target nobody registered is a diagnostic naming the ones that are, not a
  # silent fallback to a different backend.
  add_test(NAME MicroMappingCompileRejectsUnknownTarget
      COMMAND sh -c "$<TARGET_FILE:llk-compile> --mapping-target=not-a-target --mapping-root=${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/matmul_e2e.mlir 2>&1 | grep -q 'no mapping target is registered'"
  )
  add_test(NAME MicroMappingMatmulTraverses
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/matmul_e2e.mlir | $<TARGET_FILE:llk-opt> --micro-to-linalg | ${FILECHECK_BIN} --check-prefix=TRAVERSE ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/matmul_e2e.mlir"
  )
  add_test(NAME MicroMappingSwiGLUTraverses
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/swiglu_e2e.mlir | $<TARGET_FILE:llk-opt> --micro-to-linalg | ${FILECHECK_BIN} --check-prefix=TRAVERSE ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/swiglu_e2e.mlir"
  )
  add_test(NAME MicroMappingSwiGLUE2E
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/swiglu_e2e.mlir | $<TARGET_FILE:llk-opt> \"--micro-map=target=x86-avx2 machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul mode=exact\" | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/swiglu_e2e.mlir"
  )

  # LLKToMicro lowering tests. The schedule database is pointed at a path that
  # does not exist on purpose: the export then uses its built-in conservative
  # schedule, so the expected tile sizes do not depend on the working directory
  # or on the contents of schedules/schedule_db.json.
  add_test(NAME LLKToMicroSwiGLU
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/swiglu_to_micro.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/swiglu_to_micro.mlir"
  )
  add_test(NAME LLKToMicroMatmul
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/matmul_to_micro.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/matmul_to_micro.mlir"
  )
  # The search-space export, anchored on a fixture database that spells out the
  # Micro/tile schedule fields for one operation and leaves the other on the
  # pre-M11 fields, so one run covers both loaded and defaulted values.
  add_test(NAME LLKToMicroSearchSpace
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro-search-space=\"schedule-db=${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_schedule.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_from_schedule.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_from_schedule.mlir"
  )
  # The M-bucket rules prune BM choices: bucket 0 allows only BM = 1, and the
  # large-M buckets reject tiles of 4 rows or fewer.
  add_test(NAME LLKToMicroSearchSpaceMBucket
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro-search-space=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_m_bucket.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_m_bucket.mlir"
  )
  # `llk-compile --emit=micro-search` prints the search space and exits without
  # reaching the JIT: the input's llk ops are never lowered to LLVM.
  add_test(NAME LLKCompileMicroSearch
      COMMAND sh -c "$<TARGET_FILE:llk-compile> --emit=micro-search ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_from_schedule.mlir | ${FILECHECK_BIN} --check-prefix=COMPILE ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_from_schedule.mlir"
  )
  # The acceptance check that the exported Micro-IR is self-describing: the
  # performance simulator evaluates the kernel the export produced, using only
  # the metadata carried by the IR.
  add_test(NAME LLKToMicroPerfRoundTrip
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/swiglu_to_micro.mlir | $<TARGET_FILE:micro-perf> --machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml --level=1 - | ${FILECHECK_BIN} --check-prefix=PERF ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/swiglu_to_micro.mlir"
  )
  # LLKToLinalgConversion test: run llk-opt --llk-to-linalg then FileCheck
  add_test(NAME LLKToLinalgConversion
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg.mlir"
  )

  # MicroToLinalg: the bridge from canonical Micro-IR to the backend pipeline.
  # A `micro.kernel` becomes a `func.func` over memrefs and Linalg, which is
  # what lets a mapped or exported kernel reach LLVM and the JIT.
  add_test(NAME MicroToLinalg
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --micro-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/MicroToLinalg/micro_to_linalg.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/MicroToLinalg/micro_to_linalg.mlir"
  )
  # MicroToLinalgInvalid: an op with no lowering must fail the conversion
  # loudly rather than being silently left in the module. Each rejection needs
  # its own chunk: the conversion aborts at the first illegal op, so one module
  # can only ever report one of them.
  add_test(NAME MicroToLinalgInvalid
      COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
              --micro-to-linalg
              ${CMAKE_SOURCE_DIR}/test/Conversion/MicroToLinalg/micro_to_linalg_invalid.mlir
  )
  # MicroToLinalgJit: the lowered kernel reaches the JIT. This is the check that
  # the bridge produces IR the backend pipeline can finish -- which is why the
  # lowering stays in tensor land: an already-bufferized module leaves an
  # unrealized vector cast that LLVM translation rejects.
  add_test(NAME MicroToLinalgJit
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --micro-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/MicroToLinalg/micro_to_linalg_jit.mlir | $<TARGET_FILE:llk-compile> - | grep -q 'Compilation successful'"
  )
  # math_mode=triton_fast on the SiLU path: the sigmoid's exp must take the
  # bounded-fast (math.exp2) path, like bounded_fast and unsafe_fast do.
  add_test(NAME LLKToLinalgTritonFast
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg_triton_fast.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg_triton_fast.mlir"
  )
  # RoPE lowering: verify llk.rope → bounded-fast trig + even/odd interleave
  add_test(NAME RoPELowering
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/rope_lowering.mlir | ${FILECHECK_BIN} --implicit-check-not=math.cos --implicit-check-not=math.sin ${CMAKE_SOURCE_DIR}/test/Conversion/rope_lowering.mlir"
  )
  # Attention lowering: verify llk.attention → Q@K^T + softmax + @V
  add_test(NAME AttentionLowering
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/attention_lowering.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/attention_lowering.mlir"
  )
  # Schedule parse test: verify the transform dialect schedule parses correctly
  # and the named sequence round-trips through llk-opt.
  add_test(NAME ScheduleBF16Check
      COMMAND sh -c "$<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}/test/Transforms/schedule_bf16_check.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/schedule_bf16_check.mlir"
  )
  # TileAndVectorize test: run llk-opt with LLK-to-Linalg + TileAndVectorize, then FileCheck
  add_test(NAME TileAndVectorize
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg --tile-and-vectorize \
        ${CMAKE_SOURCE_DIR}/test/Transforms/tile_and_vectorize.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/tile_and_vectorize.mlir"
  )
  # MaskGen test: verify vector.create_mask and vector.mask for N=127 tail
  add_test(NAME MaskGen
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg --tile-and-vectorize \
        ${CMAKE_SOURCE_DIR}/test/Transforms/mask_gen.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/mask_gen.mlir"
  )
  # FuseDoubleContraction test
  add_test(NAME FuseDoubleContraction
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg --fuse-double-contraction \
        ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_double_contraction.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_double_contraction.mlir"
  )
  # FuseIndependent test: independent swiglus with different X are each fused
  add_test(NAME FuseIndependent
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg --fuse-double-contraction \
        ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_independent.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_independent.mlir"
  )
  # FuseReject test: two matmuls sharing X with non-SiLU consumer -> no fusion
  add_test(NAME FuseReject
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --fuse-double-contraction \
        ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_reject.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_reject.mlir"
  )
  # PackWeights test: annotate weight operands with packed_layout attribute
  add_test(NAME PackWeights
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg --pack-weights \
        ${CMAKE_SOURCE_DIR}/test/Transforms/pack_weights.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/pack_weights.mlir"
  )
  # LinearizeForall test: multi-D scf.forall → 1D
  add_test(NAME LinearizeForall
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --linearize-forall \
        ${CMAKE_SOURCE_DIR}/test/Transforms/linearize_forall.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/linearize_forall.mlir"
  )
  # SerialParallelDispatch test: threshold-based serial/parallel
  add_test(NAME SerialDispatch
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --serial-parallel-dispatch \
        ${CMAKE_SOURCE_DIR}/test/Transforms/serial_dispatch.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/serial_dispatch.mlir"
  )
  # ForallToLLRT test: scf.forall → runtime call with worker outline
  add_test(NAME ForallToLLRT
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --forall-to-llrt \
        ${CMAKE_SOURCE_DIR}/test/Transforms/forall_to_llrt.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/forall_to_llrt.mlir"
  )
  # ForallToOpenMP test: scf.forall → omp.parallel + omp.wsloop
  add_test(NAME ForallToOpenMP
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --forall-to-openmp \
        ${CMAKE_SOURCE_DIR}/test/Transforms/forall_to_openmp.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/forall_to_openmp.mlir"
  )
  # ShapeSpecialization test: check M bucket classification
  add_test(NAME ShapeSpecialization
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --shape-specialize \
        ${CMAKE_SOURCE_DIR}/test/Transforms/shape_specialization.mlir \
        2>&1 | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/shape_specialization.mlir"
  )
  # ScheduleSelection test: check correct schedule is chosen
  add_test(NAME ScheduleSelection
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --select-schedule \
        ${CMAKE_SOURCE_DIR}/test/Transforms/schedule_selection.mlir \
        2>&1 | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/schedule_selection.mlir"
  )
  # ScheduleFallback test: check fallback behavior for unknown (N,K)
  add_test(NAME ScheduleFallback
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --select-schedule \
        ${CMAKE_SOURCE_DIR}/test/Transforms/schedule_fallback.mlir \
        2>&1 | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/schedule_fallback.mlir"
  )
  # Online softmax: verify max reduction + exp + div pattern in attention lowering
  add_test(NAME OnlineSoftmax
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Transforms/online_softmax.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/online_softmax.mlir"
  )
  # RoPE vector shuffle: verify even/odd slice + interleave
  add_test(NAME RoPEVectorShuffle
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Transforms/rope_vector_shuffle.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/rope_vector_shuffle.mlir"
  )
  # RoPE broadcast: verify trig tables broadcast across BxH dimensions
  add_test(NAME RoPEBroadcast
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Transforms/rope_broadcast.mlir | ${FILECHECK_BIN} --implicit-check-not=math.cos --implicit-check-not=math.sin ${CMAKE_SOURCE_DIR}/test/Transforms/rope_broadcast.mlir"
  )
  # GridToForall: verify tt.get_program_id -> scf.forall lowering
  add_test(NAME GridToForall
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-grid-to-forall \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/grid_to_forall.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/grid_to_forall.mlir"
  )
  # GridToForallReturnOperands: regression for func.return referencing a cloned
  # op -- the pass must remap the return operand instead of crashing on erase.
  add_test(NAME GridToForallReturnOperands
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --verify-each=false --triton-grid-to-forall \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/grid_to_forall_return.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/grid_to_forall_return.mlir"
  )
  # TritonToStructured: verify tt.dot -> linalg.matmul lowering
  add_test(NAME TritonToStructured
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --triton-to-structured --allow-unregistered-dialect \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/triton_to_structured.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/triton_to_structured.mlir"
  )
  # BlockPointerToVector: verify tt.make_block_ptr + load/store -> vector.transfer
  add_test(NAME BlockPointerToVector
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-block-ptr-to-vector \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/block_ptr_to_vector.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/block_ptr_to_vector.mlir"
  )
  # BlockPointer1D: verify 1D block pointer lowering
  add_test(NAME BlockPointer1D
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-block-ptr-to-vector \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/block_ptr_1d.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/block_ptr_1d.mlir"
  )
  # BlockPointerMasked: verify boundary mask for partial tiles (N=127 not /8)
  add_test(NAME BlockPointerMasked
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-block-ptr-to-vector \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/block_ptr_masked.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/block_ptr_masked.mlir"
  )
  # SharedMemToScratch: verify tt.alloc -> memref.alloc, tt.async_copy -> memref.copy
  add_test(NAME SharedMemToScratch
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-shared-mem-to-scratch \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/shared_mem_to_scratch.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/shared_mem_to_scratch.mlir"
  )
  # AtomicToLLRT: verify tt.atomic_* -> llrt.atomic_* runtime calls
  add_test(NAME AtomicToLLRT
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-atomic-to-llrt \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/atomic_to_llrt.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/atomic_to_llrt.mlir"
  )
  # TritonCPUVerifierPass: whitelisted ops pass through with no diagnostics
  add_test(NAME TritonCPUVerifierPass
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-cpu-verify \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/cpu_verifier_pass.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/cpu_verifier_pass.mlir"
  )
  # TritonCPUVerifierReject: unsupported tt.* ops rejected with an error
  add_test(NAME TritonCPUVerifierReject
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-cpu-verify \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/cpu_verifier_reject.mlir \
        2>&1 | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/cpu_verifier_reject.mlir"
  )
  # TritonFullPipeline: all 5 Triton->LLK stages in one invocation
  add_test(NAME TritonFullPipeline
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect \
        --triton-to-structured --canonicalize --triton-cpu-verify \
        --triton-block-ptr-to-vector --triton-shared-mem-to-scratch \
        --triton-atomic-to-llrt --triton-grid-to-forall \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/triton_full_pipeline.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/triton_full_pipeline.mlir"
  )
  # Causal mask: verify cmp + select pattern for upper-triangular -inf
  add_test(NAME CausalMask
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Transforms/causal_mask.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/causal_mask.mlir"
  )
  # micro-perf: the simulator report on a concrete tile GEMM must contain the
  # counts, the schedule, the tile summary, and a bottleneck.
  add_test(NAME MicroPerfCli
      COMMAND sh -c "$<TARGET_FILE:micro-perf> --machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml --level=1 ${CMAKE_SOURCE_DIR}/test/Perf/micro_perf_cli.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Perf/micro_perf_cli.mlir"
  )
  # The help text has to say what the MVP is, so nobody reads "simulator" as
  # "functional emulator". Same fixture, other check prefix.
  add_test(NAME MicroPerfHelp
      COMMAND sh -c "$<TARGET_FILE:micro-perf> --help 2>&1 | ${FILECHECK_BIN} --check-prefix=HELP ${CMAKE_SOURCE_DIR}/test/Perf/micro_perf_cli.mlir"
  )
  # llk-tune in Micro mode: a search space and the AVX2 model go in, a ranked
  # schedule YAML comes out. The fixture's YAML checks describe that output.
  add_test(NAME LLKTuneSearchSpace
      COMMAND sh -c "$<TARGET_FILE:llk-tune> --input=${CMAKE_SOURCE_DIR}/test/Perf/llk_tune_search_space.mlir --machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml --M=8 --N=64 --K=64 --top-k=2 --output=${CMAKE_CURRENT_BINARY_DIR}/llk_tune_out.yaml && ${FILECHECK_BIN} --check-prefix=YAML ${CMAKE_SOURCE_DIR}/test/Perf/llk_tune_search_space.mlir < ${CMAKE_CURRENT_BINARY_DIR}/llk_tune_out.yaml"
  )
  # Compatibility: with no --input the driver keeps the pre-Micro grid search
  # and writes the JSON schedule_db entry the existing pipeline consumes.
  add_test(NAME LLKTuneLegacyJson
      COMMAND sh -c "$<TARGET_FILE:llk-tune> -M=8 -N=64 -K=64 -o=${CMAKE_CURRENT_BINARY_DIR}/llk_tune_legacy.json && ${FILECHECK_BIN} --check-prefix=LEGACY ${CMAKE_SOURCE_DIR}/test/Perf/llk_tune_search_space.mlir < ${CMAKE_CURRENT_BINARY_DIR}/llk_tune_legacy.json"
  )
  # TritonToLLK: verify llk.matmul / llk.make_tensor parse and verify
  add_test(NAME TritonToLLKNewOps
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/llk_new_ops.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/llk_new_ops.mlir"
  )
endif()

# Test for ops_invalid.mlir: runs llk-opt with --verify-diagnostics.
# Only registered when llk-opt is built (LLK_BUILD_TOOLS=ON).
if(LLK_BUILD_TOOLS)
add_test(NAME DialectOpsInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Dialect/ops_invalid.mlir
)
add_test(NAME MicroDialectOpsInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Dialect/Micro/ops_invalid.mlir
)
add_test(NAME MicroDialectTileOpsInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Dialect/Micro/tile_ops_invalid.mlir
)
add_test(NAME MicroDialectSearchSpaceInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Dialect/Micro/search_space_invalid.mlir
)
# LLKToMicro rejection tests: shapes and schedules the export must refuse with
# a diagnostic naming the offending axis.
add_test(NAME LLKToMicroLoweringInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            --llk-to-micro=schedule-db=missing.json
            ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/lowering_invalid.mlir
)
# ScratchAnalysis test: verify-diagnostics validates expected-warning comments
add_test(NAME ScratchAnalysis
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --scratch-analysis --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Transforms/scratch_analysis.mlir
)
endif()
