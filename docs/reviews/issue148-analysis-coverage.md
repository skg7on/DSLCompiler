# Issue #148 validation

Base: `f3b2adc` (merged A2). Implementation: the commit containing this record.
Fresh worktree build configured with:

```
cmake -S . -B build -G Ninja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
cmake --build build --target LLKPerfTests micro-perf -j 6
build/bin/LLKPerfTests
ctest --test-dir build -R '^AnalysisCoverageCLI$' --output-on-failure
```

Results: 188/188 performance GTests passed; 1/1 AnalysisCoverageCLI passed.
Before implementation, the two unknown-coverage GTests failed because shared
analysis incorrectly returned complete; the CLI test failed for missing coverage
status. The alias-only preservation test passed before and after the change.

Plan reconciliation: the actual CLI supports YAML/text, not JSON, and the report
had no completeness fields. Expose the shared verdict as `complete` and
`incomplete_reasons` in YAML and corresponding text output. No second extraction
or completeness calculation was introduced. Both selected-kernel overloads use
audited extraction, preserving measured overrides on the plan overload.

Full-suite checks remain at increment exit as requested by the issue. The local
linker reports existing duplicate-static-library warnings.
