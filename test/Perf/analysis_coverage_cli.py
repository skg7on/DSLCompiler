"""Exercise coverage through the public micro-perf YAML report."""
import pathlib
import subprocess
import sys
import tempfile

exe, source = sys.argv[1:]
with tempfile.TemporaryDirectory() as directory:
    fixture = pathlib.Path(directory) / 'coverage.mlir'
    for unknown in (True, False):
        computation = '%b = arith.addf %a, %a : f32' if unknown else ''
        fixture.write_text('module { micro.kernel @coverage {\n'
                           '%a = arith.constant 1.0 : f32\n'
                           + computation + '\nmicro.yield\n} }\n')
        for level in (0, 1):
            result = subprocess.run(
                [exe, str(fixture), '--machine=' + source + '/machines/x86-avx2-v2.yaml',
                 '--format=yaml', '--level=' + str(level)],
                text=True, capture_output=True, check=True)
            expected = 'complete: false' if unknown else 'complete: true'
            assert expected in result.stdout, result.stdout
            if unknown:
                assert 'unmodeled_leaf: arith.addf' in result.stdout, result.stdout
            else:
                assert 'incomplete_reasons: []' in result.stdout, result.stdout
