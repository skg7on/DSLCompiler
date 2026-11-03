"""Exercise pre-project compiler selection without requiring LLVM or a compiler."""

import argparse
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


class CompilerSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name).resolve()
        self.bin = self.root / "bin"
        self.bin.mkdir()

    def executables(self, *names):
        # find_program only checks existence and permissions; no compilation is
        # needed to test which executable the module selects before project().
        for name in names:
            program = self.bin / name
            program.touch()
            program.chmod(0o755)

    def select(self, *, host="Linux", env=None, settings="", after=""):
        script = self.root / "select.cmake"
        script.write_text(
            'cmake_minimum_required(VERSION 3.20)\n'
            f'set(APPLE {"TRUE" if host == "Darwin" else "FALSE"})\n'
            f'set(CMAKE_HOST_SYSTEM_NAME "{host}")\n'
            + settings + '\n'
            f'include("{SOURCE}/cmake/LLKCompiler.cmake")\n'
            + after + '\n'
            'file(WRITE "${CMAKE_CURRENT_LIST_DIR}/selected.txt" '
            '"${CMAKE_C_COMPILER}\n${CMAKE_CXX_COMPILER}\n")\n'
        )
        process_env = dict(os.environ)
        for name in ("CC", "CXX", "CMAKE_TOOLCHAIN_FILE"):
            process_env.pop(name, None)
        process_env["PATH"] = str(self.bin)
        process_env.update(env or {})
        result = subprocess.run(
            [CMAKE, "-P", str(script)], env=process_env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return (self.root / "selected.txt").read_text().splitlines()

    def test_linux_prefers_clang_over_gcc(self):
        self.executables("clang", "clang++", "clang-23", "clang++-23", "gcc", "g++")
        self.assertEqual(self.select(),
                         [str(self.bin / "clang"), str(self.bin / "clang++")])

    def test_linux_finds_versioned_clang(self):
        for version in ("13", "20", "23", "99"):
            with self.subTest(version=version):
                self.bin = self.root / f"bin-{version}"
                self.bin.mkdir()
                self.executables(f"clang-{version}", f"clang++-{version}",
                                 "gcc", "g++")
                self.assertEqual(self.select(),
                                 [str(self.bin / f"clang-{version}"),
                                  str(self.bin / f"clang++-{version}")])

    def test_linux_prefers_newest_version_when_no_unversioned_name(self):
        self.executables("clang-9", "clang++-9", "clang-20", "clang++-20",
                         "clang-23", "clang++-23")
        self.assertEqual(self.select(),
                         [str(self.bin / "clang-23"), str(self.bin / "clang++-23")])

    def test_macos_still_prefers_clang_on_path(self):
        self.executables("clang", "clang++", "gcc", "g++")
        self.assertEqual(self.select(host="Darwin", settings=
            'set(CMAKE_OSX_DEPLOYMENT_TARGET 14.0)'),
            [str(self.bin / "clang"), str(self.bin / "clang++")])

    def test_path_order_wins_over_unversioned_name_in_later_directory(self):
        self.executables("clang-20", "clang++-20")
        other = self.root / "other-bin"
        other.mkdir()
        for name in ("clang", "clang++"):
            program = other / name
            program.touch()
            program.chmod(0o755)
        self.assertEqual(self.select(env={"PATH": os.pathsep.join(
            (str(self.bin), str(other)))}),
            [str(self.bin / "clang-20"), str(self.bin / "clang++-20")])

    def test_no_clang_leaves_cmake_default_selection_available(self):
        self.executables("gcc", "g++")
        self.assertEqual(self.select(), ["", ""])

    def test_explicit_and_cached_compilers_are_preserved(self):
        self.executables("clang", "clang++", "gcc", "g++")
        settings = (
            f'set(CMAKE_C_COMPILER "{self.bin}/gcc" CACHE FILEPATH "")\n'
            f'set(CMAKE_CXX_COMPILER "{self.bin}/g++" CACHE FILEPATH "")\n'
        )
        self.assertEqual(self.select(settings=settings),
                         [str(self.bin / "gcc"), str(self.bin / "g++")])

    def test_environment_compilers_are_left_to_cmake(self):
        self.executables("clang", "clang++", "gcc", "g++")
        for cc, cxx in (("gcc", "g++"),
                        (str(self.bin / "gcc"), str(self.bin / "g++"))):
            with self.subTest(cc=cc):
                self.assertEqual(self.select(env={"CC": cc, "CXX": cxx}),
                                 ["", ""])

    def test_single_language_override_preserves_the_other_language_default(self):
        self.executables("clang", "clang++", "gcc")
        self.assertEqual(self.select(env={"CC": "gcc"}),
                         ["", str(self.bin / "clang++")])
        self.assertEqual(self.select(settings=
            f'set(CMAKE_C_COMPILER "{self.bin}/gcc")'),
            [str(self.bin / "gcc"), str(self.bin / "clang++")])

    def test_missing_absolute_environment_compilers_fall_back_to_clang(self):
        self.executables("clang", "clang++")
        selected = self.select(env={"CC": str(self.root / "missing-cc"),
                                    "CXX": str(self.root / "missing-cxx")})
        self.assertEqual(selected,
                         [str(self.bin / "clang"), str(self.bin / "clang++")])

    def test_toolchain_file_can_choose_compilers_at_project(self):
        self.executables("clang", "clang++")
        toolchain = self.root / "toolchain.cmake"
        toolchain.touch()
        self.assertEqual(self.select(settings=
            f'set(CMAKE_TOOLCHAIN_FILE "{toolchain}")'), ["", ""])
        self.assertEqual(self.select(env={"CMAKE_TOOLCHAIN_FILE": str(toolchain)}),
                         ["", ""])

    def test_existing_selection_survives_path_changes(self):
        self.executables("clang", "clang++")
        self.assertEqual(self.select(after=
            'set(ENV{PATH} "")\n'
            f'include("{SOURCE}/cmake/LLKCompiler.cmake")'),
            [str(self.bin / "clang"), str(self.bin / "clang++")])

    def test_other_hosts_keep_default_selection(self):
        self.executables("clang", "clang++")
        self.assertEqual(self.select(host="Windows"), ["", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source_dir.resolve()
    CMAKE = shutil.which("cmake")
    unittest.main(argv=[__file__, *remaining])
