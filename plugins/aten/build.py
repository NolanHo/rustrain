#!/usr/bin/env python3
"""Builds the ATen plugin `.so`.

The build is a plain C++ shared-library compile: the plugin is host code that
links libtorch, so it needs torch's include and library directories and nothing
else. `torch.utils.cpp_extension` supplies *those paths only* — not its
`CppExtension` link list, which adds `-ltorch_python`. This plugin must not
depend on libtorch_python: it is `dlopen`ed by the Rust host, which has no
Python interpreter in it.

    python3 build.py            # -> build/librustrain_aten.so
    python3 build.py --out DIR  # elsewhere

The interpreter running this script must be the one whose `torch` the plugin
will run beside (same `_GLIBCXX_USE_CXX11_ABI`, same CUDA build).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def torch_flags() -> tuple[list[str], list[str], list[str], int]:
    """Include dirs, library dirs, the torch lib dir, and the C++11 ABI flag."""
    try:
        import torch
        import torch.utils.cpp_extension as ext
    except ImportError as exc:
        sys.exit(
            f"the interpreter running build.py cannot import torch ({exc}); run it with the "
            "interpreter that has the torch this plugin will be used with"
        )
    includes = list(dict.fromkeys(ext.include_paths(device_type="cuda") + ext.include_paths()))
    libs = list(dict.fromkeys(ext.library_paths(device_type="cuda") + ext.library_paths()))
    abi = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    return includes, libs, [str(Path(torch.__file__).parent / "lib")], abi


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    parser = argparse.ArgumentParser(description="build the rustrain ATen plugin")
    parser.add_argument("--out", default=str(here / "build"), help="output directory")
    parser.add_argument("--verbose", action="store_true", help="print every command")
    args = parser.parse_args()

    includes, libs, torch_lib, abi = torch_flags()
    sources = sorted((here / "src").glob("*.cpp"))
    if not sources:
        sys.exit(f"no sources under {here / 'src'}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "librustrain_aten.so"
    cxx = os.environ.get("CXX", "c++")

    compile_flags = [
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-fvisibility=default",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        f"-I{repo / 'crates/rustrain-abi/include'}",
    ] + [f"-isystem{i}" for i in includes]

    objects: list[str] = []
    for source in sources:
        obj = out_dir / f"{source.stem}.o"
        command = [cxx, "-c", *compile_flags, str(source), "-o", str(obj)]
        if args.verbose:
            print(" ".join(command))
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            sys.stderr.write(done.stdout + done.stderr)
            return done.returncode
        if done.stderr.strip():
            sys.stderr.write(done.stderr)
        objects.append(str(obj))

    link_flags = [f"-L{lib}" for lib in dict.fromkeys(libs)]
    # NO -ltorch_python: see the module docstring.
    link_flags += ["-ltorch_cuda", "-ltorch_cpu", "-ltorch", "-lc10_cuda", "-lc10"]
    # The plugin is dlopen'ed by a host that knows nothing about torch's layout,
    # so the search path for libtorch travels inside the .so.
    link_flags += [f"-Wl,-rpath,{lib}" for lib in dict.fromkeys([*torch_lib, *libs])]
    command = [cxx, "-shared", "-o", str(target), *objects, *link_flags]
    if args.verbose:
        print(" ".join(command))
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        sys.stderr.write(done.stdout + done.stderr)
        return done.returncode
    if done.stderr.strip():
        sys.stderr.write(done.stderr)

    print(f"built {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
