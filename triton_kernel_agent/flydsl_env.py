# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Point FlyDSL at the ROCm LLVM that can link gfx9 kernels.

FlyDSL lowers through MLIR's gpu-to-binary pass, which execs ``ld.lld`` from
``PATH``. A host LLD cannot link AMDGPU bitcode. TheRock wheels ship a
matching ``ld.lld`` and device libraries next to ``torch``.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        dirs.extend(Path(p) for p in site.getsitepackages())
    except AttributeError:
        pass
    user = site.getusersitepackages()
    if user:
        dirs.append(Path(user))
    dirs.append(Path(sys.prefix) / "lib")
    dirs.extend(Path(p) for p in sys.path if p)
    return dirs


def _find_sdk() -> tuple[Path | None, Path | None]:
    seen: set[Path] = set()
    for root in _candidate_dirs():
        if root in seen:
            continue
        seen.add(root)
        core = root / "_rocm_sdk_core"
        if not core.is_dir():
            nested = root / "site-packages" / "_rocm_sdk_core"
            core = nested if nested.is_dir() else core
        if not core.is_dir():
            continue
        devel = core.parent / "_rocm_sdk_devel"
        return core, devel if devel.is_dir() else None
    return None, None


def ensure_flydsl_rocm_env() -> None:
    """Prepend TheRock's ``ld.lld`` and set ``ROCM_PATH`` when they are present."""
    core, devel = _find_sdk()
    if core is None:
        return

    llvm_bin = core / "lib" / "llvm" / "bin"
    if (llvm_bin / "ld.lld").is_file():
        prefix = str(llvm_bin)
        parts = os.environ.get("PATH", "").split(os.pathsep)
        if prefix not in parts:
            os.environ["PATH"] = prefix + os.pathsep + os.environ.get("PATH", "")

    if devel is None:
        return

    if not os.environ.get("ROCM_PATH") and (devel / "amdgcn" / "bitcode").is_dir():
        os.environ["ROCM_PATH"] = str(devel)

    lib = devel / "lib"
    if lib.is_dir():
        lib_s = str(lib)
        old = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_s not in old.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = lib_s + (os.pathsep + old if old else "")
