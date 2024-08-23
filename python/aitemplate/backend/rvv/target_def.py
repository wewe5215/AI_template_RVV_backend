#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
RVV target specialization
"""
import json
import logging
import os
import pipes
import re
import shutil
import sys
import tempfile

from pathlib import Path
from typing import List

from aitemplate.backend import registry

from aitemplate.backend.profiler_cache import ProfileCacheDB

from aitemplate.backend.target import (
    AIT_STATIC_FILES_PATH_CPU,
    Target,
    TargetType,
    XNNPACK_PATH,
)

from aitemplate.utils import environ
from aitemplate.utils.misc import is_linux, is_windows, is_macos

# pylint: disable=C0415,W0707,W0611,W0702,W1401


_LOGGER = logging.getLogger(__name__)


class RVV(Target):
    """RVV target."""

    def __init__(
        self,
        template_path = XNNPACK_PATH,
        arch="rv64gcv_zvfh",
        RVV_version=None,
        ait_static_files_path=AIT_STATIC_FILES_PATH_CPU,
        **kwargs,
    ):
        """RVV target init.

        Parameters
        ----------
        template_path : str, optional
        ait_static_files_path : str
            Absolute path to the AIT static/ directory
        """
        super().__init__(ait_static_files_path)
        self._target_type = 3
        self._template_path = template_path
        self._ait_include_path = ait_static_files_path
        self._arch = arch
        self._kwargs = kwargs
        self._compile_options = self._build_compile_options()
        if RVV_version is None:
            # try to set default RVV version based on the arch
            # clang17 supports v0.12 version
            RVV_version = "0.12"
        self._RVV_version = RVV_version

    def _pkg_path(self):
        """Initialize package target.

        Returns
        -------
        str
            path to xnnpack compiler library
        """
        # TODO : to be revised later
        xnnpack_path = os.environ.get("XNNPACK_PATH", "/home/wewe5215/Desktop/AIT_RVV/AITemplate/3rdparty/XNNPACK/build/local/")
        return xnnpack_path
    def _build_gnu_host_compiler_options(self) -> List[str]:
        return [
            "-fPIC",
            "-Wconversion",
            "-fno-strict-aliasing",
            "-fvisibility=hidden",
        ]

    def get_host_compiler_options(self) -> List[str]:
        return self._build_gnu_host_compiler_options()

    def _get_clang_debug_options(self) -> List[str]:
        CLANG_DEBUG_LEVEL_STRINGS = [[], ["-g", "-G"]]
        level = environ.get_clang_debug_level()
        if level.isdigit():
            level = int(level)
            assert (
                level >= 0 and level < 2
            ), "Debug level out of range. Must be 0 (no debug info), 1 (with debug info)"
            return CLANG_DEBUG_LEVEL_STRINGS[level]
        return [level]

    def _build_include_directories(self) -> List[str]:
        xnnpack_path = [
            os.path.join(self._template_path, "include"),
            os.path.join(self._pkg_path(), "pthreadpool-source/include"),
        ]
        return xnnpack_path

    def get_include_directories(self) -> List[str]:
        return self._build_include_directories()
    
    def _build_link_directories(self) -> List[str]:
        xnnpack_path = [
            os.path.join(self._pkg_path(), f" {self._pkg_path()}/*.a"),
            os.path.join(self._pkg_path(), "cpuinfo"),
            os.path.join(self._pkg_path(), "kleidiai"),
            os.path.join(self._pkg_path(), "pthreadpool"),
        ]
        return xnnpack_path

    def get_link_directories(self) -> List[str]:
        return self._build_link_directories()

    def _build_clang_compiler_options(self) -> List[str]:
        options = [
            # "--target=riscv64-unknown-elf",
            # "--sysroot=$(RISCV)/riscv64-unknown-elf",
            # "--gcc-toolchain=$(RISCV)",
            # "-menable-experimental-extensions",
            environ.get_compiler_opt_level(),
            "-std=c++17",
            # f"-march={self._arch}",
            # f"-mrvv-vector-bits={environ.get_mrvv_vector_bits()}",
            "-v",
        ]
        options.extend(self._get_clang_debug_options())
        return options

    def get_device_compiler_options(self) -> List[str]:
        return self._build_clang_compiler_options()

    def compile_options(self):
        return self._compile_options
    def _build_compile_options(self):
        options = self._build_gnu_host_compiler_options() + self._build_clang_compiler_options()
        include_path = self.get_include_directories()
        link_path = self.get_link_directories()
        for path in include_path:
            options.append("-I" + path)
        for path in link_path:
            options.append("-L" + path)
        if is_macos():
            options.append("-lxnnpack -lkleidiai -lpthreadpool -lcpuinfo -lpthread -g")
        else:
            options.append("-lXNNPACK -lpthreadpool -lcpuinfo -lpthread -g")
        return " ".join(options)

    def src_extension(self):
        return ".cpp"
    
    def _gen_rvv_lib_pkg(self):
        """Build composable kernel python library.

        Raises
        ------
        RuntimeError
            Failed to create cpu library.
        """
        self.lib_folder = None
        try:
            import cpu_lib  # noqa: F401
        except BaseException:
            try:
                cur_path = os.path.dirname(os.path.realpath(__file__))
                cpu_lib_path = os.path.normpath(
                    os.path.join(cur_path, "..", "..", "utils", "cpu_lib")
                )
                f_make_lib = registry.get("rvv.cpu_lib")
                dst_path = f_make_lib(cpu_lib_path)
                sys.path.insert(1, dst_path)
            except BaseException as err:
                raise RuntimeError("Failed to create cpu library") from err
            self.lib_folder = dst_path

    def __enter__(self):
        super().__enter__()
         # Generate library.
        self._gen_rvv_lib_pkg()
        # Choose the right ops to launch.
        f_gen_ops = registry.get("rvv.gen_xnnpack_ops")
        self._operators = f_gen_ops(self._arch)


    def cc(self):
        cc = "clang++-17"
        return cc

    def compile_cmd(self, executable=False):
        if executable:
            cmd = self.cc() + " " + self._compile_options + " -o {target} {src} -Wl,--start-group -lXNNPACK -lpthreadpool -lcpuinfo -lpthread -Wl,--end-group"
        else:
            cmd = self.cc() + " " + self._compile_options + " -c -o {target} {src}"
        return cmd

    def dev_select_flag(self):
        return "RVV_VISIBLE_DEVICES"



@registry.reg("rvv.create_target")
def create_target(template_path, arch, **kwargs):
    return RVV(template_path=template_path, arch=arch, **kwargs)
