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
    AIT_STATIC_FILES_PATH,
    Target,
    TargetType,
)

from aitemplate.utils import environ
from aitemplate.utils.misc import is_debug, is_linux

# pylint: disable=C0415,W0707,W0611,W0702,W1401


_LOGGER = logging.getLogger(__name__)


class RVV(Target):
    """RVV target."""

    def __init__(
        self,
        template_path = "",
        arch="rv64gcv_zvfh",
        RVV_version=None,
        ait_static_files_path=AIT_STATIC_FILES_PATH,
        **kwargs,
    ):
        """RVV target init.

        Parameters
        ----------
        template_path : str, optional
        ait_static_files_path : str
            Absolute path to the AIT static/ directory
        """
        super().__init__()
        self._target_type = 3
        self._template_path = template_path
        self._arch = arch
        self._kwargs = kwargs
        self._compile_options = self._build_compile_options()
        if RVV_version is None:
            # try to set default RVV version based on the arch
            # clang17 supports v0.12 version
            RVV_version = "0.12"
        self._RVV_version = RVV_version


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

    def _build_clang_compiler_options(self) -> List[str]:
        options = [
            "--target=riscv64-unknown-elf",
            "--sysroot=$(RISCV)/riscv64-unknown-elf",
            "--gcc-toolchain=$(RISCV)",
            "-menable-experimental-extensions",
            environ.get_compiler_opt_level(),
            "-std=c++17",
            f"-march={self._arch}",
            f"-mrvv-vector-bits={environ.get_mrvv_vector_bits()}",
            "-v",
        ]
        options.extend(self._get_clang_debug_options())
        return options

    def get_device_compiler_options(self) -> List[str]:
        return self._build_clang_compiler_options()

    def _build_compile_options(self):
        options = self._build_gnu_host_compiler_options() + self._build_clang_compiler_options
        return " ".join(options)

    def src_extension(self):
        return ".cpp"
    
    def __enter__(self):
        super().__enter__()
        self._operators = []


    def cc(self):
        cc = "clang++"
        return cc

    def compile_cmd(self, executable=False):
        if executable:
            cmd = self.cc() + " " + self._compile_options + " -o {target} {src}"
        else:
            cmd = self.cc() + " " + self._compile_options + " -c -o {target} {src}"
        return cmd

    def dev_select_flag(self):
        return "RVV_VISIBLE_DEVICES"



@registry.reg("rvv.create_target")
def create_target(template_path, arch, **kwargs):
    return RVV(template_path=template_path, arch=arch, **kwargs)
