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
Graph pass to invoke profiling.
"""
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from typing import List
import re, pathlib
from pathlib import Path
from aitemplate.backend import builder, codegen

from aitemplate.backend.profiler_runner import ProfilerRunner
from aitemplate.compiler.base import DynamicProfileStrategy, Tensor

from aitemplate.compiler.ops.gemm_universal.gemm_common import (
    gemm,
    GemmProfilerPostprocessingDelegate,
)
from aitemplate.utils.environ import force_profiler_cache

# pylint: disable=C0103,W0613,W0102


_LOGGER = logging.getLogger(__name__)


def elapsed_dt_sec(start_t_sec):
    return datetime.now() - start_t_sec


def _splitter(data, pred=bool):
    group_a = []
    group_b = []
    for d in data:
        (group_a if pred(d) else group_b).append(d)
    return group_a, group_b


def profile(
    sorted_graph: List[Tensor],
    test_name,
    workdir="./tmp",
    devices=None,
    dynamic_profiling_strategy=DynamicProfileStrategy.MAX,
    timeout=500,
):
    """Profiles kernels.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        A sorted graph which contains all functions for profiling.
    workdir : str, optional
        The base dir to generate profiling source codes. By default "./tmp"
    devices: list, optional
        A list of device ids which can be used for profiling.
        By default device 0 will be used.
    dynamic_profiling_strategy: DynamicProfileStrategy, optional
        A dynamic profiling strategy, used to filter generated profiles at compile time.
        See also: :func:`~aitemplate.compiler.transform.profile.profile`
        By default MAX is used, i.e. to profile a dynamic range, an upper bound will be used.
    """

    if devices is None:
        devices = [0]
    profiler_dir = os.path.join(workdir)
    start_t = datetime.now()
    _LOGGER.info(f"Force profiler cache = {force_profiler_cache()}")
    generated_profilers = list(
        codegen.gen_profiler(sorted_graph, profiler_dir, dynamic_profiling_strategy)
    )
    generated_profilers = [p for p in generated_profilers if p is not None]
    _LOGGER.info(
        f"generated {len(generated_profilers)} profilers elapsed time: {elapsed_dt_sec(start_t)}",
    )
    start_t = datetime.now()
    compile_engine = builder.get_compile_engine()
    compile_engine.make_profilers(generated_profilers, profiler_dir)
    _LOGGER.info(f"compiled profilers elapsed time: {elapsed_dt_sec(start_t)}")
    funcs_to_profile = OrderedDict(
        (func._attrs["name"], func)
        for node in sorted_graph
        for func in node.src_ops()
        if func._attrs["has_profiler"]
    )

    start_t = datetime.now()
    gemms, non_gemms = _splitter(
        funcs_to_profile.values(), lambda f: isinstance(f, gemm)
    )
    for f in non_gemms:
        f.profile(
            workdir=profiler_dir,
            devices=devices,
        )
    profiler_runner = ProfilerRunner(
        devices,
        postprocessing_delegate=GemmProfilerPostprocessingDelegate(),
        timeout=timeout,
    )
    for f in gemms:
        f.profile(
            workdir=profiler_dir,
            profiler_runner=profiler_runner,
        )
    profiler_runner.join()
    _LOGGER.info(
        f"ran {len(funcs_to_profile)} profilers elapsed time: {elapsed_dt_sec(start_t)}",
    )
    rows_for_record = []
    record_dictionary = {}
    for node in sorted_graph:
        for func in node.src_ops():
            if func._attrs["has_profiler"]:
                func._attrs["exec_path"] = deepcopy(
                    funcs_to_profile[func._attrs["name"]]._attrs["exec_path"]
                )
                fname = func._attrs["name"]
                if "pruning" in fname:
                    _LOGGER.info(f"fetching profile result of {fname}")
                    workloads = list(func._attrs["exec_path"].keys())
                    for wkl in workloads:
                        # record the profile summary
                        best_algo = func._attrs["exec_path"][wkl]
                        m = re.search(r'_(\d+)x(\d+)v$', best_algo)
                        tile_size, lmul = map(int, m.groups())
                        row = (f"|{func._attrs['inputs'][1]._attrs['name']}|"
                            f"{func._attrs['name']}|"
                            f"{func._attrs['CO']}, {func._attrs['KH']}, "
                            f"{func._attrs['KW']}, {func._attrs['inputs'][0]._attrs['shape'][3]._attrs['symbolic_value']}|"
                            f"{lmul}|{tile_size}|\n")
                        rows_for_record.append(row)
                        record_dictionary[f"{func._attrs['inputs'][1]._attrs['name']}_indice"] = tile_size

    if rows_for_record is not None:
        md_path = pathlib.Path(f"profile_summary_{test_name}.md")
        if not md_path.exists():
            header = (
                "|**Parameter**|operator|**Dimension**|lmul|tile_size|\n"
                "|---|---|---|---|---|\n"
            )
            md_path.write_text(header, encoding="utf-8")
        with md_path.open("a", encoding="utf-8") as fp:
            fp.writelines(rows_for_record)
    return record_dictionary
