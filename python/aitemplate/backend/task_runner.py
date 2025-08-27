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
This module is a general-purpose subprocess-based task runner.
"""

from __future__ import annotations

import os
import subprocess
import time
import typing
from collections import OrderedDict
from typing import List
import threading
import logging
_LOGGER = logging.getLogger(__name__)

class RemoteProcess:
    def __init__(self, ssh_client, cmd, env=None, shell=False, timeout=None):
        self.ssh_client = ssh_client
        self.cmd = ' '.join(cmd)
        self.env = env or {}
        self.timeout = timeout
        stdin, stdout, stderr = ssh_client.exec_command(self.cmd, timeout=self.timeout)
        self.stdin = stdin
        self._stdout = stdout.read().decode()
        self._stderr = stderr.read().decode()

        _LOGGER.info(f"remote running task with stdout = {self._stdout}, stderr = {self._stderr}")
        self._exit_status = None
        self._channel = stdout.channel
        threading.Thread(target=self._wait_for_exit, daemon=True).start()

    def _wait_for_exit(self):
        self._exit_status = self._channel.recv_exit_status()

    def communicate(self):
        self._wait_for_exit()
        return self._stdout, self._stderr
    def poll(self):
        if self._exit_status is not None:
            return self._exit_status
        if self._channel.exit_status_ready():
            self._exit_status = self._channel.recv_exit_status()
            return self._exit_status
        return None

    @property
    def returncode(self):
        return self._exit_status
    def kill(self):
        return
# pylint: disable=R1732,R1710,R1721
class Task:
    """Task is an object containing a bash command,
    process for the command, and output of the process.
    """

    def __init__(
        self, idx: typing.Union[int, str], cmd: str, name: str, **kwargs
    ) -> None:
        """

        Parameters
        ----------
        idx : Union[int, str]
            unique id for the task
        cmd : str
            bash command for the task
        name : str
            alias name of the task
        """
        from aitemplate.compiler.compiler import IS_REMOTE_COMPILE
        self._finished = False
        self._is_timeout = False
        self._failed = False
        self._idx = idx
        self._cmd = cmd
        self._name = name
        self._ret = None
        self._assigned_dev = None
        self._proc = None
        self._timestamp = 0
        self._stdout = ""
        self._stderr = ""
        self._kwargs = kwargs
        self._is_remote_compile = IS_REMOTE_COMPILE

    def __call__(self, dev_id: int) -> None:
        """Execute the bash command with a new subprocess.

        Parameters
        ----------
        dev_id : int
            Target execution device id.
        """
        self._assigned_dev = dev_id
        use_shell = False
        if "shell" in self._kwargs:
            use_shell = self._kwargs["shell"]
        env = os.environ.copy()
        if "dev_flag" in self._kwargs:
            env[self._kwargs["dev_flag"]] = str(dev_id)
        from aitemplate.utils.remote_send_receive_files import TARGET_USER, TARGET_IP, SSH_CLIENT
        
        if self._is_remote_compile:
            _LOGGER.info(f"run with cmd = {self._cmd}, ssh_client exist? {SSH_CLIENT is not None}")
            self._proc = RemoteProcess(
                ssh_client=SSH_CLIENT,
                cmd=self._cmd,
                env=env,
                shell=use_shell,
                timeout=200,
            )
        else:
            self._proc = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                shell=use_shell,
            )
        self._timestamp = time.time()

    def is_running(self) -> bool:
        """Check whether the task process is still running.

        Returns
        -------
        bool
            Whether the task process is still running
        """
        return self._proc is not None

    def is_finished(self) -> bool:
        """Check whether the task is finished

        Returns
        -------
        bool
            Whether the task is finished
        """
        return self._finished

    def is_timeout(self) -> bool:
        """Check whether the task is timeout

        Returns
        -------
        bool
            Whether the task is timeout
        """
        return self._is_timeout

    def poll(self, current_time, timeout) -> bool:
        """Given the current time, check whether
        the task is running, finished or timed out.

        Parameters
        ----------
        current_time : int
            Current timestamp
        timeout : int
            Timeout time

        Returns
        -------
        bool
            Whether the task is finished
        """
        if self.is_running() is False:
            return False
        # handle timeout job
        step = current_time - self._timestamp
        if step > timeout:
            self._proc.kill()
            self._finished = True
            self._is_timeout = True
            self._failed = True
            return True
        # handle finished job
        if self._proc.poll() is not None:
            self._finished = True
        return self._finished

    def pull(self, fproc: typing.Callable) -> None:
        """Pull stdout & stderr from process,
        process stdout & stderr with fproc, and set the output for the task.

        Parameters
        ----------
        fproc : Callable
            Process function of the task given stdout & stderr
        """
        if self._failed:
            return None
        if self._is_remote_compile:
            self._stdout = self._proc._stdout
            self._stderr = self._proc._stderr
        else:
            self._stdout = self._proc.stdout.read().decode("utf-8")
            self._stderr = self._proc.stderr.read().decode("utf-8")
        fproc(self)

    def is_failed(self) -> bool:
        """Check whether the task is failed

        Returns
        -------
        bool
            Whether the task is failed
        """
        return self._failed

    def assigned_dev(self) -> int:
        """Return the assigned device id for the task

        Returns
        -------
        int
            Assigned device id
        """
        return self._assigned_dev

    def __del__(self) -> None:
        """Clean up process resource"""
        if self._proc and self._is_remote_compile is False:
            if self._proc.stdout:
                self._proc.stdout.close()
            if self._proc.stderr:
                self._proc.stderr.close()


class DeviceFarm:
    """Device Farm is a stateful object to
    schedule and assigns a task to the available devices.
    Devices are logical devices, can be CPUs or GPUs.
    """

    def __init__(self, devs: List[int]) -> None:
        """Initialize a Device Farm given a list of device ids.

        Parameters
        ----------
        devs : List[int]
            List of device ids in int
        """
        if isinstance(devs, int):
            devs = list(range(devs))
        assert isinstance(devs, list)
        self._dev_stats = OrderedDict()
        self._devs = devs
        for dev in devs:
            self._dev_stats[dev] = False

    def next_idle_dev(self) -> typing.Optional[int]:
        """Return the next idle (available) device id

        Returns
        -------
        Union[None, int]
            The next idle device id. If all devices are busy, return None
        """
        ret_id = None
        for dev_id, dev_status in self._dev_stats.items():
            if dev_status is False:
                ret_id = dev_id
        self._dev_stats[ret_id] = True
        return ret_id

    def reset_dev_state(self, dev_id: int) -> None:
        """Rest the device id state to idle

        Parameters
        ----------
        dev_id : int
            The id of device will be reset
        """
        self._dev_stats[dev_id] = False

    def reset_all(self) -> None:
        """Reset all devices to be idle"""
        for dev in self._devs:
            self._dev_stats[dev] = False


class BaseRunner:
    """Genetic subprocess task runner for different purposes"""

    def __init__(self, devs: List[int], tag: str, timeout: int = 10) -> None:
        """
        Parameters
        ----------
        devs : List[int]
            List of device ids for tasks.
        tag : str
            Runner's name tag
        timeout : int, optional
            Timeout value. Default is 10 (seconds).
        """
        self._tag = tag
        self._devs = DeviceFarm(devs)
        self._timeout = timeout
        self._finished_tasks = set()
        self._queue = []

    def join(self) -> None:
        """Waiting until all tasks are finished."""
        while True:
            all_finished = True
            current_time = time.time()
            for task in self._queue:
                all_finished = all_finished and task.is_finished()
                if task._idx in self._finished_tasks:
                    continue
                if task.is_running() is False:
                    next_dev = self._devs.next_idle_dev()
                    if next_dev is None:
                        continue
                    task(next_dev)
                    continue
                if task.poll(current_time, self._timeout):
                    self._devs.reset_dev_state(task.assigned_dev())
                    self._finished_tasks.add(task._idx)
            if all_finished:
                break

    def reset(self) -> None:
        """Reset runner, clear task queue and device states"""
        self._devs.reset_all()
        self._finished_tasks = set()
        self._queue = []

    def pull(self, ftask_proc: typing.Callable, fret_proc: typing.Callable) -> List:
        """Pull results from all tasks executed on the runner.

        Parameters
        ----------
        ftask_proc : Callable
            Function to process each task's output
        fret_proc : Callable
            Function to extract returns from task

        Returns
        -------
        List
            Aggregated returns from all tasks
        """
        ret = []
        for task in self._queue:
            task.pull(ftask_proc)
            if task.is_finished():
                if task._ret is not None:
                    ret.append(fret_proc(task))
        self.reset()
        return ret

    def push(self, idx: typing.Union[int, str], cmd: str):
        """Push a task into runner

        Parameters
        ----------
        idx : Union[int, str]
            id of the task
        cmd : str
            bash command line for the task

        """
        raise NotImplementedError
