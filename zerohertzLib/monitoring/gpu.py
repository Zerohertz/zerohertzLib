"""
MIT License

Copyright (c) 2023 Hyogeun Oh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import subprocess
import time
from collections import defaultdict
from typing import List, Optional, Tuple

from zerohertzLib.plot import plot


def _get_gpu_usages() -> List[float]:
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    usages = result.strip().split("\n")
    return [float(usage) for usage in usages]


def gpu_usages(
    tick: Optional[int] = 1,
    threshold: Optional[int] = 10,
    grep: Optional[List[int]] = None,
    path: Optional[str] = "GPU Usages",
    dpi: Optional[int] = 100,
) -> None:
    """시간에 따른 GPU의 사용량을 각 GPU에 따라 line chart로 시각화

    Args:
        tick (``Optional[int]``): Update 주기
        threshold (``Optional[int]``): 시각화할 총 시간
        grep (``Optional[List[int]]``): 시각화할 GPU의 번호
        path (``Optional[str]``): Graph를 저장할 경로
        dpi (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        ``None``: 지정한 경로에 바로 graph 저장

    Examples:
        >>> zz.monitoring.gpu_usages(threshold=15)

        .. image:: _static/examples/static/monitoring.gpu_usages.png
            :align: center
            :width: 600px
    """
    tmp = 0
    time_list = []
    data = defaultdict(list)
    while True:
        time_list.append(tmp)
        gpu_usages_list = _get_gpu_usages()
        for num, gpu_usage in enumerate(gpu_usages_list):
            if grep is None or num in grep:
                data[f"GPU {num}"].append(gpu_usage)
        plot(
            time_list,
            data,
            xlab="Time [Sec]",
            ylab="GPU Usages [%]",
            ylim=[0, 100],
            ncol=2,
            title=path,
            figsize=(25, 10),
            dpi=dpi,
        )
        time.sleep(tick)
        tmp += tick
        if tmp > threshold:
            break


def _get_gpu_memory() -> List[Tuple[float]]:
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,nounits,noheader",
        ],
        encoding="utf-8",
    )
    lines = result.strip().split("\n")
    memory_usage = []
    for line in lines:
        used_memory, total_memory = line.split(",")
        memory_usage.append((float(used_memory), float(total_memory)))
    return memory_usage


def gpu_memory(
    tick: Optional[int] = 1,
    threshold: Optional[int] = 10,
    grep: Optional[List[int]] = None,
    path: Optional[str] = "GPU Memory",
    dpi: Optional[int] = 100,
) -> None:
    """시간에 따른 GPU의 memory 사용량을 각 GPU에 따라 line chart로 시각화

    Args:
        tick (``Optional[int]``): Update 주기
        threshold (``Optional[int]``): 시각화할 총 시간
        grep (``Optional[List[int]]``): 시각화할 GPU의 번호
        path (``Optional[str]``): Graph를 저장할 경로
        dpi (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        ``None``: 지정한 경로에 바로 graph 저장

    Examples:
        >>> zz.monitoring.gpu_memory(threshold=15)

        .. image:: _static/examples/static/monitoring.gpu_memory.png
            :align: center
            :width: 600px
    """
    tmp = 0
    time_list = []
    data = defaultdict(list)
    gpu_memory_max = 0
    while True:
        time_list.append(tmp)
        gpu_usages_list = _get_gpu_memory()
        for num, (gpu_memory_usage, gpu_memory_total) in enumerate(gpu_usages_list):
            if grep is None or num in grep:
                gpu_memory_max = max(gpu_memory_total / 1024, gpu_memory_max)
                data[f"GPU {num}"].append(gpu_memory_usage / 1024)
        plot(
            time_list,
            data,
            xlab="Time [Sec]",
            ylab="GPU Memory [GB]",
            ylim=[0, gpu_memory_max],
            ncol=2,
            title=path,
            figsize=(25, 10),
            dpi=dpi,
        )
        time.sleep(tick)
        tmp += tick
        if tmp > threshold:
            break
