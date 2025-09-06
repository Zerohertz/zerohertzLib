# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import subprocess
import time
from collections import defaultdict

from zerohertzLib.plot import plot


def _get_gpu_usages() -> list[float]:
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    usages = result.strip().split("\n")
    return [float(usage) for usage in usages]


def gpu_usages(
    tick: int = 1,
    threshold: int = 10,
    grep: list[int] | None = None,
    path: str = "GPU Usages",
    dpi: int = 100,
) -> None:
    """시간에 따른 GPU의 사용량을 각 GPU에 따라 line chart로 시각화

    Args:
        tick: Update 주기
        threshold: 시각화할 총 시간
        grep: 시각화할 GPU의 번호
        path: Graph를 저장할 경로
        dpi: Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        지정한 경로에 바로 graph 저장

    Examples:
        >>> zz.monitoring.gpu_usages(threshold=15)

        ![GPU usage monitoring example](../../../assets/monitoring/gpu_usages.png){ width="600" }
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


def _get_gpu_memory() -> list[tuple[float, float]]:
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
    tick: int = 1,
    threshold: int = 10,
    grep: list[int] | None = None,
    path: str = "GPU Memory",
    dpi: int = 100,
) -> None:
    """시간에 따른 GPU의 memory 사용량을 각 GPU에 따라 line chart로 시각화

    Args:
        tick: Update 주기
        threshold: 시각화할 총 시간
        grep: 시각화할 GPU의 번호
        path: Graph를 저장할 경로
        dpi: Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        지정한 경로에 바로 graph 저장

    Examples:
        >>> zz.monitoring.gpu_memory(threshold=15)

        ![GPU memory monitoring example](../../../assets/monitoring/gpu_memory.png){ width="600" }
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
