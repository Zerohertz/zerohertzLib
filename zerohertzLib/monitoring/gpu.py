import subprocess
import time
from collections import defaultdict
from typing import Optional

import zerohertzLib as zz


def _get_gpu_usages():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    usages = result.strip().split("\n")
    return [float(usage) for usage in usages]


def gpu_usages(
    tick: Optional[int] = 1,
    threshold: Optional[int] = 10,
    path: Optional[str] = "gpu usages",
    dpi: Optional[int] = 100,
) -> None:
    """시간에 따른 GPU의 사용량을 각 GPU에 따라 line chart로 시각화

    Args:
        tick (``Optional[int]``): Update 주기
        threshold: (``Optional[int]``): 시각화할 총 시간
        path (``str``): Graph를 저장할 경로
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        ``None``: 지정한 경로에 바로 graph 저장

    Examples:
        >>> zz.monitoring.gpu_usages(threshold=15)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284106493-2c6ac15c-ee68-47b4-a157-3734a0f53ddc.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    t = 0
    ti = []
    data = defaultdict(list)
    while True:
        ti.append(t)
        gpu_usages = _get_gpu_usages()
        for num, gpu_usage in enumerate(gpu_usages):
            data[f"GPU {num+1}"].append(gpu_usage)
        zz.plot.plot(
            ti,
            data,
            "시간 [초]",
            "GPU 사용률 [%]",
            ylim=[0, 100],
            ncol=2,
            title=path,
            ratio=(25, 10),
            dpi=dpi,
        )
        time.sleep(tick)
        t += tick
        if t > threshold:
            break


def _get_gpu_memory():
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
    path: Optional[str] = "gpu memory",
    dpi: Optional[int] = 100,
) -> None:
    """시간에 따른 GPU의 memory 사용량을 각 GPU에 따라 line chart로 시각화

    Args:
        tick (``Optional[int]``): Update 주기
        threshold: (``Optional[int]``): 시각화할 총 시간
        path (``str``): Graph를 저장할 경로
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        ``None``: 지정한 경로에 바로 graph 저장

    Examples:
        >>> zz.monitoring.gpu_memory(threshold=15)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284106494-ad8eec4c-62e4-4524-a696-29fda244428d.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    t = 0
    ti = []
    data = defaultdict(list)
    gpu_memory_max = 0
    while True:
        ti.append(t)
        gpu_usages = _get_gpu_memory()
        for num, (gpu_memory_usage, gpu_memory_total) in enumerate(gpu_usages):
            gpu_memory_max = max(gpu_memory_total / 1024, gpu_memory_max)
            data[f"GPU {num+1}"].append(gpu_memory_usage / 1024)
        zz.plot.plot(
            ti,
            data,
            "시간 [초]",
            "GPU Memory [GB]",
            ylim=[0, gpu_memory_max],
            ncol=2,
            title=path,
            ratio=(25, 10),
            dpi=dpi,
        )
        time.sleep(tick)
        t += tick
        if t > threshold:
            break
