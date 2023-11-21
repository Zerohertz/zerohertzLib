import time
from collections import defaultdict
from typing import Optional

import psutil

import zerohertzLib as zz


def cpu(
    tick: Optional[int] = 1,
    threshold: Optional[int] = 10,
    path: Optional[str] = "cpu",
    dpi: Optional[int] = 100,
) -> None:
    """시간에 따른 CPU의 사용량을 각 코어에 따라 line chart로 시각화

    Args:
        tick (``Optional[int]``): Update 주기
        threshold: (``Optional[int]``): 시각화할 총 시간
        path (``str``): Graph를 저장할 경로
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        ``None``: 지정한 경로에 바로 graph 저장

    Examples:
        >>> zz.monitoring.cpu(threshold=15)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284103719-cdbbb87c-ee2a-4ce7-87cf-df648d10b317.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    t = 0
    ti = []
    data = defaultdict(list)
    while True:
        ti.append(t)
        cpu_usages = psutil.cpu_percent(interval=1, percpu=True)
        for core, cpu_usage in enumerate(cpu_usages):
            data[f"Core {core+1}"].append(cpu_usage)
        zz.plot.plot(
            ti,
            data,
            "시간 [초]",
            "CPU 사용률 [%]",
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
