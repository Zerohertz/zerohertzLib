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

import time
from collections import defaultdict
from typing import Optional

import psutil

from zerohertzLib.plot import plot


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
    tmp = 0
    time_list = []
    data = defaultdict(list)
    while True:
        time_list.append(tmp)
        cpu_usages = psutil.cpu_percent(interval=1, percpu=True)
        for core, cpu_usage in enumerate(cpu_usages):
            data[f"Core {core+1}"].append(cpu_usage)
        plot(
            time_list,
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
        tmp += tick
        if tmp > threshold:
            break
