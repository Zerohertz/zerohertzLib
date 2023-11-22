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

import os
from typing import Optional

from zerohertzLib.plot import pie


def _get_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
            total += os.path.getsize(filepath)
        elif os.path.isdir(filepath):
            total += _get_size(filepath)
    return total


def storage(path: str, threshold: Optional[int] = 1) -> None:
    """지정한 경로에 존재하는 file에 따른 용량을 pie graph로 시각화

    Args:
        path (``str``): 용량을 시각화할 경로
        threshold: (``Optional[int]``): Etc.로 분류될 임계값 (단위: %)

    Returns:
        ``None``: 지정한 경로에 바로 graph 저장

    Examples:
        >>> zz.monitoring.storage(".")

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/282481985-15ce10ff-e4b1-4b6a-84ea-6e948b684e0c.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    sizes = {}
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        sizes[filename] = _get_size(filepath) / (1024**3)
    total_size = sum(sizes.values())
    etc = 0
    pops = []
    for key, value in sizes.items():
        if value / total_size * 100 <= threshold:
            etc += value
            pops.append(key)
    for pop in pops:
        sizes.pop(pop)
    sizes["Etc."] = etc
    data = dict(sorted(sizes.items(), key=lambda item: item[1], reverse=True))
    pie(data, "GB", os.path.abspath(path).split("/")[-1])
