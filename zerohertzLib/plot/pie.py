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

from typing import Dict, Optional, Tuple, Union

from matplotlib import pyplot as plt

from .util import _save, color


def pie(
    data: Dict[str, Union[int, float]],
    dim: Optional[str] = "",
    title: Optional[str] = "tmp",
    ratio: Optional[Tuple[int]] = (15, 10),
    dpi: Optional[int] = 300,
    int_label: Optional[bool] = True,
) -> None:
    """Dictionary로 입력받은 데이터를 pie chart로 시각화

    Args:
        data (``Dict[str, Union[int, float]]``): 입력 데이터
        dim: (``Optional[str]``): 입력 ``data`` 의 단위
        title (``Optional[str]``): Graph에 표시될 제목 및 file 이름
        ratio (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)
        int_label: (``Optional[bool]``): Label 내 수치의 소수점 표기 여부

    Returns:
        ``None``: 현재 directory에 바로 graph 저장

    Examples:
        >>> zz.plot.pie({"테란": 27, "저그": 40, "프로토스": 30}, dim="명", title="Star Craft")

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/282473748-bec83476-9ed6-4fe8-8f1a-2651344c1b7c.png
            :alt: Visualzation Result
            :align: center
            :width: 300px
    """
    colors = color(len(data))
    plt.figure(figsize=ratio)
    if int_label:
        if dim == "":
            labels = [f"{k} ({v:.0f})" for k, v in data.items()]
        else:
            labels = [f"{k} ({v:.0f} {dim})" for k, v in data.items()]
    else:
        if dim == "":
            labels = [f"{k} ({v:.2f})" for k, v in data.items()]
        else:
            labels = [f"{k} ({v:.2f} {dim})" for k, v in data.items()]
    plt.pie(
        data.values(),
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        radius=1,
        colors=colors,
        normalize=True,
    )
    plt.title(title, fontsize=25)
    plt.axis("equal")
    _save(title, dpi)
