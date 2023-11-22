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

from typing import Dict, List, Optional, Tuple, Union

from matplotlib import pyplot as plt

from .util import _save, color


def scatter(
    data: Dict[str, List[List[Union[int, float]]]],
    size: Optional[float] = 36,
    xlab: Optional[str] = "x축 [단위]",
    ylab: Optional[str] = "y축 [단위]",
    title: Optional[str] = "tmp",
    ratio: Optional[Tuple[int]] = (15, 10),
    dpi: Optional[int] = 300,
) -> None:
    """Dictionary로 입력받은 데이터를 scatter plot으로 시각화

    Args:
        data (``Dict[str, List[List[Union[int, float]]]]``): 입력 데이터
        size (``Optional[int]``): Graph에 출력될 marker의 크기
        xlab (``Optional[str]``): Graph에 출력될 X축 label
        ylab (``Optional[str]``): Graph에 출력될 Y축 label
        title (``Optional[str]``): Graph에 표시될 제목 및 file 이름
        ratio (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        ``None``: 현재 directory에 바로 graph 저장

    Examples:
        >>> zz.plot.scatter({"테란": [list(np.random.rand(200) * 10), list(np.random.rand(200) * 10)], "저그": [list(np.random.rand(200) * 5 - 1), list(np.random.rand(200) * 5 + 1)], "프로토스": [list(np.random.rand(200) * 10 + 3), list(np.random.rand(200) * 10 - 2)]}, size=400, xlab="비용 [미네랄]", ylab="전투력 [점]", title="Star Craft")

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/282639459-efca04cc-3c4a-42c5-b07d-e64705a5f791.png
            :alt: Visualzation Result
            :align: center
            :width: 300px
    """
    colors = color(len(data))
    plt.figure(figsize=ratio)
    # import matplotlib.markers as mmarkers
    # markers = list(mmarkers.MarkerStyle.markers.keys())
    marker = ["o", "v", "^", "s", "p", "*", "x"]
    for i, (key, value) in enumerate(data.items()):
        plt.scatter(
            value[0],
            value[1],
            s=size,
            color=colors[i],
            marker=marker[i % len(marker)],
            label=key,
            zorder=2,
        )
    plt.grid(zorder=0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title, fontsize=25)
    plt.legend()
    _save(title, dpi)
