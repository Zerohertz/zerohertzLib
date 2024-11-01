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

from .util import _color, figure, savefig


def scatter(
    data: Dict[str, List[List[Union[int, float]]]],
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    xlim: Optional[List[Union[int, float]]] = None,
    ylim: Optional[List[Union[int, float]]] = None,
    ncol: Optional[int] = 1,
    title: Optional[str] = "tmp",
    colors: Optional[Union[str, List]] = None,
    markersize: Optional[int] = 36,
    figsize: Optional[Tuple[int]] = (15, 10),
    dpi: Optional[int] = 300,
    save: Optional[bool] = True,
) -> str:
    """Dictionary로 입력받은 data를 scatter plot으로 시각화

    Args:
        data (``Dict[str, List[List[Union[int, float]]]]``): 입력 data
        xlab (``Optional[str]``): Graph에 출력될 X축 label
        ylab (``Optional[str]``): Graph에 출력될 Y축 label
        xlim (``Optional[List[Union[int, float]]]``): Graph에 출력될 X축 limit
        ylim (``Optional[List[Union[int, float]]]``): Graph에 출력될 Y축 limit
        ncol (``Optional[int]``): Graph에 표시될 legend 열의 수
        title (``Optional[str]``): Graph에 표시될 제목 및 file 이름
        colors (``Optional[Union[str, List]]``): 각 요소의 색
        markersize (``Optional[int]``): Graph에 출력될 marker의 크기
        figsize (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)
        save (``Optional[bool]``): Graph 저장 여부

    Returns:
        ``str``: 저장된 graph의 절대 경로

    Examples:
        >>> data = {"Terran": [list(np.random.rand(200) * 10), list(np.random.rand(200) * 10)], "Zerg": [list(np.random.rand(200) * 5 - 1), list(np.random.rand(200) * 5 + 1)], "Protoss": [list(np.random.rand(200) * 10 + 3), list(np.random.rand(200) * 10 - 2)]}
        >>> zz.plot.scatter(data, xlab="Cost [Mineral]", ylab="Scores", title="Star Craft", markersize=400)

        .. image:: _static/examples/dynamic/plot.scatter.png
            :align: center
            :width: 500px
    """
    colors = _color(data, colors)
    if save:
        figure(figsize=figsize)
    # import matplotlib.markers as mmarkers
    # markers = list(mmarkers.MarkerStyle.markers.keys())
    marker = ["o", "v", "^", "s", "p", "*", "x"]
    for i, (key, value) in enumerate(data.items()):
        plt.scatter(
            value[0],
            value[1],
            s=markersize,
            color=colors[i],
            marker=marker[i % len(marker)],
            label=key,
            zorder=2,
        )
    plt.grid(zorder=0)
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.title(title, fontsize=25)
    if len(data) > 1:
        plt.legend(ncol=ncol)
    if save:
        return savefig(title, dpi)
    return None
