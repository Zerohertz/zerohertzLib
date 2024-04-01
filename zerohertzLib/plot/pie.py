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

from .util import _color, savefig


def pie(
    data: Dict[str, Union[int, float]],
    dim: Optional[str] = None,
    title: Optional[str] = "tmp",
    colors: Optional[Union[str, List]] = None,
    figsize: Optional[Tuple[int]] = (15, 10),
    int_label: Optional[bool] = True,
    dpi: Optional[int] = 300,
    save: Optional[bool] = True,
) -> str:
    """Dictionary로 입력받은 data를 pie chart로 시각화

    Args:
        data (``Dict[str, Union[int, float]]``): 입력 data
        dim (``Optional[str]``): 입력 ``data`` 의 단위
        title (``Optional[str]``): Graph에 표시될 제목 및 file 이름
        colors (``Optional[Union[str, List]]``): 각 요소의 색
        figsize (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        int_label (``Optional[bool]``): Label 내 수치의 소수점 표기 여부
        dpi (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)
        save (``Optional[bool]``): Graph 저장 여부

    Returns:
        ``str``: 저장된 graph의 절대 경로

    Examples:
        >>> data = {"Terran": 27, "Zerg": 40, "Protoss": 30}
        >>> zz.plot.pie(data, dim="$", title="Star Craft")

        .. image:: _static/examples/dynamic/plot.pie.png
            :align: center
            :width: 500px
    """
    colors = _color(data, colors)
    if save:
        plt.figure(figsize=figsize)
    if int_label:
        if dim is None:
            labels = [f"{k} ({v:.0f})" for k, v in data.items()]
        elif dim in ["₩", "$"]:
            labels = [f"{k} ({dim}{v:,.0f})" for k, v in data.items()]
        else:
            labels = [f"{k} ({v:.0f} {dim})" for k, v in data.items()]
    else:
        if dim is None:
            labels = [f"{k} ({v:.2f})" for k, v in data.items()]
        elif dim in ["₩", "$"]:
            labels = [f"{k} ({dim}{v:,.2f})" for k, v in data.items()]
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
    if save:
        return savefig(title, dpi)
    return None
