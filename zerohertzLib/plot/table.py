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

from typing import List, Optional, Tuple, Union

from matplotlib import pyplot as plt

from .util import savefig


def table(
    data: List[List[Union[int, float, str]]],
    col: List[Union[int, float, str]] = None,
    row: List[Union[int, float, str]] = None,
    title: Optional[str] = "tmp",
    fontsize: Optional[int] = 35,
    figsize: Optional[Tuple[int]] = (20, 8),
    dpi: Optional[int] = 300,
) -> str:
    """Dictionary로 입력받은 데이터를 scatter plot으로 시각화

    Args:
        data (``List[List[Union[int, float, str]]]``): ``len(row) X len(col)`` 의 크기를 가지는 list
        col (``List[Union[int, float, str]]]``): 열 (column)의 label
        row (``List[Union[int, float, str]]]``): 행 (row)의 label
        title (``Optional[str]``): 저장될 file의 이름
        fontsize (``Optional[int]``): 문자의 크기
        figsize (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        ``str``: 저장된 graph의 절대 경로

    Examples:
        >>> data = [["123", 123, 123.4], [123.4, "123", 123], [123, 123.4, "123"], ["123", 123, 123.4]]
        >>> col = ["테란", "저그", "프로토스"]
        >>> row = ["test1", "test2", "test3", "test4"]
        >>> zz.plot.table(data, col, row, title="Star Craft")
        >>> zz.plot.table(data, col, row, title="Star Craft2", fontsize=50)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/286497412-cd081b32-1c12-46b3-bdcb-b18d6221bd08.png
            :alt: Visualzation Result
            :align: center
            :width: 500px
    """
    fig, ax = plt.subplots(figsize=figsize)
    tbl = ax.table(cellText=data, colLabels=col, rowLabels=row, loc="center")
    for _, cell in tbl.get_celld().items():
        cell.get_text().set_fontsize(fontsize)
        cell.set_text_props(ha="center", va="center")
    ax.axis("off")
    fig.canvas.draw()
    bbox = tbl.get_window_extent(fig.canvas.get_renderer())
    table_width, table_height = bbox.width, bbox.height
    fig_width, fig_height = fig.get_size_inches() * fig.dpi
    scale_x = fig_width / table_width
    scale_y = fig_height / table_height
    tbl.scale(scale_x, scale_y)
    return savefig(title, dpi)
