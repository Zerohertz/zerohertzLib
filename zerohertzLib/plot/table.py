# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

from matplotlib import pyplot as plt

from .util import savefig


def table(
    data: list[list[int | float | str]],
    col: list[int | float | str] | None = None,
    row: list[int | float | str] | None = None,
    title: str = "tmp",
    fontsize: int = 35,
    figsize: tuple[int, int] = (20, 8),
    dpi: int = 300,
) -> str:
    """Dictionary로 입력받은 data를 scatter plot으로 시각화

    Args:
        data: `len(row) X len(col)` 의 크기를 가지는 list
        col: 열 (column)의 label
        row: 행 (row)의 label
        title: 저장될 file의 이름
        fontsize: 문자의 크기
        figsize: Graph의 가로, 세로 길이
        dpi: Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        저장된 graph의 절대 경로

    Examples:
        >>> data = [["123", 123, 123.4], [123.4, "123", 123], [123, 123.4, "123"], ["123", 123, 123.4]]
        >>> col = ["Terran", "Zerg", "Protoss"]
        >>> row = ["test1", "test2", "test3", "test4"]
        >>> zz.plot.table(data, col, row, title="Star Craft")
        >>> zz.plot.table(data, col, row, title="Star Craft2", fontsize=50)

        ![Table example](../../../assets/plot/table.png){ width="500" }
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
