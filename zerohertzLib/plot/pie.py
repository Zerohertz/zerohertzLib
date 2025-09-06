# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

from matplotlib import pyplot as plt

from . import config
from .util import _color, savefig


def pie(
    data: dict[str, int | float],
    dim: str | None = None,
    title: str = "tmp",
    colors: str | list | None = None,
    figsize: tuple[int, int] = (15, 10),
    int_label: bool = True,
    dpi: int = 300,
) -> str | None:
    """Dictionary로 입력받은 data를 pie chart로 시각화

    Args:
        data: 입력 data
        dim: 입력 `data` 의 단위
        title: Graph에 표시될 제목 및 file 이름
        colors: 각 요소의 색
        figsize: Graph의 가로, 세로 길이
        int_label: Label 내 수치의 소수점 표기 여부
        dpi: Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        저장된 graph의 절대 경로

    Examples:
        >>> data = {"Terran": 27, "Zerg": 40, "Protoss": 30}
        >>> zz.plot.pie(data, dim="$", title="Star Craft")

        ![Pie chart example](../../../assets/plot/pie.png){ width="500" }
    """
    colors = _color(data, colors)
    if config.SAVE:
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
    if config.SAVE:
        return savefig(title, dpi)
    return None
