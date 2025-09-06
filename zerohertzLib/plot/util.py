# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import os
import random
from typing import Any

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from . import config


def figure(figsize: tuple[int, int] = (15, 10)) -> Figure:
    """Graph 생성을 위한 함수

    Args:
        figsize: Graph의 가로, 세로 길이

    Returns:
        Graph window 생성

    Examples:
        >>> zz.plot.figure()
        <Figure size 1500x1000 with 0 Axes>
        >>> zz.plot.figure((20, 20))
        <Figure size 2000x2000 with 0 Axes>
    """
    fig = plt.figure(figsize=figsize)
    config.SAVE = False
    return fig


def subplot(*args: Any, **kwargs: Any) -> Axes:
    """Subplot 생성을 위한 함수

    Args:
        *args: matplotlib.pyplot.subplot의 위치 인수들
        **kwargs: matplotlib.pyplot.subplot의 키워드 인수들

    Returns:
        Subplot axes 생성

    Examples:
        >>> zz.plot.subplot(nrows, ncols, index, **kwargs)
        >>> zz.plot.subplot(2, 1, 1)
        <Axes: >
    """
    return plt.subplot(*args, **kwargs)


def savefig(title: str, dpi: int = 300) -> str:
    """Graph 저장 함수

    Args:
        title: Graph file 이름
        dpi: Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        저장된 graph의 절대 경로

    Examples:
        >>> zz.plot.savefig("Star Craft")
    """
    title = title.lower().replace(" ", "_").replace("/", "-")
    plt.savefig(
        f"{title}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close("all")
    config.SAVE = True
    return os.path.abspath(f"{title}.png")


def color(
    cnt: int = 1,
    rand: bool = False,
    uint8: bool = False,
    palette: str = "husl",
) -> (
    tuple[float, float, float]
    | list[int]
    | list[tuple[float, float, float]]
    | list[list[int]]
):
    """색 추출 함수

    Args:
        cnt: 추출할 색의 수
        rand: Random 추출 여부
        uint8: 출력 색상의 type
        palette: 추출할 색들의 palette

    Returns:
        단일 색 또는 list로 구성된 여러 색

    Examples:
        >>> zz.plot.color()
        (0.9710194877714075, 0.4645444048369612, 0.21958695134807432)
        >>> zz.plot.color(4)
        [(0.9677975592919913, 0.44127456009157356, 0.5358103155058701), (0.5920891529639701, 0.6418467016378244, 0.1935069134991043), (0.21044753832183283, 0.6773105080456748, 0.6433941168468681), (0.6423044349219739, 0.5497680051256467, 0.9582651433656727)]
        >>> zz.plot.color(4, True)
        [(0.22420518847992715, 0.6551391052055489, 0.8272616286387289), (0.9677975592919913, 0.44127456009157356, 0.5358103155058701), (0.21125140522513897, 0.6760830215342485, 0.6556099802889619), (0.9590000285927794, 0.36894286394742526, 0.9138608732554839)]
        >>> zz.plot.color(uint8=True)
        [53, 172, 167]
        >>> zz.plot.color(4, uint8=True)
        [[246, 112, 136], [150, 163, 49], [53, 172, 164], [163, 140, 244]]
        >>> zz.plot.color(4, True, True)
        [[247, 117, 79], [73, 160, 244], [54, 171, 176], [110, 172, 49]]
    """
    if cnt == 1:
        if uint8:
            return [
                int(i * 255)
                for i in random.choice(sns.color_palette(palette, n_colors=100))
            ]
        return random.choice(sns.color_palette(palette, n_colors=100))
    if rand:
        if uint8:
            return [
                [int(j * 255) for j in i]
                for i in random.choices(sns.color_palette(palette, n_colors=100), k=cnt)
            ]
        return random.choices(sns.color_palette(palette, n_colors=100), k=cnt)
    if uint8:
        return [
            [int(j * 255) for j in i] for i in sns.color_palette(palette, n_colors=cnt)
        ]
    return sns.color_palette(palette, n_colors=cnt)


def _color(data: Any, colors: Any) -> list[tuple[float, float, float] | str]:
    if isinstance(colors, list):
        if len(data) > len(colors):
            return colors + ["black" for _ in range(len(data) - len(colors))]
        return colors
    if colors is None:
        colors = "husl"
    colors = color(len(data), palette=colors)
    if len(data) == 1:
        return [colors]
    return colors
