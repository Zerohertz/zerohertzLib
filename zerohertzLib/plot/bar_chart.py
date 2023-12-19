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
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from .util import color, savefig


def barv(
    data: Dict[str, Any],
    xlab: Optional[str] = "변수 [단위]",
    ylab: Optional[str] = "빈도 [단위]",
    title: Optional[str] = "tmp",
    figsize: Optional[Tuple[int]] = (15, 10),
    dpi: Optional[int] = 300,
    rot: Optional[int] = 0,
    per: Optional[bool] = True,
    save: Optional[bool] = True,
) -> str:
    """Dictionary로 입력받은 데이터를 가로 bar chart로 시각화

    Args:
        data (``Dict[str, Any]``): 입력 데이터
        xlab (``Optional[str]``): Graph에 출력될 X축 label
        ylab (``Optional[str]``): Graph에 출력될 Y축 label
        title (``Optional[str]``): Graph에 표시될 제목 및 file 이름
        figsize (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)
        rot: (``Optional[int]``): X축의 눈금 회전 각도
        per: (``Optional[bool]``): 각 bar 상단에 percentage 표시 여부
        save (``Optional[bool]``): Graph 저장 여부

    Returns:
        ``str``: 저장된 graph의 절대 경로

    Examples:
        >>> data = {"테란": 27, "저그": 40, "프로토스": 30}
        >>> zz.plot.barv(data, xlab="종족", ylab="인구 [명]", title="Star Craft")
        >>> data = {"xticks": ["테란", "저그", "프로토스"], "Type A": [4, 5, 6], "Type B": [4, 3, 2], "Type C": [8, 5, 12], "Type D": [6, 3, 2]}
        >>> zz.plot.barv(data, xlab="종족", ylab="시간 [초]", title="Star Craft")

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/291453369-55c38b80-9edd-4d6a-ae01-ff834676d541.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    colors = color(len(data))
    if len(data) == 1:
        colors = [colors]
    if save:
        plt.figure(figsize=figsize)
    if isinstance(list(data.values())[-1], list):
        data = data.copy()
        try:
            xticks = data.pop("xticks")
        except KeyError:
            xticks = list(range(len(list(data.values())[-1])))
        bottom = np.array([0 for _ in range(len(list(data.values())[-1]))])
        for i, (key, value) in enumerate(data.items()):
            bars = plt.bar(
                xticks, value, color=colors[i], zorder=2, label=key, bottom=bottom
            )
            bottom += np.array(value)
        plt.legend()
        plt.ylim([0, 1.1 * bottom.max()])
        if per:
            maximum = bottom.max()
            total = bottom.sum()
            for bar_, bot in zip(bars, bottom):
                percentage = (bot / total) * 100
                plt.text(
                    bar_.get_x() + bar_.get_width() / 2,
                    bot + maximum * 0.01,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="bottom",
                )
    else:
        bars = plt.bar(
            data.keys(),
            data.values(),
            color=colors,
            zorder=2,
        )
        if min(data.values()) > 0:
            plt.ylim([0, 1.1 * max(list(data.values()))])
        if per:
            maximum = max(list(data.values()))
            total = sum(list(data.values()))
            for bar_ in bars:
                height = bar_.get_height()
                percentage = (height / total) * 100
                plt.text(
                    bar_.get_x() + bar_.get_width() / 2,
                    height + maximum * 0.01,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="bottom",
                )
    plt.grid(zorder=0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(rotation=rot)
    plt.title(title, fontsize=25)
    if save:
        return savefig(title, dpi)
    return None


def barh(
    data: Dict[str, Any],
    xlab: Optional[str] = "빈도 [단위]",
    ylab: Optional[str] = "변수 [단위]",
    title: Optional[str] = "tmp",
    figsize: Optional[Tuple[int]] = (10, 15),
    dpi: Optional[int] = 300,
    rot: Optional[int] = 0,
    per: Optional[bool] = True,
    save: Optional[bool] = True,
) -> str:
    """Dictionary로 입력받은 데이터를 세로 bar chart로 시각화

    Args:
        data (``Dict[str, Any]``): 입력 데이터
        xlab (``Optional[str]``): Graph에 출력될 X축 label
        ylab (``Optional[str]``): Graph에 출력될 Y축 label
        title (``Optional[str]``): Graph에 표시될 제목 및 file 이름
        figsize (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)
        rot: (``Optional[int]``): X축의 눈금 회전 각도
        per: (``Optional[bool]``): 각 bar 상단에 percentage 표시 여부
        save (``Optional[bool]``): Graph 저장 여부

    Returns:
        ``str``: 저장된 graph의 절대 경로

    Examples:
        >>> data = {"테란": 27, "저그": 40, "프로토스": 30}
        >>> zz.plot.barh(data, xlab="인구 [명]", ylab="종족", title="Star Craft")
        >>> data = {"yticks": ["테란", "저그", "프로토스"], "Type A": [4, 5, 6], "Type B": [4, 3, 2], "Type C": [8, 5, 12], "Type D": [6, 3, 2]}
        >>> zz.plot.barh(data, xlab="시간 [초]", ylab="종족", title="Star Craft")

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/291453365-e5cd1d87-8938-4f9f-8d81-ce8051ef6a68.png
            :alt: Visualzation Result
            :align: center
            :width: 450px
    """
    colors = color(len(data))
    if len(data) == 1:
        colors = [colors]
    if save:
        plt.figure(figsize=figsize)
    if isinstance(list(data.values())[-1], list):
        data = data.copy()
        try:
            yticks = data.pop("yticks")
        except KeyError:
            yticks = list(range(len(list(data.values())[-1])))
        left = np.array([0 for _ in range(len(list(data.values())[-1]))])
        for i, (key, value) in enumerate(data.items()):
            bars = plt.barh(
                yticks, value, color=colors[i], zorder=2, label=key, left=left
            )
            left += np.array(value)
        plt.legend()
        plt.xlim([0, 1.1 * left.max()])
        if per:
            maximum = left.max()
            total = left.sum()
            for bar_, left_ in zip(bars, left):
                percentage = (left_ / total) * 100
                plt.text(
                    left_ + maximum * 0.01,
                    bar_.get_y() + bar_.get_height() / 2,
                    f"{percentage:.1f}%",
                    ha="left",
                    va="center",
                    rotation=270,
                )
    else:
        bars = plt.barh(
            list(data.keys()),
            list(data.values()),
            color=colors,
            zorder=2,
        )
        if min(data.values()) > 0:
            plt.xlim([0, 1.1 * max(list(data.values()))])
        if per:
            maximum = max(list(data.values()))
            total = sum(list(data.values()))
            for bar_ in bars:
                width = bar_.get_width()
                percentage = (width / total) * 100
                plt.text(
                    width + maximum * 0.01,
                    bar_.get_y() + bar_.get_height() / 2,
                    f"{percentage:.1f}%",
                    ha="left",
                    va="center",
                    rotation=270,
                )
    plt.grid(zorder=0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.yticks(rotation=rot)
    plt.title(title, fontsize=25)
    if save:
        return savefig(title, dpi)
    return None


def hist(
    data: Dict[str, List[Union[int, float]]],
    xlab: Optional[str] = "변수 [단위]",
    ylab: Optional[str] = "빈도 [단위]",
    title: Optional[str] = "tmp",
    cnt: Optional[int] = 30,
    ovp: Optional[bool] = True,
    figsize: Optional[Tuple[int]] = (15, 10),
    dpi: Optional[int] = 300,
    save: Optional[bool] = True,
) -> str:
    """Dictionary로 입력받은 데이터를 histogram으로 시각화

    Args:
        data (``Dict[str, List[Union[int, float]]]``): 입력 데이터
        xlab (``Optional[str]``): Graph에 출력될 X축 label
        ylab (``Optional[str]``): Graph에 출력될 Y축 label
        title (``Optional[str]``): Graph에 표시될 제목 및 file 이름
        cnt (``Optional[int]``): Bin의 개수
        ovp (``Optional[bool]``): Class에 따른 histogram overlap 여부
        figsize (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)
        save (``Optional[bool]``): Graph 저장 여부

    Returns:
        ``str``: 저장된 graph의 절대 경로

    Examples:
        >>> data = {"테란": list(np.random.rand(1000) * 10), "저그": list(np.random.rand(1000) * 10 + 1), "프로토스": list(np.random.rand(1000) * 10 + 2)}
        >>> zz.plot.hist(data, xlab="성적 [점]", ylab="인원 [명]", title="Star Craft")

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/280599183-2508d4d4-7398-48ac-ad94-54296934c300.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    colors = color(len(data))
    if len(data) == 1:
        colors = [colors]
    minimum, maximum = sys.maxsize, -sys.maxsize
    for ydata in data.values():
        minimum = min(*ydata, minimum)
        maximum = max(*ydata, maximum)
    gap = max(0.01, (maximum - minimum) / cnt)
    bins = np.linspace(minimum - gap, maximum + gap, cnt)
    if save:
        plt.figure(figsize=figsize)
    if ovp:
        for i, (key, value) in enumerate(data.items()):
            plt.hist(value, bins=bins, color=colors[i], label=key, alpha=0.7, zorder=2)
    else:
        plt.hist(
            list(data.values()),
            bins=bins,
            color=colors,
            label=list(data.keys()),
            alpha=1,
            zorder=2,
        )
    plt.grid(zorder=0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title, fontsize=25)
    if len(data) > 1:
        plt.legend()
    if save:
        return savefig(title, dpi)
    return None
