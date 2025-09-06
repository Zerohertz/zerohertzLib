# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

from typing import Any

import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import config
from .util import _color, color, savefig


def plot(
    xdata: list[int | float] | dict[str, int | float],
    ydata: list[int | float] | dict[str, int | float],
    xlab: str | None = None,
    ylab: str | None = None,
    xlim: list[int | float] | None = None,
    ylim: list[int | float] | None = None,
    stacked: bool = False,
    ncol: int = 1,
    title: str = "tmp",
    colors: str | list | None = None,
    markersize: int = 12,
    figsize: tuple[int, int] = (15, 10),
    dpi: int = 300,
) -> str | None:
    """List와 Dictionary로 입력받은 data를 line chart로 시각화

    Args:
        xdata: 입력 data (X축)
        ydata: 입력 data (Y축)
        xlab: Graph에 출력될 X축 label
        ylab: Graph에 출력될 Y축 label
        xlim: Graph에 출력될 X축 limit
        ylim: Graph에 출력될 Y축 limit
        stacked: Stacked plot 여부
        ncol: Graph에 표시될 legend 열의 수
        title: Graph에 표시될 제목 및 file 이름
        colors: 각 요소의 색
        markersize: Graph에 표시될 marker의 size
        figsize: Graph의 가로, 세로 길이
        dpi: Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        저장된 graph의 절대 경로

    Examples:
        `stacked=False`:
            >>> xdata = [i for i in range(20)]
            >>> ydata = {"Terran": list(np.random.rand(20) * 10), "Zerg": list(np.random.rand(20) * 10 + 1), "Protoss": list(np.random.rand(20) * 10 + 2)}
            >>> zz.plot.plot(xdata, ydata, xlab="Time [Sec]", ylab="Scores", title="Star Craft")

            ![Plot example 1](../../../assets/plot/plot.1.png){ width="500" }

        `stacked=True`:
            >>> ydata["Total"] = [sum(data) + 10 for data in zip(ydata["Terran"], ydata["Protoss"], ydata["Zerg"])]
            >>> zz.plot.plot(xdata, ydata, xlab="Time [Sec]", ylab="Scores", stacked=True, title="Star Craft")

            ![Plot example 2](../../../assets/plot/plot.2.png){ width="500" }
    """
    if config.SAVE:
        plt.figure(figsize=figsize)
    if stacked:
        bias = np.zeros(len(xdata))
        assert not isinstance(xdata, dict)
    if not isinstance(ydata, dict):
        ydata = {"": ydata}
    if not isinstance(xdata, dict):
        _xdata = {}
        for key in ydata.keys():
            _xdata[key] = xdata
        xdata = _xdata
    colors = _color(ydata, colors)
    for i, (key, yvalue) in enumerate(ydata.items()):
        xvalue = xdata[key]
        if stacked:
            if key == "Total":
                colors[i] = (0.5, 0.5, 0.5)
            else:
                yvalue = np.array(yvalue) + bias
        plt.plot(
            xvalue,
            yvalue,
            color=colors[i],
            linestyle=config.LINESTYLE[i % len(config.LINESTYLE)],
            linewidth=2,
            marker=config.MARKER[i % len(config.MARKER)],
            markersize=markersize,
            label=key,
        )
        if stacked:
            plt.fill_between(xvalue, yvalue, bias, color=colors[i], alpha=0.5)
            bias = yvalue
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
    if len(ydata) > 1:
        plt.legend(ncol=ncol)
    if config.SAVE:
        return savefig(title, dpi)
    return None


def candle(
    data: pd.DataFrame,
    title: str = "tmp",
    figsize: tuple[int, int] = (18, 10),
    signals: dict[str, Any] | None = None,
    threshold: int | tuple[int, int] = 1,
    dpi: int = 300,
) -> str | None:
    """OHLCV (Open, High, Low, Close, Volume) data에 따른 candle chart

    Note:
        - 적색: 매수
        - 청색: 매도
        - 실선: Backtest 시 signal이 존재하는 매수, 매도
        - 파선: Backtest 시 사용하지 않은 signal의 매수, 매도
        - 일점쇄선: Backtest logic에 의한 매수, 매도

    Args:
        data: OHLCV (Open, High, Low, Close, Volume) data
        title: Graph에 표시될 제목 및 file 이름
        figsize: Graph의 가로, 세로 길이
        signals: 추가적으로 plot할 data
        threshold: 매수, 매도를 결정할 `signals` 경계값
        dpi: Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        저장된 graph의 절대 경로

    Examples:
        >>> zz.plot.candle(data, title)
        >>> signals = zz.quant.macd(data)
        >>> zz.plot.candle(data, "MACD", signals=signals)

        ![Candle chart example](../../../assets/plot/candle.png){ width="600" }
    """
    if not isinstance(threshold, int):
        threshold_sell, threshold_buy = threshold
    else:
        threshold_sell, threshold_buy = -threshold, threshold
    marketcolors = mpf.make_marketcolors(
        up="red",
        down="blue",
        edge="black",
        volume={"up": "red", "down": "blue"},
        inherit=False,
    )
    marketcolors["vcedge"] = {"up": "#000000", "down": "#000000"}
    style = mpf.make_mpf_style(
        marketcolors=marketcolors,
        mavcolors=color(3),
        facecolor="white",
        edgecolor="black",
        figcolor="white",
        gridcolor="gray",
        gridstyle="-",
        rc={
            "font.size": plt.rcParams["font.size"],
            "font.family": plt.rcParams["font.family"],
            "figure.titlesize": 35,
        },
    )
    # bands = _bollinger_bands(data)
    # bollinger = mpf.make_addplot(bands[["lower_band", "upper_band"]], type="line")
    _, axlist = mpf.plot(
        data,
        type="candle",
        mav=(5, 20, 60, 120),
        volume=True,
        figsize=figsize,
        title=title,
        style=style,
        returnfig=True,
        # addplot=bollinger,
    )
    if signals is not None:
        new_axis = axlist[0].twinx()
        xdata = axlist[0].get_lines()[0].get_xdata()
        buy_idx_signal = []
        sell_idx_signal = []
        buy_idx_backtest = []
        sell_idx_backtest = []
        buy_idx_logic = []
        sell_idx_logic = []
        if "logic" not in signals.columns:
            signals["logic"] = 0
        for idx, (pos_signals, pos_logic) in enumerate(
            zip(signals["signals"], signals["logic"])
        ):
            if pos_logic == 1:
                buy_idx_backtest.append(idx)
            elif pos_logic == -1:
                sell_idx_backtest.append(idx)
            elif pos_logic == 2:
                buy_idx_logic.append(idx)
            elif pos_logic == -2:
                sell_idx_logic.append(idx)
            elif pos_signals >= threshold_buy:
                buy_idx_signal.append(idx)
            elif pos_signals <= threshold_sell:
                sell_idx_signal.append(idx)
        for i in buy_idx_signal:
            new_axis.axvline(
                x=xdata[i], color="red", linestyle="--", linewidth=2, alpha=0.3
            )
        for i in sell_idx_signal:
            new_axis.axvline(
                x=xdata[i], color="blue", linestyle="--", linewidth=2, alpha=0.3
            )
        for i in buy_idx_backtest:
            new_axis.axvline(
                x=xdata[i], color="red", linestyle="-", linewidth=2, alpha=0.3
            )
        for i in sell_idx_backtest:
            new_axis.axvline(
                x=xdata[i], color="blue", linestyle="-", linewidth=2, alpha=0.3
            )
        for i in buy_idx_logic:
            new_axis.axvline(
                x=xdata[i], color=(1, 0.2, 0), linestyle="-.", linewidth=2, alpha=0.3
            )
        for i in sell_idx_logic:
            new_axis.axvline(
                x=xdata[i], color=(0, 0.2, 1), linestyle="-.", linewidth=2, alpha=0.3
            )
        colors = color(len(signals.columns), palette="Set1")
        if len(signals.columns) > 1:
            for idx, col in enumerate(signals.columns[:-2]):
                new_axis.plot(
                    xdata,
                    signals[col],
                    color=colors[idx],
                    linewidth=3,
                    alpha=0.5,
                    label=_method2str(col),
                )
            plt.legend()
        new_axis.set_yticks([])
        new_axis = axlist[0].twinx()
        new_axis.plot(
            xdata,
            signals["signals"],
            color="black",
            linewidth=1,
        )
        new_axis.set_yticks([])
    if config.SAVE:
        return savefig(title, dpi)
    return None


def _method2str(method: str) -> str:
    if "_" in method:
        methods = method.split("_")
        for idx, met in enumerate(methods):
            methods[idx] = met[0].upper() + met[1:]
        return " ".join(methods)
    if "momentum" == method:
        return "Momentum"
    return method.upper()
