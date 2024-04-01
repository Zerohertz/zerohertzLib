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

from typing import Any, Dict, List, Optional, Tuple, Union

import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .util import _color, color, savefig


def plot(
    xdata: List[Union[int, float]],
    ydata: Dict[str, List[Union[int, float]]],
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    xlim: Optional[List[Union[int, float]]] = None,
    ylim: Optional[List[Union[int, float]]] = None,
    stacked: Optional[bool] = False,
    ncol: Optional[int] = 1,
    title: Optional[str] = "tmp",
    colors: Optional[Union[str, List]] = None,
    markersize: Optional[int] = 12,
    figsize: Optional[Tuple[int]] = (15, 10),
    dpi: Optional[int] = 300,
    save: Optional[bool] = True,
) -> str:
    """List와 Dictionary로 입력받은 data를 line chart로 시각화

    Args:
        xdata (``List[Union[int, float]]``): 입력 data (X축)
        ydata (``Dict[str, List[Union[int, float]]]``): 입력 data (Y축)
        xlab (``Optional[str]``): Graph에 출력될 X축 label
        ylab (``Optional[str]``): Graph에 출력될 Y축 label
        xlim (``Optional[List[Union[int, float]]]``): Graph에 출력될 X축 limit
        ylim (``Optional[List[Union[int, float]]]``): Graph에 출력될 Y축 limit
        stacked (``Optional[bool]``): Stacked plot 여부
        ncol (``Optional[int]``): Graph에 표시될 legend 열의 수
        title (``Optional[str]``): Graph에 표시될 제목 및 file 이름
        colors (``Optional[Union[str, List]]``): 각 요소의 색
        markersize (``Optional[int]``): Graph에 표시될 marker의 size
        figsize (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)
        save (``Optional[bool]``): Graph 저장 여부

    Returns:
        ``str``: 저장된 graph의 절대 경로

    Examples:
        ``stacked=False``:
            >>> xdata = [i for i in range(20)]
            >>> ydata = {"Terran": list(np.random.rand(20) * 10), "Zerg": list(np.random.rand(20) * 10 + 1), "Protoss": list(np.random.rand(20) * 10 + 2)}
            >>> zz.plot.plot(xdata, ydata, xlab="Time [Sec]", ylab="Scores", title="Star Craft")

            .. image:: _static/examples/dynamic/plot.plot.1.png
                :align: center
                :width: 500px

        ``stacked=True``:
            >>> ydata["Total"] = [sum(data) + 10 for data in zip(ydata["Terran"], ydata["Protoss"], ydata["Zerg"])]
            >>> zz.plot.plot(xdata, ydata, xlab="Time [Sec]", ylab="Scores", stacked=True, title="Star Craft")

            .. image:: _static/examples/dynamic/plot.plot.2.png
                :align: center
                :width: 500px
    """
    colors = _color(ydata, colors)
    if save:
        plt.figure(figsize=figsize)
    # import matplotlib.markers as mmarkers
    # markers = list(mmarkers.MarkerStyle.markers.keys())
    marker = ["o", "v", "^", "s", "p", "*", "x"]
    if stacked:
        bias = np.zeros(len(xdata))
        linestyle = ["-"]
    else:
        # list(plt.Line2D.lineStyles.keys())
        linestyle = ["-", "--", "-.", ":"]
    for i, (key, value) in enumerate(ydata.items()):
        if stacked:
            if key == "Total":
                colors[i] = (0.5, 0.5, 0.5)
            else:
                value = np.array(value) + bias
        plt.plot(
            xdata,
            value,
            color=colors[i],
            linestyle=linestyle[i % len(linestyle)],
            linewidth=2,
            marker=marker[i % len(marker)],
            markersize=markersize,
            label=key,
        )
        if stacked:
            plt.fill_between(xdata, value, bias, color=colors[i], alpha=0.5)
            bias = value
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
    if save:
        return savefig(title, dpi)
    return None


def candle(
    data: pd.DataFrame,
    title: Optional[str] = "tmp",
    figsize: Optional[Tuple[int]] = (18, 10),
    signals: Optional[Dict[str, Any]] = None,
    threshold: Optional[Union[int, Tuple[int]]] = 1,
    dpi: Optional[int] = 300,
    save: Optional[bool] = True,
) -> str:
    """OHLCV (Open, High, Low, Close, Volume) data에 따른 candle chart

    Note:
        - 적색: 매수
        - 청색: 매도
        - 실선 (``-``): Backtest 시 signal이 존재하는 매수, 매도
        - 파선 (``--``): Backtest 시 사용하지 않은 signal의 매수, 매도
        - 일점쇄선 (``-.``): Backtest logic에 의한 매수, 매도

    Args:
        data (``pd.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        title (``Optional[str]``): Graph에 표시될 제목 및 file 이름
        figsize (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        signals (``Optional[Dict[str, Any]]``): 추가적으로 plot할 data
        threshold (``Optional[Union[int, Tuple[int]]]``): 매수, 매도를 결정할 ``signals`` 경계값
        dpi (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)
        save (``Optional[bool]``): Graph 저장 여부

    Returns:
        ``str``: 저장된 graph의 절대 경로

    Examples:
        >>> zz.plot.candle(data, title)
        >>> signals = zz.quant.macd(data)
        >>> zz.plot.candle(data, "MACD", signals=signals)

        .. image:: _static/examples/dynamic/plot.candle.png
            :align: center
            :width: 600px
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
    if save:
        return savefig(title, dpi)
    return None


# def _bollinger_bands(data: pd.DataFrame) -> pd.DataFrame:
#     """Bollinger band 계산 함수

#     Args:
#         data (``pd.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data

#     Returns:
#         ``pd.DataFrame``: Bollinger band
#     """
#     bands = pd.DataFrame(index=data.index)
#     bands["middle_band"] = data.iloc[:, :4].mean(1).rolling(window=20).mean()
#     std_dev = data.iloc[:, :4].mean(1).rolling(window=20).std()
#     bands["upper_band"] = bands["middle_band"] + (std_dev * 2)
#     bands["lower_band"] = bands["middle_band"] - (std_dev * 2)
#     return bands


def _method2str(method: str) -> str:
    if "_" in method:
        methods = method.split("_")
        for idx, met in enumerate(methods):
            methods[idx] = met[0].upper() + met[1:]
        return " ".join(methods)
    if "momentum" == method:
        return "Momentum"
    return method.upper()
