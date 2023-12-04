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

import matplotlib as mpl
import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt

from .util import color, savefig


def plot(
    xdata: List[Union[int, float]],
    ydata: Dict[str, List[Union[int, float]]],
    xlab: Optional[str] = "x축 [단위]",
    ylab: Optional[str] = "y축 [단위]",
    xlim: Optional[List[Union[int, float]]] = None,
    ylim: Optional[List[Union[int, float]]] = None,
    ncol: Optional[int] = 1,
    title: Optional[str] = "tmp",
    figsize: Optional[Tuple[int]] = (15, 10),
    dpi: Optional[int] = 300,
    save: Optional[bool] = True,
) -> None:
    """List와 Dictionary로 입력받은 데이터를 line chart로 시각화

    Args:
        xdata (``List[Union[int, float]]``): 입력 데이터 (X축)
        ydata (``Dict[str, List[Union[int, float]]]``): 입력 데이터 (Y축)
        xlab (``Optional[str]``): Graph에 출력될 X축 label
        ylab (``Optional[str]``): Graph에 출력될 Y축 label
        xlim (``Optional[List[Union[int, float]]]``): Graph에 출력될 X축 limit
        ylim (``Optional[List[Union[int, float]]]``): Graph에 출력될 Y축 limit
        ncol (``Optional[int]``): Graph에 표시될 legend 열의 수
        title (``Optional[str]``): Graph에 표시될 제목 및 file 이름
        figsize (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)
        save (``Optional[bool]``): Graph 저장 여부

    Returns:
        ``None``: 현재 directory에 바로 graph 저장

    Examples:
        >>> xdata = [i for i in range(20)]
        >>> ydata = {"테란": list(np.random.rand(20) * 10), "저그": list(np.random.rand(20) * 10 + 1), "프로토스": list(np.random.rand(20) * 10 + 2)}
        >>> zz.plot.plot(xdata, ydata, xlab="시간 [초]", ylab="성적 [점]", title="Star Craft")

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/280603766-22a0f42c-91b0-4f34-aa73-29de6fdbd4e9.png
            :alt: Visualzation Result
            :align: center
            :width: 500px
    """
    colors = color(len(ydata))
    if save:
        plt.figure(figsize=figsize)
    # list(plt.Line2D.lineStyles.keys())
    linestyle = ["-", "--", "-.", ":"]
    # import matplotlib.markers as mmarkers
    # markers = list(mmarkers.MarkerStyle.markers.keys())
    marker = ["o", "v", "^", "s", "p", "*", "x"]
    for i, (key, value) in enumerate(ydata.items()):
        plt.plot(
            xdata,
            value,
            color=colors[i],
            linestyle=linestyle[i % len(linestyle)],
            linewidth=2,
            marker=marker[i % len(marker)],
            markersize=12,
            label=key,
        )
    plt.grid(zorder=0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title, fontsize=25)
    plt.legend(ncol=ncol)
    if save:
        savefig(title, dpi)


def candle(
    data: pd.core.frame.DataFrame,
    title: Optional[str] = "tmp",
    figsize: Optional[Tuple[int]] = (18, 10),
    dpi: Optional[int] = 300,
    save: Optional[bool] = True,
    signals: Optional[Dict[str, Any]] = None,
) -> Tuple[mpl.figure.Figure, List[mpl.axes._axes.Axes]]:
    """OHLCV (Open, High, Low, Close, Volume) data에 따른 candle chart

    Args:
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        title (``Optional[str]``): Graph에 표시될 제목 및 file 이름
        figsize (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)
        save (``Optional[bool]``): Graph 저장 여부
        signals (``Optional[Dict[str, Any]]``): 추가적으로 plot할 data

    Returns:
        ``None``: 현재 directory에 바로 graph 저장

    Examples:
        >>> broker = zz.api.KoreaInvestment()
        >>> samsung = broker.get_ohlcv("005930", "D", "20221205")
        >>> title, data = broker.response2ohlcv(samsung)
        >>> signals = zz.quant.moving_average(data)
        >>> zz.plot.candle(data, title, signals=signals)
        >>> apple = broker.get_ohlcv("AAPL", kor=False)
        >>> title, data = broker.response2ohlcv(apple)
        >>> zz.plot.candle(data, title)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/287729767-fc7b51c2-13c3-496b-8bda-309a4059bd56.png
            :alt: Visualzation Result
            :align: center
            :width: 800px
    """
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
    bands = _bollinger_bands(data)
    bollinger = mpf.make_addplot(bands[["lower_band", "upper_band"]], type="line")
    _, axlist = mpf.plot(
        data,
        type="candle",
        mav=(5, 10, 20),
        volume=True,
        figsize=figsize,
        title=title,
        style=style,
        returnfig=True,
        addplot=bollinger,
    )
    if signals is not None:
        new_axis = axlist[0].twinx()
        xdata = axlist[0].get_lines()[0].get_xdata()
        # new_axis.plot(
        #     xdata,
        #     signals["signals"],
        #     color="black",
        #     linewidth=1,
        #     alpha=0.8,
        # )
        buy_indices = []
        sell_indices = []
        for idx, pos in enumerate(signals["positions"]):
            if pos == -1:
                buy_indices.append(idx)
            elif pos == 1:
                sell_indices.append(idx)
        for i in buy_indices:
            new_axis.axvline(
                x=xdata[i], color="blue", linestyle="--", linewidth=2, alpha=0.5
            )
        for i in sell_indices:
            new_axis.axvline(
                x=xdata[i], color="red", linestyle="--", linewidth=2, alpha=0.5
            )
        new_axis.set_yticks([])
        # new_axis.set_yticks([-1, 0, 1])
        # new_axis.set_yticklabels(["Buy", "", "Sell"])
        # plt.legend()
    if save:
        savefig(title, dpi)


def _bollinger_bands(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """Bollinger band 계산 함수

    Args:
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data

    Returns:
        ``pd.core.frame.DataFrame``: Bollinger band
    """
    bands = pd.DataFrame(index=data.index)
    bands["middle_band"] = data.iloc[:, :4].mean(1).rolling(window=20).mean()
    std_dev = data.iloc[:, :4].mean(1).rolling(window=20).std()
    bands["upper_band"] = bands["middle_band"] + (std_dev * 2)
    bands["lower_band"] = bands["middle_band"] - (std_dev * 2)
    return bands
