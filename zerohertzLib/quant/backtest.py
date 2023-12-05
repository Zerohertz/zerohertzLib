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

from collections import defaultdict, deque
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from zerohertzLib.plot import candle

from .strategies import bollinger_bands, momentum, moving_average, rsi


def backtest(
    data: Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]],
    signals: Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]],
    ohlc: Optional[str] = "",
    threshold: Optional[int] = 1,
) -> Dict[str, Union[float, List[float]]]:
    """전략에 의해 생성된 ``signals`` backtest

    Args:
        data (``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``): OHLCV (Open, High, Low, Close, Volume) data
        signals (``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``): ``"signals"`` column이 포함된 data
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        threshold (``Optional[int]``): 매수, 매도를 결정할 ``signals`` 경계값

    Returns:
        ``Dict[str, Union[float, List[float]]]``: 수익률 (단위: %), 손실 거래 비율 (단위: %), 손실 거래 비율에 따른 수익률, 거래 내역

    Examples:
        >>> results = zz.quant.backtest(data, signals)
        >>> results.keys()
        dict_keys(['profit', 'loss', 'weighted_profit', 'transaction'])
    """
    if isinstance(data, list):
        results = defaultdict(list)
        for data_, signals_ in zip(data, signals):
            result = backtest(data_, signals_, ohlc, threshold)
            for key, value in result.items():
                results[key].append(value)
        return results
    wallet = 10_000_000_000
    stock = deque()
    transactions = []
    for idx in data.index:
        if ohlc == "":
            price = data.loc[idx][:4].mean()
        else:
            price = data.loc[idx, ohlc]
        position = signals.loc[idx, "signals"]
        if position >= threshold:
            cnt = wallet // price
            stock.append((price, cnt))
            wallet -= price * cnt
        elif position <= -threshold:
            while stock:
                price_buy, cnt = stock.pop()
                transactions.append((price - price_buy) / price * 100)
                wallet += price * cnt
    while stock:
        price_buy, cnt = stock.pop()
        if ohlc == "":
            wallet += data.iloc[:, :4].mean(1)[-1] * cnt
        else:
            wallet += data[ohlc][-1] * cnt
    wallet = wallet / 10_000_000_000 * 100 - 100
    loss = []
    if len(transactions) == 0:
        loss = 0
    else:
        bad = 0
        for transaction in transactions:
            if transaction < 0:
                bad += 1
        loss = bad / len(transactions) * 100
    return {
        "profit": wallet,
        "loss": loss,
        "weighted_profit": wallet * (100 - loss),
        "transaction": transactions,
    }


def experiments(
    title: Union[str, List[str]],
    data: Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]],
    strategy: Callable[
        [Any], Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]
    ],
    exps: List[List[Any]],
    ohlc: Optional[str] = "",
    vis: Optional[bool] = False,
    dpi: Optional[int] = 100,
) -> Tuple[str, List[Union[int, float]], pd.core.frame.DataFrame]:
    """Full factorial design 기반의 backtest 수행 함수

    Args:
        title(``Union[str, List[str]]``): 종목 이름
        data (``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``): OHLCV (Open, High, Low, Close, Volume) data
        strategy (``Callable[[Any], Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]]``): Full factorial을 수행할 전략 함수
        exps (``List[List[Any]]``): 전략 함수에 입력될 변수들의 범위
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        vis (``Optional[bool]``): Candle chart 시각화 여부
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        ``Tuple[str, List[Union[int, float]], pd.core.frame.DataFrame]``: Report와 손실 거래 비율에 따른 수익률 기반 최적 parameter와 ``signals``

    Examples:
        Single:

        >>> exps = [[10, 20, 30], [50, 60, 70]]
        >>> zz.quant.experiments(title, data, zz.quant.moving_average, exps)
        10-50:  15.47%  20.00%
        ...
        30-70:  13.95%  0.00%
        ====================================================================================================
        BEST:   20-50   31.66%  25.00%
        OPTIM:  20-60   30.96%  0.00%
        WORST:  30-60   9.83%   25.00%
        ====================================================================================================

        Multi:

        >>> zz.quant.experiments(title, data, zz.quant.moving_average, exps)
        10-50:  10.97%  6.67%                   15.47%  -14.22% 31.66%
        ...
        30-70:  4.13%   16.67%                  13.95%  -9.61%  8.04%
        ====================================================================================================
        BEST:   20-50   24.10%  8.33%           31.66%  0.79%   39.84%
        OPTIM:  20-50   24.10%  8.33%           31.66%  0.79%   39.84%
        WORST:  30-60   -1.99%  41.67%          9.83%   -14.49% -1.32%
        ====================================================================================================
    """
    results = []
    reports = []
    for exp in product(*exps):
        signals = strategy(data, *exp, ohlc=ohlc)
        backtest_results = backtest(data, signals, ohlc=ohlc)
        name = "-".join(list(map(str, exp)))
        if isinstance(data, list):
            profit_total = sum(backtest_results["profit"]) / len(
                backtest_results["profit"]
            )
            loss_total = sum(backtest_results["loss"]) / len(backtest_results["loss"])
            weighted_profit_total = sum(backtest_results["weighted_profit"]) / len(
                backtest_results["weighted_profit"]
            )
        else:
            profit_total = backtest_results["profit"]
            loss_total = backtest_results["loss"]
            weighted_profit_total = backtest_results["weighted_profit"]
        if profit_total == 0:
            reports.append(f"{name}:\t{profit_total:.2f}%\t{loss_total:.2f}")
            continue
        profilt_all = ""
        if isinstance(data, list):
            for profit in backtest_results["profit"]:
                profilt_all += f"\t{profit:.2f}%"
        if vis:
            if isinstance(data, list):
                candle(data[0], f"{title[0]}-{name}", signals=signals[0], dpi=dpi)
            else:
                candle(data, f"{title}-{name}", signals=signals, dpi=dpi)
        reports.append(
            f"{name}:\t{profit_total:.2f}%\t{loss_total:.2f}%\t\t{profilt_all}"
        )
        results.append(
            (
                profit_total,
                weighted_profit_total,
                {
                    "name": name,
                    "profit_total": profit_total,
                    "loss_total": loss_total,
                    "profit_all": profilt_all,
                    "exp": exp,
                    "signals": signals,
                },
            )
        )
    results.sort(key=lambda x: x[0])
    best = f"BEST:\t{results[-1][2]['name']}\t{results[-1][2]['profit_total']:.2f}%\t{results[-1][2]['loss_total']:.2f}%\t{results[-1][2]['profit_all']}"
    worst = f"WORST:\t{results[0][2]['name']}\t{results[0][2]['profit_total']:.2f}%\t{results[0][2]['loss_total']:.2f}%\t{results[0][2]['profit_all']}"
    results.sort(key=lambda x: x[1])
    optim = f"OPTIM:\t{results[-1][2]['name']}\t{results[-1][2]['profit_total']:.2f}%\t{results[-1][2]['loss_total']:.2f}%\t{results[-1][2]['profit_all']}"
    reports.append("=" * 100)
    reports.append(best)
    reports.append(optim)
    reports.append(worst)
    reports.append("=" * 100)
    reports = "\n".join(reports)
    print(reports)
    return reports, results[-1][2]["exp"], results[-1][2]["signals"]


class Experiments:
    """Full factorial design 기반의 backtest 수행 class

    Args:
        title(``Union[str, List[str]]``): 종목 이름
        data (``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``): OHLCV (Open, High, Low, Close, Volume) data
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름

    Examples:
        >>> experiments = zz.quant.Experiments(title, data)
    """

    def __init__(
        self,
        title: Union[str, List[str]],
        data: Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]],
        ohlc: Optional[str] = "",
    ) -> None:
        self.title = title
        self.data = data
        self.ohlc = ohlc
        self.exps_moving_average = [[10, 20, 30], [50, 60, 70], [100, 500, 1000]]
        self.exps_rsi = [[10, 20, 30], [70, 80, 90], [7, 14, 21]]
        self.exps_bollinger_bands = [[7, 10, 14, 20], [1.9, 2, 2.1]]
        self.exps_momentum = [[5, 10, 15], [5, 10, 15], [25, 50, 75]]

    def _experiments(
        self,
        strategy: Callable[
            [Any], Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]
        ],
        exps: List[List[Any]],
    ):
        return experiments(self.title, self.data, strategy, exps, ohlc=self.ohlc)

    def moving_average(
        self, exps: List[List[Any]] = None
    ) -> Tuple[str, List[Union[int, float]], pd.core.frame.DataFrame]:
        """Moving average 전략 실험

        Args:
            exps (``List[List[Any]]``): 전략 함수에 입력될 변수들의 범위

        Returns:
            ``Tuple[str, List[Union[int, float]], pd.core.frame.DataFrame]``: Report와 손실 거래 비율에 따른 수익률 기반 최적 parameter와 ``signals``
        """
        if exps is None:
            exps = self.exps_moving_average
        return self._experiments(moving_average, exps)

    def rsi(
        self, exps: List[List[Any]] = None
    ) -> Tuple[str, List[Union[int, float]], pd.core.frame.DataFrame]:
        """RSI 전략 실험

        Args:
            exps (``List[List[Any]]``): 전략 함수에 입력될 변수들의 범위

        Returns:
            ``Tuple[str, List[Union[int, float]], pd.core.frame.DataFrame]``: Report와 손실 거래 비율에 따른 수익률 기반 최적 parameter와 ``signals``
        """
        if exps is None:
            exps = self.exps_rsi
        return self._experiments(rsi, exps)

    def bollinger_bands(
        self, exps: List[List[Any]] = None
    ) -> Tuple[str, List[Union[int, float]], pd.core.frame.DataFrame]:
        """Bollinger bands 전략 실험

        Args:
            exps (``List[List[Any]]``): 전략 함수에 입력될 변수들의 범위

        Returns:
            ``Tuple[str, List[Union[int, float]], pd.core.frame.DataFrame]``: Report와 손실 거래 비율에 따른 수익률 기반 최적 parameter와 ``signals``
        """
        if exps is None:
            exps = self.exps_bollinger_bands
        return self._experiments(bollinger_bands, exps)

    def momentum(
        self, exps: List[List[Any]] = None
    ) -> Tuple[str, List[Union[int, float]], pd.core.frame.DataFrame]:
        """Momentum 전략 실험

        Args:
            exps (``List[List[Any]]``): 전략 함수에 입력될 변수들의 범위

        Returns:
            ``Tuple[str, List[Union[int, float]], pd.core.frame.DataFrame]``: Report와 손실 거래 비율에 따른 수익률 기반 최적 parameter와 ``signals``
        """
        if exps is None:
            exps = self.exps_momentum
        return self._experiments(momentum, exps)
