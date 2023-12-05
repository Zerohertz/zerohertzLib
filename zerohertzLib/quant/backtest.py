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

from itertools import product
from typing import Any, Callable, List, Optional, Tuple, Union

import pandas as pd

from zerohertzLib.plot import candle

from .strategies import bollinger_bands, momentum, moving_average, rsi


def backtest(
    data: Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]],
    signals: Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]],
    ohlc: Optional[str] = "Open",
) -> List[float]:
    """전략에 의해 생성된 ``signals`` backtest

    Args:
        data (``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``): OHLCV (Open, High, Low, Close, Volume) data
        signals (``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``): ``"positions"`` column이 포함된 data
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름

    Returns:
        ``List[float]``: 수익률 (단위: %)

    Examples:
        >>> zz.quant.backtest(data, signals)
        (4.23899999999999, [(10000000.0, 61500.0, 162.0), (10048600.0, -61800.0, 0), ...])
    """
    if isinstance(data, pd.core.frame.DataFrame) and isinstance(
        signals, pd.core.frame.DataFrame
    ):
        data = [data]
        signals = [signals]
    if not (isinstance(data, list) and isinstance(signals, list)):
        raise TypeError("The 'data' and 'signals' must be 'list'")
    for data_, signals_ in zip(data, signals):
        if not (
            (data[0].index == data_.index).all()
            and (data_.index == signals_.index).all()
        ):
            raise ValueError("The 'index' of 'data' and 'signals' must be same")
    wallet = [10_000_000_000 // len(data) for _ in range(len(data))]
    wallet[0] += 10_000_000_000 % len(data)
    wallet_init = wallet.copy()
    stock = [0 for _ in range(len(data))]
    for idx in data[0].index:
        for i, (data_, signals_) in enumerate(zip(data, signals)):
            price = data_.loc[idx, ohlc]
            position = signals_.loc[idx, "positions"]
            if position == 1:
                cnt = wallet[i] // price
                stock[i] += cnt
                wallet[i] -= price * cnt
            elif position == -1:
                if stock[i] > 0:
                    wallet[i] += price * stock[i]
                    stock[i] = 0
    for i, data_ in enumerate(data):
        if stock[i] > 0:
            wallet[i] += data_[ohlc][-1] * stock[i]
        wallet[i] = wallet[i] / wallet_init[i] * 100 - 100
    return wallet


def experiments(
    title: Union[str, List[str]],
    data: Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]],
    strategy: Callable[
        [Any], Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]
    ],
    exps: List[List[Any]],
) -> str:
    """Full factorial design 기반의 backtest 수행 함수

    Args:
        title(``Union[str, List[str]]``): 종목 이름
        data (``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``): OHLCV (Open, High, Low, Close, Volume) data
        strategy (``Callable[[Any], Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]]``): Full factorial을 수행할 전략 함수
        exps (``List[List[Any]]``): 전략 함수에 입력될 변수들의 범위

    Returns:
        ``str``: 실험의 결과

    Examples:
        >>> exps = [[10, 15, 20], [40, 50, 60]]
        >>> zz.quant.experiments(title, data, zz.quant.moving_average, exps)
        10-40:  8.85%                   13.92%  -10.43% 25.47%  4.92%   10.38%
        ...
        20-60:  15.98%                  33.33%  -3.62%  29.45%  4.45%   16.27%
        ====================================================================================================
        BEST:   20-50   16.75%          29.44%  -2.38%  38.63%  -0.76%  18.82%
        WORST:  10-40   8.85%           13.92%  -10.43% 25.47%  4.92%   10.38%
        ====================================================================================================
    """
    results = []
    reports = []
    for exp in product(*exps):
        signals = strategy(data, *exp)
        profit = backtest(data, signals)
        exp_str = "-".join(list(map(str, exp)))
        profit_total = sum(profit) / len(profit)
        if profit_total == 0:
            reports.append(f"{exp_str}:\t{profit_total:.2f}%")
            continue
        profilt_all = ""
        if isinstance(data, list):
            candle(
                data[0],
                f"{title[0]}-{exp_str}",
                signals=signals[0],
            )
            for profit_ in profit:
                profilt_all += f"\t{profit_:.2f}%"
        else:
            candle(
                data,
                f"{title}-{exp_str}",
                signals=signals,
            )
        reports.append(f"{exp_str}:\t{profit_total:.2f}%\t\t{profilt_all}")
        results.append((exp_str, profit_total, profilt_all))
    results.sort(key=lambda x: x[1])
    reports.append("=" * 100)
    reports.append(f"BEST:\t{results[-1][0]}\t{results[-1][1]:.2f}%\t{results[-1][2]}")
    reports.append(f"WORST:\t{results[0][0]}\t{results[0][1]:.2f}%\t{results[0][2]}")
    reports.append("=" * 100)
    reports = "\n".join(reports)
    print(reports)
    return reports


class Experiments:
    """Full factorial design 기반의 backtest 수행 class

    Args:
        title(``Union[str, List[str]]``): 종목 이름
        data (``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``): OHLCV (Open, High, Low, Close, Volume) data

    Returns:
        ``None``: ``print`` 로 backtest 결과 출력

    Examples:
        >>> experiments = zz.quant.Experiments(title, data)
        >>> experiments.moving_average()
        10-40:  8.85%                   13.92%  -10.43% 25.47%  4.92%   10.38%
        ...
        20-60:  15.98%                  33.33%  -3.62%  29.45%  4.45%   16.27%
        ====================================================================================================
        BEST:   20-50   16.75%          29.44%  -2.38%  38.63%  -0.76%  18.82%
        WORST:  10-40   8.85%           13.92%  -10.43% 25.47%  4.92%   10.38%
        ====================================================================================================
        >>> experiments.rsi()
        20-60:  11.03%                  10.31%  -24.07% 64.53%  -9.26%  13.66%
        ...
        40-80:  13.10%                  13.63%  18.55%  24.13%  5.34%   3.86%
        ====================================================================================================
        BEST:   30-70   21.17%          15.95%  40.18%  32.86%  0.57%   16.27%
        WORST:  20-80   7.38%           15.54%  -7.85%  35.18%  0.78%   -6.75%
        ====================================================================================================
        >>> experiments.bollinger_bands()
        7-1.9:  44.57%                  31.24%  86.12%  48.79%  17.93%  38.75%
        ...
        14-2.1: 23.27%                  51.23%  32.36%  9.79%   12.30%  10.68%
        ====================================================================================================
        BEST:   7-1.9   44.57%          31.24%  86.12%  48.79%  17.93%  38.75%
        WORST:  7-2.1   17.21%          20.25%  8.85%   16.72%  18.57%  21.68%
        ====================================================================================================
        >>> experiments.momentum()
        7:      -9.92%                  13.41%  -24.07% -14.46% -20.00% -4.48%
        ...
        14:     -4.76%                  4.22%   -28.19% 19.19%  -23.29% 4.28%
        ====================================================================================================
        BEST:   14      -4.76%          4.22%   -28.19% 19.19%  -23.29% 4.28%
        WORST:  7       -9.92%          13.41%  -24.07% -14.46% -20.00% -4.48%
        ====================================================================================================
    """

    def __init__(
        self,
        title: Union[str, List[str]],
        data: Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]],
    ) -> None:
        self.title = title
        self.data = data

    def moving_average(
        self, exps: List[List[Any]] = [[10, 15, 20], [40, 50, 60]]
    ) -> None:
        experiments(self.title, self.data, moving_average, exps)

    def rsi(self, exps: List[List[Any]] = [[20, 30, 40], [60, 70, 80]]) -> None:
        experiments(self.title, self.data, rsi, exps)

    def bollinger_bands(
        self, exps: List[List[Any]] = [[7, 10, 14], [1.9, 2, 2.1]]
    ) -> None:
        experiments(self.title, self.data, bollinger_bands, exps)

    def momentum(self, exps: List[List[Any]] = [[7, 10, 14]]) -> None:
        experiments(self.title, self.data, momentum, exps)
