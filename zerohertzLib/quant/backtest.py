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
from typing import Any, Callable, List, Tuple

import pandas as pd

from zerohertzLib.plot import candle


def backtest(
    data: pd.core.frame.DataFrame, signals: pd.core.frame.DataFrame
) -> Tuple[float, List[Tuple[float]]]:
    """전략에 의해 생성된 ``signals`` backtest

    Args:
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        signals (``pd.core.frame.DataFrame``): ``"positions"`` column이 포함된 data

    Returns:
        ``Tuple[float, List[Tuple[float]]]``: 수익률 (단위: %)과 매수, 매도 log

    Examples:
        >>> zz.quant.backtest(data, signals)
        (4.23899999999999, [(10000000.0, 61500.0, 162.0), (10048600.0, -61800.0, 0), ...])
    """
    wallet = 10_000_000
    stock = 0
    logs = []
    for price, signal in zip(data["Close"], signals["positions"]):
        if signal == -1:
            cnt = wallet // price
            stock += cnt
            wallet -= price * cnt
            logs.append((price * stock + wallet, price, stock))
        else:
            if stock > 0:
                wallet += price * stock
                logs.append((wallet, -price, 0))
                stock = 0
    if stock > 0:
        wallet += data["Close"][-1] * stock
        logs.append((wallet, -data["Close"][-1], 0))
    return wallet / 10_000_000 * 100 - 100, logs


def experiments(
    title: str,
    data: pd.core.frame.DataFrame,
    strategy: Callable[[Any], pd.core.frame.DataFrame],
    exps: List[List[Any]],
) -> str:
    """Full factorial design 기반의 backtest 수행 함수

    Args:
        title(``str``): 종목 이름
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        strategy (``Callable[[Any], pd.core.frame.DataFrame]``): Full factorial을 수행할 전략 함수
        exps (``List[List[Any]]``): 전략 함수에 입력될 변수들의 범위

    Returns:
        ``str``: 실험의 결과

    Examples:
        >>> exps = [[5, 10, 15], [40, 50, 60]]
        >>> zz.quant.experiments(title, data, zz.quant.moving_average, exps)
        5-40:   -1.68%  9,832,000
        5-50:   0.41%   10,040,900
        5-60:   -3.06%  9,693,700
        10-40:  0.24%   10,023,700
        10-50:  -0.72%  9,927,800
        10-60:  2.81%   10,281,300
        15-40:  2.67%   10,267,300
        15-50:  4.24%   10,423,900
        15-60:  -3.68%  9,632,400
        ==================================================
        BEST:   15-50   4.24%
        WORST:  15-60   -3.68%
        ==================================================
    """
    results = []
    reports = []
    for exp in product(*exps):
        signals = strategy(data, *exp)
        profit, logs = backtest(data, signals)
        exp_str = "-".join(list(map(str, exp)))
        candle(
            data,
            f"{title}-{exp_str}",
            signals={
                "Signals": signals["signals"],
                "Positions": signals["positions"],
            },
        )
        results.append((exp_str, profit))
        if profit == 0 and len(logs) == 0:
            reports.append(f"{exp_str}:\t{profit:.2f}%")
            continue
        reports.append(f"{exp_str}:\t{profit:.2f}%\t{logs[-1][0]:,.0f}")
    results.sort(key=lambda x: x[1])
    reports.append("=" * 50)
    reports.append(f"BEST:\t{results[-1][0]}\t{results[-1][1]:.2f}%")
    reports.append(f"WORST:\t{results[0][0]}\t{results[0][1]:.2f}%")
    reports.append("=" * 50)
    reports = "\n".join(reports)
    print(reports)
    return reports
