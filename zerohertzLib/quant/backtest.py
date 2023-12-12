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
from prettytable import PrettyTable

from zerohertzLib.plot import candle

from .strategies import bollinger_bands, momentum, moving_average, rsi


def backtest(
    data: pd.core.frame.DataFrame,
    signals: pd.core.frame.DataFrame,
    ohlc: Optional[str] = "",
    threshold: Optional[Union[int, Tuple[int]]] = 1,
    signal_key: Optional[str] = "signals",
) -> Dict[str, Any]:
    """전략에 의해 생성된 ``signals`` backtest

    Args:
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        signals (``pd.core.frame.DataFrame``): ``"signals"`` column이 포함된 data (다른 이름으로 지정했을 시 ``signal_key`` 사용)
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        threshold (``Optional[Union[int, Tuple[int]]]``): 매수, 매도를 결정할 ``signals`` 경계값
        signal_key (``Optional[str]``): ``"signals"`` 의 key 값

    Returns:
        ``Dict[str, Any]``: 총 수익률 (단위: %), 손실 거래 비율 (단위: %), 손실 거래 비율에 따른 수익률, 거래 정보 (매수가, 매도가, 수익률, 거래 기간), 총 매수, 총 매도

    Examples:
        >>> results = zz.quant.backtest(data, signals)
        >>> results.keys()
        dict_keys(['profit', 'loss', 'weighted_profit', 'transaction', 'buy', 'sell'])
        >>> results["transaction"].keys()
        dict_keys(['buy', 'sell', 'profit', 'period'])
    """
    if not isinstance(threshold, int):
        threshold_sell, threshold_buy = threshold
    else:
        threshold_sell, threshold_buy = -threshold, threshold
    wallet_buy = 0
    wallet_sell = 0
    stock = deque()
    transactions = defaultdict(list)
    signals["logic"] = 0
    for idx in data.index:
        position = signals.loc[idx, signal_key]
        if ohlc == "":
            price = data.loc[idx][:4].mean()
        else:
            price = data.loc[idx, ohlc]
        if position >= threshold_buy:
            signals.loc[idx, "logic"] = 1
            stock.append((price, idx))
            transactions["buy"].append(price)
            wallet_buy += price
        elif position <= threshold_sell:
            while stock:
                signals.loc[idx, "logic"] = -1
                price_buy, day = stock.popleft()
                transactions["sell"].append(price)
                transactions["profit"].append((price - price_buy) / price * 100)
                transactions["period"].append((idx - day).days)
                wallet_sell += price
        elif stock:
            # Rule
            # -10%의 손실 혹은 +20%의 이익이 발생하면 판매
            # -10%와 0% 사이의 주가 변동 발생 시 추가 구매
            # 구매 이후 1년 이상의 매도 signal이 없을 시 판매
            price_buy = sum(price_buy for (price_buy, _) in stock) / len(stock)
            profit = (price - price_buy) / price_buy * 100
            if profit <= -10 or profit >= 20:
                signals.loc[idx, "logic"] = -2
                while stock:
                    price_buy, day = stock.popleft()
                    transactions["sell"].append(price)
                    transactions["profit"].append((price - price_buy) / price * 100)
                    transactions["period"].append((idx - day).days)
                    wallet_sell += price
            elif profit <= 0:
                signals.loc[idx, "logic"] = 2
                stock.append((price, idx))
                transactions["buy"].append(price)
                wallet_buy += price
            while stock:
                price_buy, day = stock[0]
                if (idx - day).days > 365:
                    signals.loc[idx, "logic"] = -2
                    stock.popleft()
                    transactions["sell"].append(price)
                    transactions["profit"].append((price - price_buy) / price * 100)
                    transactions["period"].append((idx - day).days)
                    wallet_sell += price
                else:
                    break
    while stock:
        price_buy, _ = stock.pop()
        wallet_buy -= price_buy
    if wallet_buy == 0 or len(transactions["profit"]) == 0:
        return {
            "profit": -100,
            "loss": 100,
            "weighted_profit": -100,
            "transaction": transactions,
            "buy": 0,
            "sell": 0,
        }
    wallet = (wallet_sell - wallet_buy) / wallet_buy * 100
    loss = []
    bad = 0
    for transaction in transactions["profit"]:
        if transaction < 0:
            bad += 1
    loss = bad / len(transactions["profit"]) * 100
    return {
        "profit": wallet,
        "loss": loss,
        "weighted_profit": wallet * (100 - loss),
        "transaction": transactions,
        "buy": wallet_buy,
        "sell": wallet_sell,
    }


def experiments(
    title: str,
    data: pd.core.frame.DataFrame,
    strategy: Callable[[Any], pd.core.frame.DataFrame],
    exps: List[List[Any]],
    ohlc: Optional[str] = "",
    vis: Optional[bool] = False,
    dpi: Optional[int] = 100,
    report: Optional[bool] = True,
) -> Dict[str, List[Any]]:
    """Full factorial design 기반의 backtest 수행 함수

    Args:
        title (``str``): 종목 이름
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        strategy (``Callable[[Any], pd.core.frame.DataFrame]``): Full factorial을 수행할 전략 함수
        exps (``List[List[Any]]``): 전략 함수에 입력될 변수들의 범위
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        vis (``Optional[bool]``): Candle chart 시각화 여부
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)
        report: (``Optional[bool]``): Experiment 결과 출력 여부

    Returns:
        ``Dict[str, List[Any]]``: 손실 거래 비율에 따른 수익률, ``signals``, parameters

    Examples:
        >>> exps = [[10, 20, 25, 30], [70, 75, 80, 85, 90], [14, 21, 31]]
        >>> results = zz.quant.experiments(title, data, zz.quant.rsi, exps)
        +----------------------+------------+------------+
        | EXP                  |     PROFIT | LOSS RATIO |
        +----------------------+------------+------------+
        | 10-70-14             |      5.65% |      0.00% |
        | 10-70-21             |   -100.00% |    100.00% |
        | ...                  |        ... |        ... |
        | 30-90-21             |     21.25% |      0.00% |
        | 30-90-31             |     20.98% |      0.00% |
        | ==================== | ========== | ========== |
        | WORST (10-70-21)     |   -100.00% |    100.00% |
        | BEST (25-90-31)      |     21.53% |      0.00% |
        | OPTIM (25-75-31)     |     21.53% |      0.00% |
        +----------------------+------------+------------+
        >>> results.keys()
        dict_keys(['profits', 'signals', 'exps_str', 'exps_tup'])
        >>> results["profits"]
        [21.530811750223275, ...]
        >>> results["signals"][0].columns
        Index(['RSI', 'signals', 'logic'], dtype='object')
        >>> results["exps_str"]
        ['25-75-31', ...]
        >>> results["exps_tup"]
        [(25, 75, 31), ...]
    """
    results = []
    if report:
        reports = PrettyTable(["EXP", "PROFIT", "LOSS RATIO"])
        reports.align["EXP"] = "l"
        reports.align["PROFIT"] = "r"
        reports.align["LOSS RATIO"] = "r"
    for exp in product(*exps):
        signals = strategy(data, *exp, ohlc=ohlc)
        backtest_results = backtest(data, signals, ohlc=ohlc)
        exp_str = "-".join(list(map(str, exp)))
        profit_total = backtest_results["profit"]
        loss_total = backtest_results["loss"]
        weighted_profit_total = backtest_results["weighted_profit"]
        if report:
            reports.add_row([exp_str, f"{profit_total:.2f}%", f"{loss_total:.2f}%"])
        if profit_total == 0:
            continue
        if vis:
            candle(data, f"{title}-{exp_str}", signals=signals, dpi=dpi)
        results.append(
            (
                profit_total,
                weighted_profit_total,
                {
                    "exps_tup": exp,
                    "exp_str": exp_str,
                    "signals": signals,
                    "profit_total": profit_total,
                    "loss_total": loss_total,
                },
            )
        )
    if report:
        reports.add_row(["=" * 20, "=" * 10, "=" * 10])
        results.sort(key=lambda x: x[0])
        reports.add_row(
            [
                f"WORST ({results[0][2]['exp_str']})",
                f"{results[0][2]['profit_total']:.2f}%",
                f"{results[0][2]['loss_total']:.2f}%",
            ]
        )
        reports.add_row(
            [
                f"BEST ({results[-1][2]['exp_str']})",
                f"{results[-1][2]['profit_total']:.2f}%",
                f"{results[-1][2]['loss_total']:.2f}%",
            ]
        )
    results.sort(key=lambda x: x[1], reverse=True)
    if report:
        reports.add_row(
            [
                f"OPTIM ({results[0][2]['exp_str']})",
                f"{results[0][2]['profit_total']:.2f}%",
                f"{results[0][2]['loss_total']:.2f}%",
            ]
        )
        print(reports)
    profits, signals, exps_str, exps_tup = [], [], [], []
    for result in results:
        profits.append(result[2]["profit_total"])
        signals.append(result[2]["signals"])
        exps_str.append(result[2]["exp_str"])
        exps_tup.append(result[2]["exps_tup"])
    return {
        "profits": profits,
        "signals": signals,
        "exps_str": exps_str,
        "exps_tup": exps_tup,
    }


class Experiments:
    """Full factorial design 기반의 backtest 수행 class

    Args:
        title (``str``): 종목 이름
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        vis (``Optional[bool]``): Candle chart 시각화 여부
        report: (``Optional[bool]``): Experiment 결과 출력 여부

    Examples:
        >>> experiments = zz.quant.Experiments(title, data)
    """

    def __init__(
        self,
        title: str,
        data: pd.core.frame.DataFrame,
        ohlc: Optional[str] = "",
        vis: Optional[bool] = False,
        report: Optional[bool] = True,
    ) -> None:
        self.title = title
        self.data = data
        self.ohlc = ohlc
        self.vis = vis
        self.report = report
        self.exps_moving_average = [
            [5, 10, 15, 20, 25],
            [60, 65, 70, 75, 80],
            [50, 100, 150, 200],
        ]
        self.exps_rsi = [[10, 20, 25, 30], [70, 75, 80, 85, 90], [14, 21, 31]]
        self.exps_bollinger_bands = [[20, 30, 40], [1.9, 1.95, 2, 2.05, 2.1]]
        self.exps_momentum = [[5, 10, 15], [5, 10, 15], [10, 25, 50, 75]]

    def _experiments(
        self,
        strategy: Callable[[Any], pd.core.frame.DataFrame],
        exps: List[List[Any]],
    ) -> Dict[str, List[Any]]:
        return experiments(
            self.title,
            self.data,
            strategy,
            exps,
            ohlc=self.ohlc,
            vis=self.vis,
            report=self.report,
        )

    def moving_average(self, exps: List[List[Any]] = None) -> Dict[str, List[Any]]:
        """Moving average 전략 실험

        Args:
            exps (``List[List[Any]]``): 전략 함수에 입력될 변수들의 범위

        Returns:
            ``Dict[str, List[Any]]``: 손실 거래 비율에 따른 수익률, ``signals``, parameters
        """
        if exps is None:
            exps = self.exps_moving_average
        return self._experiments(moving_average, exps)

    def rsi(self, exps: List[List[Any]] = None) -> Dict[str, List[Any]]:
        """RSI 전략 실험

        Args:
            exps (``List[List[Any]]``): 전략 함수에 입력될 변수들의 범위

        Returns:
            ``Dict[str, List[Any]]``: 손실 거래 비율에 따른 수익률, ``signals``, parameters
        """
        if exps is None:
            exps = self.exps_rsi
        return self._experiments(rsi, exps)

    def bollinger_bands(self, exps: List[List[Any]] = None) -> Dict[str, List[Any]]:
        """Bollinger bands 전략 실험

        Args:
            exps (``List[List[Any]]``): 전략 함수에 입력될 변수들의 범위

        Returns:
            ``Dict[str, List[Any]]``: 손실 거래 비율에 따른 수익률, ``signals``, parameters
        """
        if exps is None:
            exps = self.exps_bollinger_bands
        return self._experiments(bollinger_bands, exps)

    def momentum(self, exps: List[List[Any]] = None) -> Dict[str, List[Any]]:
        """Momentum 전략 실험

        Args:
            exps (``List[List[Any]]``): 전략 함수에 입력될 변수들의 범위

        Returns:
            ``Dict[str, List[Any]]``: 손실 거래 비율에 따른 수익률, ``signals``, parameters
        """
        if exps is None:
            exps = self.exps_momentum
        return self._experiments(momentum, exps)
