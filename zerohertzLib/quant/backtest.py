# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

from collections import defaultdict, deque
from itertools import product
from typing import Any, Callable

import pandas as pd
from prettytable import PrettyTable

from zerohertzLib.plot import candle

from .methods import bollinger_bands, macd, momentum, moving_average, rsi


def _backtest_buy(
    price: pd.Series,
    idx: pd.Timestamp,
    stock: deque,
    transactions: dict[str, list[Any]],
) -> None:
    stock.append((price[idx], idx))
    transactions["buy"].append(price[idx])


def _backtest_sell(
    price: pd.Series,
    idx: pd.Timestamp,
    stock: deque,
    transactions: dict[str, list[Any]],
) -> None:
    price_buy, day = stock.popleft()
    transactions["sell"].append(price[idx])
    transactions["profit"].append((price[idx] - price_buy) / price_buy * 100)
    transactions["period"].append((idx - day).days)


def backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    ohlc: str = "",
    threshold: int | tuple[int, int] = 1,
    signal_key: str = "signals",
) -> dict[str, Any]:
    """전략에 의해 생성된 `signals` backtest

    Args:
        data: OHLCV (Open, High, Low, Close, Volume) data
        signals: `"signals"` column이 포함된 data (다른 이름으로 지정했을 시 `signal_key` 사용)
        ohlc: 사용할 `data` 의 column 이름
        threshold: 매수, 매도를 결정할 `signals` 경계값
        signal_key: `"signals"` 의 key 값

    Returns:
        %), 손실 거래 비율 (단위: %), 손실 거래 비율에 따른 수익률, 거래 정보 (매수가, 매도가, 수익률, 거래 기간), 총 매수, 총 매도

    Examples:
        >>> results = zz.quant.backtest(data, signals)
        >>> results.keys()
        dict_keys(['profit', 'loss', 'weighted_profit', 'transaction', 'buy', 'sell'])
        >>> results["transaction"].keys()
        dict_keys(['buy', 'sell', 'profit', 'period'])
    """
    if isinstance(threshold, int):
        threshold_sell, threshold_buy = -threshold, threshold
    else:
        threshold_sell, threshold_buy = threshold
    wallet_buy = 0
    wallet_sell = 0
    wallet = [0, 0]
    stock = deque()
    transactions = defaultdict(list)
    signals["logic"] = 0
    if ohlc == "":
        price = data.iloc[:, :4].mean(1)
    else:
        price = data.loc[:, ohlc]
    for idx in data.index:
        position = signals.loc[idx, signal_key]
        if position >= threshold_buy:
            signals.loc[idx, "logic"] = 1
            _backtest_buy(price, idx, stock, transactions)
            wallet_buy += price[idx]
            wallet[0] += price[idx]
            wallet[1] += 1
        elif position <= threshold_sell:
            while stock:
                signals.loc[idx, "logic"] = -1
                _backtest_sell(price, idx, stock, transactions)
                wallet_sell += price[idx]
                wallet = [0, 0]
        elif stock:
            # Rule
            # -10%의 손실 혹은 +20%의 이익이 발생하면 판매
            # -10%와 0% 사이의 주가 변동 발생 시 추가 구매
            # 구매 이후 1년 이상의 매도 signal이 없을 시 판매
            price_buy = wallet[0] / wallet[1]
            profit = (price[idx] - price_buy) / price_buy * 100
            if profit <= -10 or profit >= 20:
                signals.loc[idx, "logic"] = -2
                while stock:
                    _backtest_sell(price, idx, stock, transactions)
                    wallet_sell += price[idx]
                wallet = [0, 0]
            elif profit <= 0:
                signals.loc[idx, "logic"] = 2
                _backtest_buy(price, idx, stock, transactions)
                wallet_buy += price[idx]
                wallet[0] += price[idx]
                wallet[1] += 1
            while stock:
                price_buy, day = stock[0]
                if (idx - day).days > 365:
                    signals.loc[idx, "logic"] = -2
                    _backtest_sell(price, idx, stock, transactions)
                    wallet_sell += price[idx]
                    wallet[0] -= price_buy
                    wallet[1] -= 1
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
    profit = (wallet_sell - wallet_buy) / wallet_buy * 100
    loss = []
    bad = 0
    for transaction in transactions["profit"]:
        if transaction < 0:
            bad += 1
    loss = bad / len(transactions["profit"]) * 100
    return {
        "profit": profit,
        "loss": loss,
        "weighted_profit": profit * (100 - loss),
        "transaction": transactions,
        "buy": wallet_buy,
        "sell": wallet_sell,
    }


def experiments(
    title: str,
    data: pd.DataFrame,
    method: Callable[[Any], pd.DataFrame],
    exps: list[list[Any]],
    ohlc: str = "",
    vis: bool = False,
    dpi: int = 100,
    report: bool = True,
) -> dict[str, list[Any]]:
    """Full factorial design 기반의 backtest 수행 function

    Args:
        title: 종목 이름
        data: OHLCV (Open, High, Low, Close, Volume) data
        method: Full factorial을 수행할 전략 function
        exps: 전략 function에 입력될 변수들의 범위
        ohlc: 사용할 `data` 의 column 이름
        vis: Candle chart 시각화 여부
        dpi: Graph 저장 시 DPI (Dots Per Inch)
        report: Experiment 결과 출력 여부

    Returns:
        손실 거래 비율에 따른 수익률, `signals`, parameters

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
        signals = method(data, *exp, ohlc=ohlc)
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
            candle(data[-500:], f"{title}-{exp_str}", signals=signals, dpi=dpi)
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
        title: 종목 이름
        data: OHLCV (Open, High, Low, Close, Volume) data
        ohlc: 사용할 `data` 의 column 이름
        vis: Candle chart 시각화 여부
        report: Experiment 결과 출력 여부

    Examples:
        >>> experiments = zz.quant.Experiments(title, data)
    """

    def __init__(
        self,
        title: str,
        data: pd.DataFrame,
        ohlc: str = "",
        vis: bool = False,
        report: bool = True,
    ) -> None:
        self.title = title
        self.data = data
        self.ohlc = ohlc
        self.vis = vis
        self.report = report
        self.exps_moving_average = [[20, 30, 40], [60, 70, 80], [0.0, 0.5, 1.0]]
        self.exps_rsi = [[10, 15, 20], [60, 70, 80], [15, 30]]
        self.exps_bollinger_bands = [[10, 30, 60], [1.9, 2, 2.25, 2.5]]
        self.exps_momentum = [[5, 10, 15, 30]]
        self.exps_macd = [[6, 12, 24, 36], [5, 9, 18]]

    def _experiments(
        self,
        method: Callable[[Any], pd.DataFrame],
        exps: list[list[Any]],
    ) -> dict[str, list[Any]]:
        return experiments(
            self.title,
            self.data,
            method,
            exps,
            ohlc=self.ohlc,
            vis=self.vis,
            report=self.report,
        )

    def moving_average(
        self, exps: list[list[Any]] | None = None
    ) -> dict[str, list[Any]]:
        """Moving average 전략 실험

        Args:
            exps: 전략 function에 입력될 변수들의 범위

        Returns:
            손실 거래 비율에 따른 수익률, `signals`, parameters
        """
        if exps is None:
            exps = self.exps_moving_average
        return self._experiments(moving_average, exps)

    def rsi(self, exps: list[list[Any]] | None = None) -> dict[str, list[Any]]:
        """RSI 전략 실험

        Args:
            exps: 전략 function에 입력될 변수들의 범위

        Returns:
            손실 거래 비율에 따른 수익률, `signals`, parameters
        """
        if exps is None:
            exps = self.exps_rsi
        return self._experiments(rsi, exps)

    def bollinger_bands(
        self, exps: list[list[Any]] | None = None
    ) -> dict[str, list[Any]]:
        """Bollinger bands 전략 실험

        Args:
            exps: 전략 function에 입력될 변수들의 범위

        Returns:
            손실 거래 비율에 따른 수익률, `signals`, parameters
        """
        if exps is None:
            exps = self.exps_bollinger_bands
        return self._experiments(bollinger_bands, exps)

    def momentum(self, exps: list[list[Any]] | None = None) -> dict[str, list[Any]]:
        """Momentum 전략 실험

        Args:
            exps: 전략 function에 입력될 변수들의 범위

        Returns:
            손실 거래 비율에 따른 수익률, `signals`, parameters
        """
        if exps is None:
            exps = self.exps_momentum
        return self._experiments(momentum, exps)

    def macd(self, exps: list[list[Any]] | None = None) -> dict[str, list[Any]]:
        """MACD 전략 실험

        Args:
            exps: 전략 function에 입력될 변수들의 범위

        Returns:
            손실 거래 비율에 따른 수익률, `signals`, parameters
        """
        if exps is None:
            exps = self.exps_macd
        return self._experiments(macd, exps)
