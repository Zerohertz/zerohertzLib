# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import multiprocessing as mp
import os
import time
import traceback
from abc import abstractmethod
from collections import defaultdict
from itertools import combinations
from typing import Any

import FinanceDataReader as fdr
import pandas as pd

from zerohertzLib.api import DiscordBot, SlackBot
from zerohertzLib.api.base import MockedBot
from zerohertzLib.plot import barh, barv, candle, figure, hist, savefig, subplot

from .backtest import Experiments, backtest
from .util import _cash2str, _method2str, _seconds_to_hms


class Quant(Experiments):
    """한 가지 종목에 대해 full factorial design 기반의 backtest를 수행하고 최적의 전략을 융합하는 class

    Args:
        title: 종목 이름
        data: OHLCV (Open, High, Low, Close, Volume) data
        ohlc: 사용할 data의 column 이름
        top: Experiment 과정에서 사용할 각 전략별 수
        methods: 사용할 전략들의 function명 및 parameters
        report: Experiment 결과 출력 여부

    Attributes:
        signals: 융합된 전략의 signal
        methods: 융합된 전략명
        profit: 융합된 전략의 backtest profit
        buy: 융합된 전략의 backtest 시 총 매수
        sell: 융합된 전략의 backtest 시 총 매도
        transaction: 융합된 전략의 backtest 시 거래 정보 (매수가, 매도가, 수익률, 거래 기간)
        threshold_buy: 융합된 전략의 매수 signal threshold
        threshold_sell: 융합된 전략의 매도 signal threshold
        total_cnt: 융합된 전략의 수
        methods_cnt: 각 전략에 따른 이익이 존재하는 수
        exps_cnt: 각 전략과 parameter에 따른 이익이 존재하는 수
        exps_str: 각 전략에 따른 이익이 존재하는 paramter 문자열

    Examples:
        >>> qnt = zz.quant.Quant(title, data, top=3)
        >>> qnt.signals.columns
        Index(['moving_average', 'rsi', 'bollinger_bands', 'momentum', 'macd', 'signals', 'logic'], dtype='object')
        >>> qnt.methods
        ('moving_average', 'bollinger_bands', 'macd')
        >>> qnt.profit
        23.749412256412935
        >>> qnt.buy
        3828200.0
        >>> qnt.sell
        4737375.0
        >>> qnt.transaction
        defaultdict(<class 'list'>, {'buy': [92850.0, ...], 'sell': [105275.0, ...], 'profit': [11.802422227499406, ...], 'period': [205, ...]})
        >>> qnt.threshold_buy
        1
        >>> qnt.threshold_sell
        -4
        >>> qnt.total_cnt
        9
        >>> qnt.methods_cnt
        defaultdict(<class 'int'>, {'moving_average': 3, 'rsi': 3, 'bollinger_bands': 3, 'momentum': 1, 'macd': 3})
        >>> qnt.exps_cnt
        defaultdict(None, {'moving_average': [defaultdict(<class 'int'>, {'20': 3}), ...], ...})
        >>> qnt.exps_str
        defaultdict(<class 'list'>, {'moving_average': ['20-70-1.0', '20-60-1.0', '20-70-0.0'], ...})
        >>> qnt()
        defaultdict(<class 'list'>, {'moving_average': [0, 0.0], 'bollinger_bands': [0, 0.0], 'macd': [0, 0.0], 'logic': 0, 'total': [0, 0.0], 'position': 'None'})
        >>> qnt("20231211")
        defaultdict(<class 'list'>, {'moving_average': [0, 0.0], 'bollinger_bands': [3, 100.0], 'macd': [0, 0.0], 'logic': 1, 'total': [3, 33.33333333333333], 'position': 'Buy'})
        >>> qnt("2023-12-08")
        defaultdict(<class 'list'>, {'moving_average': [0, 0.0], 'bollinger_bands': [3, 100.0], 'macd': [0, 0.0], 'logic': 1, 'total': [3, 33.33333333333333], 'position': 'Buy'})
    """

    def __init__(
        self,
        title: str,
        data: pd.DataFrame,
        ohlc: str = "",
        top: int = 1,
        methods: dict[str, list[list[Any]]] | None = None,
        report: bool = False,
    ) -> None:
        super().__init__(title, data, ohlc, False, report)
        self.signals = pd.DataFrame(index=data.index)
        self.total_cnt = 0
        self.methods_cnt = defaultdict(int)
        self.exps_cnt = defaultdict()
        self.exps_str = defaultdict(list)
        if methods is None:
            methods = {
                "moving_average": self.exps_moving_average,
                "rsi": self.exps_rsi,
                "bollinger_bands": self.exps_bollinger_bands,
                "momentum": self.exps_momentum,
                "macd": self.exps_macd,
            }
        # 선정한 전략들의 parameter 최적화
        is_profit = 0
        for method, exps in methods.items():
            if hasattr(self, method):
                self.signals[method] = 0
                results = getattr(self, method)(exps)
                profits, signals, exps_str, exps_tup = (
                    results["profits"],
                    results["signals"],
                    results["exps_str"],
                    results["exps_tup"],
                )
                exps_cnt = [defaultdict(int) for _ in range(len(exps_tup[0]))]
                for profit, signal, exp_str, exp_tup in zip(
                    profits, signals, exps_str, exps_tup
                ):
                    if profit > 0:
                        self.signals[method] += signal["signals"]
                        self.exps_str[method].append(exp_str)
                        for i, ex in enumerate(exp_tup):
                            exps_cnt[i][str(ex)] += 1
                        self.methods_cnt[method] += 1
                        is_profit += 1
                self.exps_cnt[method] = exps_cnt
            else:
                raise AttributeError(f"'Quant' object has no attribute '{method}'")
        # 전략 간 조합 최적화
        if is_profit >= 1:
            backtests = []
            for cnt in range(1, min(3, len(methods)) + 1):
                for methods_in_use in combinations(methods.keys(), cnt):
                    miu_total = 0
                    for miu in methods_in_use:
                        miu_total += self.methods_cnt[miu]
                        if self.methods_cnt[miu] < 1:
                            miu_total = 0
                            break
                    if miu_total == 0:
                        continue
                    self.signals["signals"] = self.signals.loc[:, methods_in_use].sum(1)
                    for threshold_sell in range(1, max(2, miu_total)):
                        for threshold_buy in range(1, max(2, miu_total)):
                            results = backtest(
                                self.data,
                                self.signals,
                                ohlc=ohlc,
                                threshold=(-threshold_sell, threshold_buy),
                            )
                            backtests.append(
                                {
                                    "profit": results["profit"],
                                    "weighted_profit": results["weighted_profit"],
                                    "threshold": (-threshold_sell, threshold_buy),
                                    "methods": methods_in_use,
                                    "total": miu_total,
                                    "transaction": results["transaction"],
                                    "buy": results["buy"],
                                    "sell": results["sell"],
                                    "logic": self.signals["logic"].copy(),
                                }
                            )
            backtests.sort(key=lambda x: x["weighted_profit"], reverse=True)
            # 최적 융합 전략
            self.signals["signals"] = self.signals.loc[:, backtests[0]["methods"]].sum(
                1
            )
            self.signals["logic"] = backtests[0]["logic"]
            self.methods = backtests[0]["methods"]
            self.profit = backtests[0]["profit"]
            self.total_cnt = backtests[0]["total"]
            self.buy, self.sell = backtests[0]["buy"], backtests[0]["sell"]
            self.transaction = backtests[0]["transaction"]
            self.threshold_sell, self.threshold_buy = backtests[0]["threshold"]

    def __call__(self, day: str | int = -1) -> dict[str, Any]:
        """
        입력된 날짜에 대해 분석 정보 return

        Args:
            day: 분석할 날짜

        Returns:
            각 전략에 따른 분석 정보 및 결론
        """
        if self.total_cnt < 1:
            return {"position": "NULL"}
        if day != -1 and "-" not in day:
            day = day[:4] + "-" + day[4:6] + "-" + day[6:8]
        possibility = defaultdict(list)
        for key in self.methods:
            possibility[key] = [
                self.signals[key].iloc[day],
                self.signals[key].iloc[day] / self.methods_cnt[key] * 100,
            ]
        possibility["logic"] = self.signals["logic"].iloc[day]
        possibility["total"] = [
            self.signals["signals"].iloc[day],
            self.signals["signals"].iloc[day] / self.total_cnt * 100,
        ]
        if 0 < possibility["logic"]:
            possibility["position"] = "Buy"
        elif 0 > possibility["logic"]:
            possibility["position"] = "Sell"
        elif self.threshold_buy <= possibility["total"][0]:
            possibility["position"] = "Buy"
            possibility["logic"] = 1
        elif self.threshold_sell >= possibility["total"][0]:
            possibility["position"] = "Sell"
            possibility["logic"] = -1
        else:
            possibility["position"] = "None"
        return possibility


class QuantBot:
    """입력된 여러 종목에 대해 매수, 매도 signal을 판단하고 Slack으로 message와 graph를 전송하는 class

    Note:
        Abstract Base Class: 종목 code에 따른 종목명과 data를 불러오는 abstract method `_get_data` 정의 후 사용

        ```python
        def _get_data(self, symbol: str) -> tuple[str, pd.DataFrame]:
            title = data = None
            return title, data
        ```

    Args:
        symbols: 종목 code들
        start_day: 조회 시작 일자 (`YYYYMMDD`)
        ohlc: 사용할 data의 column 이름
        top: Experiment 과정에서 사용할 각 전략별 수
        methods: 사용할 전략들의 function명 및 parameters
        report: Experiment 결과 출력 여부
        token: Bot의 token (`xoxb-` prefix로 시작하면 `SlackBot`, 아니면 `DiscordBot`)
        channel: Bot이 전송할 channel
        name: Bot의 표시될 이름
        icon_emoji: Bot의 표시될 사진 (emoji)
        mp_num: 병렬 처리에 사용될 process의 수 (0: 직렬 처리)
        analysis: 각 전략의 보고서 전송 여부
        kor: 국내 여부

    Attributes:
        exps: 각 전략에 따른 parameter 분포

    Examples:
        >>> qsb = zz.quant.QuantBot(symbols, token=token, channel=channel)
        >>> qsb.index()

    ![QuantSlackBot example](../../../assets/quant/QuantSlackBot.png){ width="800" }
    """

    def __init__(
        self,
        symbols: int | list[str],
        start_day: str = "",
        ohlc: str = "",
        top: int = 1,
        methods: dict[str, list[list[Any]]] | None = None,
        report: bool = False,
        token: str | None = None,
        channel: str | None = None,
        name: str | None = None,
        icon_emoji: str | None = None,
        mp_num: int = 0,
        analysis: bool = False,
        kor: bool = True,
    ) -> None:
        if token is None or channel is None:
            self.bot = MockedBot()
        elif token.startswith("xoxb-"):
            self.bot = SlackBot(
                token=token, channel=channel, name=name, icon_emoji=icon_emoji
            )
        else:
            self.bot = DiscordBot(token=token, channel=channel)
        self.symbols = symbols
        self.start_day = start_day
        self.ohlc = ohlc
        self.top = top
        self.methods = methods
        if mp_num > mp.cpu_count():
            self.mp_num = mp.cpu_count()
        else:
            self.mp_num = mp_num
        self.analysis = analysis
        self.kor = kor
        self.report = report

    def _plot(self, quant: Quant) -> tuple[str, str]:
        candle_path = candle(
            quant.data[-500:],
            quant.title,
            signals=quant.signals.iloc[-500:, :].loc[
                :, [*quant.methods, "signals", "logic"]
            ],
            dpi=100,
            threshold=(quant.threshold_sell, quant.threshold_buy),
        )
        figure((10, 18))
        subplot(3, 1, 1)
        if self.kor:
            dim = "₩"
        else:
            dim = "$"
        hist(
            {"Buy": quant.transaction["buy"], "Sell": quant.transaction["sell"]},
            xlab=f"매수/매도가 [{dim}]",
            title="",
        )
        subplot(3, 1, 2)
        hist(
            {"Profit": quant.transaction["profit"]},
            xlab="이율 [%]",
            title="",
        )
        subplot(3, 1, 3)
        hist(
            {"Period": quant.transaction["period"]},
            xlab="거래 기간 [일]",
            title="",
        )
        hist_path = savefig(f"{quant.title}_backtest", 100)
        return candle_path, hist_path

    def _report(
        self, symbol: str, quant: Quant, today: dict[str, Any]
    ) -> dict[str, str]:
        logic = {-2: "손절", -1: "매도", 0: "중립", 1: "매수", 2: "추가 매수"}
        report = defaultdict(str)
        if today["position"] == "Buy":
            report["main"] += "> :chart_with_upwards_trend: [Buy Signal]"
        elif today["position"] == "Sell":
            report["main"] += "> :chart_with_downwards_trend: [Sell Signal]"
        else:
            report["main"] += "> :egg: [None Signal]"
        report["main"] += f" *{quant.title}* (`{symbol}`)\n"
        report["main"] += (
            f"\t:technologist: Signal Info: {today['total'][1]:.2f}% ({int(today['total'][0])}/{int(quant.total_cnt)}) → {logic[today['logic']]}\n"
        )
        report["param"] += "> :information_desk_person: *Parameter Info*"
        for key in quant.methods:
            report["main"] += (
                f"\t\t:hammer: {_method2str(key)}: {today[key][1]:.2f}% ({int(today[key][0])}/{int(quant.methods_cnt[key])})\n"
            )
            report["param"] += (
                f"\n\t:hammer: {_method2str(key)}: `{'`, `'.join(quant.exps_str[key])}`"
            )
        report["main"] += "\t:memo: Threshold:\n"
        report["main"] += (
            f"\t\t:arrow_double_up: Buy: {quant.threshold_buy}\n\t\t:arrow_double_down: Sell: {quant.threshold_sell}"
        )
        report["backtest"] += (
            f"> :computer: *Backtest* ({self.start_day[:4]}/{self.start_day[4:6]}/{self.start_day[6:]} ~)\n\t:money_with_wings: Total Profit:\t{quant.profit:.2f}%\n"
        )
        report["backtest"] += (
            f"\t:chart_with_upwards_trend: Total Buy:\t{_cash2str(quant.buy, self.kor)}\n"
        )
        report["backtest"] += (
            f"\t:chart_with_downwards_trend: Total Sell:\t{_cash2str(quant.sell, self.kor)}\n"
        )
        report["candle"], report["hist"] = self._plot(quant)
        return report

    @abstractmethod
    def _get_data(self, symbol: str) -> tuple[str, pd.DataFrame]:
        title = data = None
        return title, data

    def _run(self, args: list[str]) -> tuple[dict[str, str] | None, Quant | None]:
        symbol, mode = args
        try:
            title, data = self._get_data(symbol)
            if len(data) < 20:
                return None, None
        except KeyError as error:
            response = self.bot.message(f":x: `{symbol}` was not found")
            thread_id = self.bot.get_thread_id(
                response, name=f"`{symbol}` was not found"
            )
            self.bot.message(str(error), codeblock=True, thread_id=thread_id)
            self.bot.message(
                traceback.format_exc(), codeblock=True, thread_id=thread_id
            )
            return None, None
        try:
            quant = Quant(
                title,
                data,
                ohlc=self.ohlc,
                top=self.top,
                methods=self.methods,
                report=self.report,
            )
            today = quant()
        except IndexError as error:
            response = self.bot.message(
                f":x: `{symbol}` ({title}): {data.index[0]} ({len(data)})"
            )
            thread_id = self.bot.get_thread_id(
                response, name=f"`{symbol}` ({title}): {data.index[0]} ({len(data)})"
            )
            self.bot.message(str(error), codeblock=True, thread_id=thread_id)
            self.bot.message(
                traceback.format_exc(), codeblock=True, thread_id=thread_id
            )
            return None, None
        if today["position"] == "NULL":
            return None, None
        if mode == "Buy":
            positions = ["Buy"]
        else:
            positions = ["Buy", "Sell", "None"]
        if today["position"] in positions:
            return self._report(symbol, quant, today), quant
        return None, quant

    def _send(self, report: dict[str, str]) -> None:
        response = self.bot.message(report["main"])
        thread_id = self.bot.get_thread_id(response, name=report["main"])
        self.bot.file(report["candle"], thread_id=thread_id)
        response = self.bot.message(report["backtest"], thread_id=thread_id)
        self.bot.file(report["hist"], thread_id=thread_id)
        response = self.bot.message(report["param"], thread_id=thread_id)

    def _analysis_update(
        self,
        quant: Quant,
    ) -> None:
        methods, total_cnt, methods_cnt, exps_cnt = (
            quant.methods,
            quant.total_cnt,
            quant.methods_cnt,
            quant.exps_cnt,
        )
        self.quant_cnt += 1
        for method in methods:
            self.miu_cnt[_method2str(method)] += 1
        self.total_cnt.append(total_cnt)
        for method, cnt in methods_cnt.items():
            if cnt > 0:
                self.methods_cnt[_method2str(method)].append(cnt)
        for method, cnt in exps_cnt.items():
            if method not in self.exps_cnt.keys():
                self.exps_cnt[method] = [defaultdict(int) for _ in range(len(cnt))]
            for idx, _cnt in enumerate(cnt):
                for param, __cnt in _cnt.items():
                    self.exps_cnt[method][idx][param] += __cnt

    def _analysis_send(self) -> None:
        response = self.bot.message("> :memo: Parameter Analysis")
        thread_id = self.bot.get_thread_id(response, name="Parameter Analysis")
        figure((30, 20))
        subplot(2, 2, 1)
        barv(
            dict(sorted(self.miu_cnt.items())),
            title=f"Methods in Use (Avg: {sum(self.miu_cnt.values()) / self.quant_cnt:.2f})",
            dim="%",
        )
        subplot(2, 2, 2)
        hist(
            {"": self.total_cnt},
            title=f"Distribution of Methods in Use (Avg: {sum(self.total_cnt) / self.quant_cnt:.2f})",
            cnt=max(self.total_cnt) * 2,
            ovp=True,
        )
        subplot(2, 2, 3)
        barv(
            dict(
                ((key, sum(value)) for key, value in sorted(self.methods_cnt.items()))
            ),
            title="Available Methods",
            dim="%",
        )
        subplot(2, 2, 4)
        hist(
            dict(sorted(self.methods_cnt.items())),
            title="Distribution of Available Methods",
            cnt=self.top * 2,
            ovp=False,
        )
        path = savefig("Methods", 100)
        self.bot.file(path, thread_id=thread_id)
        for method, cnt in self.exps_cnt.items():
            figure((18, 8))
            stg = True
            for idx, count in enumerate(cnt):
                try:
                    subplot(1, len(cnt), idx + 1)
                    barh(count, title="", dim="%")
                except IndexError:
                    stg = False
                    print(f"'{method}' was not available: {count}")
                    break
            if stg:
                path = savefig(method, dpi=100)
                self.bot.file(path, thread_id=thread_id)
            else:
                self.bot.message(
                    f":no_bell: '{method}' was not available", thread_id=thread_id
                )

    def _inference(self, symbols: list[str], mode: str) -> None:
        start = time.time()
        if self.analysis:
            # 유효한 Quant instance의 수
            self.quant_cnt = 0
            # [Methods in Use: O] 사용된 전략 종류의 수
            self.miu_cnt = defaultdict(int)
            # [Methods in Use: O] 사용된 전략의 수 (같은 전략 포함)
            self.total_cnt = []
            # [Methods in Use: X] 전략에 따른 이익이 존재하는 수
            self.methods_cnt = defaultdict(list)
            # [Methods in Use: X] 전략과 parameter에 따른 이익이 존재하는 수
            self.exps_cnt = defaultdict(list)
        response = self.bot.message(f"> :moneybag: Check {mode} Signals")
        thread_id = self.bot.get_thread_id(response, name=f"Check {mode} Signals")
        self.bot.message(", ".join(symbols), codeblock=True, thread_id=thread_id)
        if self.mp_num == 0 or self.mp_num >= len(symbols):
            for symbol in symbols:
                report, quant = self._run([symbol, mode])
                if report is not None:
                    self._send(report)
                if self.analysis and quant is not None:
                    self._analysis_update(quant)
        else:
            args = [[symbol, mode] for symbol in symbols]
            with mp.Pool(processes=self.mp_num) as pool:
                results = pool.map(self._run, args)
            for report, quant in results:
                if report is not None:
                    self._send(report)
                if self.analysis and quant is not None:
                    self._analysis_update(quant)
        if self.analysis:
            self._analysis_send()
        end = time.time()
        self.bot.message(f"> :tada: Done! (`{_seconds_to_hms(end - start)}`)")

    def buy(self) -> None:
        """매수 signals 탐색"""
        self._inference(self.symbols, "Buy")

    def index(self) -> None:
        """모든 signals 탐색"""
        self._inference(self.symbols, "All")


class QuantBotFDR(QuantBot):
    """[FinanceDataReader](https://github.com/FinanceData/FinanceDataReader) module 기반으로 입력된 여러 종목에 대해 매수, 매도 signal을 판단하고 Bot을 통해 message와 graph를 전송하는 class

    Args:
        symbols: 종목 code들 혹은 시가 총액 순위
        start_day: 조회 시작 일자 (`YYYYMMDD`)
        ohlc: 사용할 data의 column 이름
        top: Experiment 과정에서 사용할 각 전략별 수
        methods: 사용할 전략들의 function명 및 parameters
        report: Experiment 결과 출력 여부
        token: Bot의 token (`xoxb-` prefix로 시작하면 `SlackBot`, 아니면 `DiscordBot`)
        channel: Bot이 전송할 channel
        name: Bot의 표시될 이름
        icon_emoji: Bot의 표시될 사진 (emoji)
        mp_num: 병렬 처리에 사용될 process의 수 (0: 직렬 처리)
        analysis: 각 전략의 보고서 전송 여부
        kor: 국내 여부

    Attributes:
        exps: 각 전략에 따른 parameter 분포
        market: kor에 따른 시장 목록

    Examples:
        >>> qbf = zz.quant.QuantBotFDR(symbols, token=token, channel=channel)
        >>> qbf = zz.quant.QuantBotFDR(10, token=token, channel=channel)
    """

    def __init__(
        self,
        symbols: int | list[str],
        start_day: str = "",
        ohlc: str = "",
        top: int = 1,
        methods: dict[str, list[list[Any]]] | None = None,
        report: bool = False,
        token: str | None = None,
        channel: str | None = None,
        name: str | None = None,
        icon_emoji: str | None = None,
        mp_num: int = 0,
        analysis: bool = False,
        kor: bool = True,
    ) -> None:
        QuantBot.__init__(
            self,
            symbols=symbols,
            start_day=start_day,
            ohlc=ohlc,
            top=top,
            methods=methods,
            report=report,
            token=token,
            channel=channel,
            name=name,
            icon_emoji=icon_emoji,
            mp_num=mp_num,
            analysis=analysis,
            kor=kor,
        )
        if kor:
            # FIXME:
            # FDR 의존성 내에서 KRX-DESC 코드 사용 시 오류 발생
            # https://github.com/FinanceData/FinanceDataReader/pull/254
            market = os.getenv("QUANT_MARKET_KOR", "ETF/KR")
        else:
            market = os.getenv("QUANT_MARKET_OVS", "NASDAQ")
        self.market = fdr.StockListing(market)
        if isinstance(symbols, int):
            self.symbols = list(self.market[self.market.columns[0]])[:symbols]

    def _get_data(self, symbol: str) -> tuple[str, pd.DataFrame]:
        try:
            title = self.market[
                self.market[self.market.columns[0]] == symbol
            ].Name.iloc[0]
        except IndexError:
            title = symbol
        data = fdr.DataReader(symbol, self.start_day)
        return title, data
