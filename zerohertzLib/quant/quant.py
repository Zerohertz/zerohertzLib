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

import multiprocessing as mp
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import FinanceDataReader as fdr
import pandas as pd
from slack_sdk.web import SlackResponse

from zerohertzLib.api import SlackBot
from zerohertzLib.plot import barh, barv, candle, figure, hist, savefig, subplot

from .backtest import Experiments, backtest
from .util import _cash2str, _method2str, _seconds_to_hms


class Quant(Experiments):
    """한 가지 종목에 대해 full factorial design 기반의 backtest를 수행하고 최적의 전략을 융합하는 class

    Args:
        title (``str``): 종목 이름
        data (``pd.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        top (``Optional[int]``): Experiment 과정에서 사용할 각 전략별 수
        methods (``Optional[Dict[str, List[List[Any]]]]``): 사용할 전략들의 함수명 및 parameters
        report (``Optional[bool]``): Experiment 결과 출력 여부

    Attributes:
        signals (``pd.DataFrame``): 융합된 전략의 signal
        methods (``Tuple[str]``): 융합된 전략명
        profit (``float``): 융합된 전략의 backtest profit
        buy (``Union[int, float]``): 융합된 전략의 backtest 시 총 매수
        sell (``Union[int, float]``): 융합된 전략의 backtest 시 총 매도
        transaction (``Dict[str, Union[int, float]]``): 융합된 전략의 backtest 시 거래 정보 (매수가, 매도가, 수익률, 거래 기간)
        threshold_buy (``int``): 융합된 전략의 매수 signal threshold
        threshold_sell (``int``): 융합된 전략의 매도 signal threshold
        total_cnt (``int``): 융합된 전략의 수
        methods_cnt (``Dict[str, int]``): 각 전략에 따른 이익이 존재하는 수
        exps_cnt (``Dict[str, List[Dict[str, int]]]``): 각 전략과 parameter에 따른 이익이 존재하는 수
        exps_str (``Dict[str, List[str]]``): 각 전략에 따른 이익이 존재하는 paramter 문자열

    Methods:
        __call__:
            입력된 날짜에 대해 분석 정보 return

            Args:
                day (``Optional[str]``): 분석할 날짜

            Returns:
                ``Dict[str, Any]``: 각 전략에 따른 분석 정보 및 결론

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
        ohlc: Optional[str] = "",
        top: Optional[int] = 1,
        methods: Optional[Dict[str, List[List[Any]]]] = None,
        report: Optional[bool] = False,
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
                    profits[:top], signals[:top], exps_str[:top], exps_tup[:top]
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

    def __call__(self, day: Optional[str] = -1) -> Dict[str, Any]:
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


class QuantSlackBot(ABC, SlackBot):
    """입력된 여러 종목에 대해 매수, 매도 signal을 판단하고 Slack으로 message와 graph를 전송하는 class

    Note:
        Abstract Base Class: 종목 code에 따른 종목명과 data를 불러오는 abstract method ``_get_data`` 정의 후 사용

        .. code-block:: python

            def _get_data(self, symbol: str) -> Tuple[str, pd.DataFrame]:
                title = data = None
                return title, data

    Args:
        symbols (``List[str]``): 종목 code들
        start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        top (``Optional[int]``): Experiment 과정에서 사용할 각 전략별 수
        methods (``Optional[Dict[str, List[List[Any]]]]``): 사용할 전략들의 함수명 및 parameters
        report (``Optional[bool]``): Experiment 결과 출력 여부
        token (``Optional[str]``): Slack Bot의 token
        channel (``Optional[str]``): Slack Bot이 전송할 channel
        name (``Optional[str]``): Slack Bot의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Bot의 표시될 사진 (emoji)
        mp_num (``Optional[int]``): 병렬 처리에 사용될 process의 수 (``0``: 직렬 처리)
        analysis (``Optional[bool]``): 각 전략의 보고서 전송 여부
        kor (``Optional[bool]``): 국내 여부

    Attributes:
        exps (``Dict[str, List[Dict[str, int]]]``): 각 전략에 따른 parameter 분포

    Examples:
        >>> qsb = zz.quant.QuantSlackBot(symbols, token=token, channel=channel)
        >>> qsb.index()

        .. image:: _static/examples/static/quant.QuantSlackBot.png
            :align: center
            :width: 800px
    """

    def __init__(
        self,
        symbols: List[str],
        start_day: Optional[str] = "",
        ohlc: Optional[str] = "",
        top: Optional[int] = 1,
        methods: Optional[Dict[str, List[List[Any]]]] = None,
        report: Optional[bool] = False,
        token: Optional[str] = None,
        channel: Optional[str] = None,
        name: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        mp_num: Optional[int] = 0,
        analysis: Optional[bool] = False,
        kor: Optional[bool] = True,
    ) -> None:
        if token is None or channel is None:
            self.slack = False
            token = ""
        else:
            self.slack = True
        SlackBot.__init__(self, token, channel, name, icon_emoji)
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

    def message(
        self,
        message: str,
        codeblock: Optional[bool] = False,
        thread_ts: Optional[str] = None,
    ) -> SlackResponse:
        """``token`` 혹은 ``channel`` 이 입력되지 않을 시 전송 불가

        Args:
            message (``str``): 전송할 message
            codeblock (``Optional[bool]``): 전송되는 message의 스타일
            thread_ts (``Optional[str]``): 댓글을 전송할 thread의 timestamp

        Returns:
            ``slack_sdk.web.slack_response.SlackResponse``: Slack Bot의 응답
        """
        if self.slack:
            return super().message(message, codeblock, thread_ts)
        return None

    def file(self, path: str, thread_ts: Optional[str] = None) -> SlackResponse:
        """``token`` 혹은 ``channel`` 이 입력되지 않을 시 전송 불가

        Args:
            path (``str``): 전송할 file 경로
            thread_ts (``Optional[str]``): 댓글을 전송할 thread의 timestamp

        Returns:
            ``slack_sdk.web.slack_response.SlackResponse``: Slack Bot의 응답
        """
        if self.slack:
            return super().file(path, thread_ts)
        return None

    def _plot(self, quant: Quant) -> Tuple[str]:
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
        self, symbol: str, quant: Quant, today: Dict[str, Any]
    ) -> Dict[str, str]:
        logic = {-2: "손절", -1: "매도", 0: "중립", 1: "매수", 2: "추가 매수"}
        report = defaultdict(str)
        if today["position"] == "Buy":
            report["main"] += "> :chart_with_upwards_trend: [Buy Signal]"
        elif today["position"] == "Sell":
            report["main"] += "> :chart_with_downwards_trend: [Sell Signal]"
        else:
            report["main"] += "> :egg: [None Signal]"
        report["main"] += f" *{quant.title}* (`{symbol}`)\n"
        report[
            "main"
        ] += f"\t:technologist: Signal Info: {today['total'][1]:.2f}% ({int(today['total'][0])}/{int(quant.total_cnt)}) → {logic[today['logic']]}\n"
        report["param"] += "> :information_desk_person: *Parameter Info*"
        for key in quant.methods:
            report[
                "main"
            ] += f"\t\t:hammer: {_method2str(key)}: {today[key][1]:.2f}% ({int(today[key][0])}/{int(quant.methods_cnt[key])})\n"
            report[
                "param"
            ] += (
                f"\n\t:hammer: {_method2str(key)}: `{'`, `'.join(quant.exps_str[key])}`"
            )
        report["main"] += "\t:memo: Threshold:\n"
        report[
            "main"
        ] += f"\t\t:arrow_double_up: Buy: {quant.threshold_buy}\n\t\t:arrow_double_down: Sell: {quant.threshold_sell}"
        report[
            "backtest"
        ] += f"> :computer: *Backtest* ({self.start_day[:4]}/{self.start_day[4:6]}/{self.start_day[6:]} ~)\n\t:money_with_wings: Total Profit:\t{quant.profit:.2f}%\n"
        report[
            "backtest"
        ] += f"\t:chart_with_upwards_trend: Total Buy:\t{_cash2str(quant.buy, self.kor)}\n"
        report[
            "backtest"
        ] += f"\t:chart_with_downwards_trend: Total Sell:\t{_cash2str(quant.sell, self.kor)}\n"
        report["candle"], report["hist"] = self._plot(quant)
        return report

    @abstractmethod
    def _get_data(self, symbol: str) -> Tuple[str, pd.DataFrame]:
        title = data = None
        return title, data

    def _run(self, args: List[str]) -> Tuple[Dict[str, str], Quant]:
        symbol, mode = args
        try:
            title, data = self._get_data(symbol)
            if len(data) < 20:
                return None, None
        except KeyError as error:
            response = self.message(f":x: `{symbol}` was not found")
            thread_ts = response.get("ts")
            self.message(str(error), True, thread_ts)
            self.message(traceback.format_exc(), True, thread_ts)
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
            response = self.message(
                f":x: `{symbol}` ({title}): {data.index[0]} ({len(data)})"
            )
            thread_ts = response.get("ts")
            self.message(str(error), True, thread_ts)
            self.message(traceback.format_exc(), True, thread_ts)
            return None, None
        if today["position"] == "NULL":
            return None, None
        if not self.slack:
            return None, quant
        if mode == "Buy":
            positions = ["Buy"]
        else:
            positions = ["Buy", "Sell", "None"]
        if today["position"] in positions:
            return self._report(symbol, quant, today), quant
        return None, quant

    def _send(self, report: Dict[str, str]) -> None:
        if report is None:
            return
        response = self.message(report["main"])
        thread_ts = response.get("ts")
        self.file(report["candle"], thread_ts)
        response = self.message(report["backtest"], thread_ts=thread_ts)
        self.file(report["hist"], thread_ts)
        response = self.message(report["param"], thread_ts=thread_ts)

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
            for idx, cnt_ in enumerate(cnt):
                for param, cnt__ in cnt_.items():
                    self.exps_cnt[method][idx][param] += cnt__

    def _analysis_send(self) -> None:
        response = self.message("> :memo: Parameter Analysis")
        thread_ts = response.get("ts")
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
        self.file(path, thread_ts)
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
                self.file(path, thread_ts)
            else:
                self.message(
                    f":no_bell: '{method}' was not available", thread_ts=thread_ts
                )

    def _inference(self, symbols: List[str], mode: str) -> None:
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
        response = self.message(f"> :moneybag: Check {mode} Signals")
        self.message(", ".join(symbols), True, response.get("ts"))
        if self.mp_num == 0 or self.mp_num >= len(symbols):
            for symbol in symbols:
                report, quant = self._run([symbol, mode])
                self._send(report)
                if self.analysis and quant is not None:
                    self._analysis_update(quant)
        else:
            args = [[symbol, mode] for symbol in symbols]
            with mp.Pool(processes=self.mp_num) as pool:
                results = pool.map(self._run, args)
            for report, quant in results:
                self._send(report)
                if self.analysis and quant is not None:
                    self._analysis_update(quant)
        if self.analysis:
            self._analysis_send()
        end = time.time()
        self.message(f"> :tada: Done! (`{_seconds_to_hms(end - start)}`)")

    def buy(self) -> None:
        """매수 signals 탐색"""
        self._inference(self.symbols, "Buy")

    def index(self) -> None:
        """모든 signals 탐색"""
        self._inference(self.symbols, "All")


class QuantSlackBotFDR(QuantSlackBot):
    """`FinanceDataReader <https://github.com/FinanceData/FinanceDataReader>`_ module 기반으로 입력된 여러 종목에 대해 매수, 매도 signal을 판단하고 Slack으로 message와 graph를 전송하는 class

    Args:
        symbols (``Union[int, List[str]]``): 종목 code들 혹은 시가 총액 순위
        start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        top (``Optional[int]``): Experiment 과정에서 사용할 각 전략별 수
        methods (``Optional[Dict[str, List[List[Any]]]]``): 사용할 전략들의 함수명 및 parameters
        report (``Optional[bool]``): Experiment 결과 출력 여부
        token (``Optional[str]``): Slack Bot의 token
        channel (``Optional[str]``): Slack Bot이 전송할 channel
        name (``Optional[str]``): Slack Bot의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Bot의 표시될 사진 (emoji)
        mp_num (``Optional[int]``): 병렬 처리에 사용될 process의 수 (``0``: 직렬 처리)
        analysis (``Optional[bool]``): 각 전략의 보고서 전송 여부
        kor (``Optional[bool]``): 국내 여부

    Attributes:
        exps (``Dict[str, List[Dict[str, int]]]``): 각 전략에 따른 parameter 분포
        market (``pd.DataFrame``): ``kor`` 에 따른 시장 목록

    Examples:
        >>> qsb = zz.quant.QuantSlackBotFDR(symbols, token=token, channel=channel)
        >>> qsb = zz.quant.QuantSlackBotFDR(10, token=token, channel=channel)
    """

    def __init__(
        self,
        symbols: Union[int, List[str]],
        start_day: Optional[str] = "",
        ohlc: Optional[str] = "",
        top: Optional[int] = 1,
        methods: Optional[Dict[str, List[List[Any]]]] = None,
        report: Optional[bool] = False,
        token: Optional[str] = None,
        channel: Optional[str] = None,
        name: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        mp_num: Optional[int] = 0,
        analysis: Optional[bool] = False,
        kor: Optional[bool] = True,
    ) -> None:
        QuantSlackBot.__init__(
            self,
            symbols,
            start_day,
            ohlc,
            top,
            methods,
            report,
            token,
            channel,
            name,
            icon_emoji,
            mp_num,
            analysis,
            kor,
        )
        if kor:
            self.market = fdr.StockListing("KRX-DESC")
        else:
            self.market = fdr.StockListing("NASDAQ")
        if isinstance(symbols, int):
            if kor:
                krx = fdr.StockListing("KRX")
                krx = krx.sort_values("Marcap", ascending=False)
                self.symbols = list(krx["Code"])[:symbols]
            else:
                self.symbols = list(self.market["Symbol"])[:symbols]

    def _get_data(self, symbol: str) -> Tuple[str, pd.DataFrame]:
        try:
            if self.kor:
                title = self.market[self.market["Code"] == symbol].iloc[0, 1]
            else:
                title = self.market[self.market["Symbol"] == symbol].iloc[0, 1]
        except IndexError:
            title = symbol
        data = fdr.DataReader(symbol, self.start_day)
        return title, data
