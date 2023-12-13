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
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, ItemsView, List, Optional, Tuple, Union

import FinanceDataReader as fdr
import pandas as pd
import requests
from matplotlib import pyplot as plt

from zerohertzLib.api import KoreaInvestment, SlackBot
from zerohertzLib.plot import barh, barv, candle, figure, hist, savefig, table

from .backtest import Experiments, backtest
from .util import _cash2str, _seconds_to_hms


class Quant(Experiments):
    """한 가지 종목에 대해 full factorial design 기반의 backtest를 수행하고 최적의 전략을 융합하는 class

    Args:
        title (``str``): 종목 이름
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        top (``Optional[int]``): Experiment 과정에서 사용할 각 전략별 수
        methods (``Optional[List[str]]``): 사용할 전략들의 함수명
        report: (``Optional[bool]``): Experiment 결과 출력 여부

    Attributes:
        signals (``pd.core.frame.DataFrame``): 융합된 전략의 signal
        methods (``Tuple[str]``): 융합된 전략명
        profit (``float``): 융합된 전략의 backtest profit
        buy (``Union[int, float]``): 융합된 전략의 backtest 시 총 매수
        sell (``Union[int, float]``): 융합된 전략의 backtest 시 총 매도
        transaction (``Dict[str, Union[int, float]]``): 융합된 전략의 backtest 시 거래 정보 (매수가, 매도가, 수익률, 거래 기간)
        threshold_buy (``int``): 융합된 전략의 매수 signal threshold
        threshold_sell (``int``): 융합된 전략의 매도 signal threshold
        total_cnt (``int``): 융합된 전략의 수
        method_cnt (``Dict[str, int]``): 각 전략에 따른 이익이 존재하는 수
        exps_cnt (``Dict[str, List[Dict[str, int]]]``): 각 전략과 parameter에 따른 이익이 존재하는 수
        exps_str (``Dict[str, List[str]]``): 각 전략에 따른 이익이 존재하는 paramter 문자열

    Methods:
        __getitem__:
            입력된 날짜에 대해 분석 정보 return

            Args:
                day (``Optional[str]``): 분석할 날짜

            Returns:
                ``Dict[str, Any]``: 각 전략에 따른 분석 정보 및 결론

    Examples:
        >>> qnt = zz.quant.Quant(title, data, top=3)
        >>> qnt.signals.columns
        Index(['moving_average', 'rsi', 'bollinger_bands', 'momentum', 'signals', 'logic'], dtype='object')
        >>> qnt.methods
        ('moving_average',)
        >>> qnt.profit
        22.04593267882188
        >>> qnt.buy
        1140800.0
        >>> qnt.sell
        1392300.0
        >>> qnt.transaction
        defaultdict(<class 'list'>, {'buy': [89900.0, ...], 'sell': [107100.0, ...], 'profit': [16.059757236227824, ...], 'period': [79, ...]})
        >>> qnt.threshold_buy
        1
        >>> qnt.threshold_sell
        -1
        >>> qnt.total_cnt
        3
        >>> qnt.method_cnt
        defaultdict(<class 'int'>, {'moving_average': 3, 'rsi': 3, 'bollinger_bands': 3, 'momentum': 3})
        >>> qnt.exps_cnt
        defaultdict(None, {'moving_average': [defaultdict(<class 'int'>, {'15': 3}), ...], ...})
        >>> qnt.exps_str
        defaultdict(<class 'list'>, {'moving_average': ['15-75-100', '15-80-100', '15-80-150'], ...})
        >>> qnt()
        defaultdict(<class 'list'>, {'moving_average': [0.0, 0.0], 'logic': 0, 'total': [0.0, 0.0], 'position': 'None'})
        >>> qnt("20221208")
        defaultdict(<class 'list'>, {'moving_average': [0.0, 0.0], 'logic': 2, 'total': [0.0, 0.0], 'position': 'Buy'})
        >>> qnt("2023-11-22")
        defaultdict(<class 'list'>, {'moving_average': [2.0, 66.66666666666666], 'logic': 0, 'total': [2.0, 66.66666666666666], 'position': 'Buy'})
    """

    def __init__(
        self,
        title: str,
        data: pd.core.frame.DataFrame,
        ohlc: Optional[str] = "",
        top: Optional[int] = 1,
        methods: Optional[List[str]] = None,
        report: Optional[bool] = False,
    ) -> None:
        super().__init__(title, data, ohlc, False, report)
        self.signals = pd.DataFrame(index=data.index)
        self.total_cnt = 0
        self.method_cnt = defaultdict(int)
        self.exps_cnt = defaultdict()
        self.exps_str = defaultdict(list)
        if methods is None:
            methods = [
                "moving_average",
                "rsi",
                "bollinger_bands",
                "momentum",
            ]
        # 선정한 전략들의 parameter 최적화
        is_profit = 0
        for method in methods:
            if hasattr(self, method):
                self.signals[method] = 0
                results = getattr(self, method)()
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
                        self.method_cnt[method] += 1
                        is_profit += 1
                self.exps_cnt[method] = exps_cnt
            else:
                raise AttributeError(f"'Quant' object has no attribute '{method}'")
        # 전략 간 조합 최적화
        if is_profit >= 1:
            backtests = []
            for cnt in range(1, len(methods)):
                for methods_in_use in combinations(methods, cnt):
                    miu_total = 0
                    for miu in methods_in_use:
                        miu_total += self.method_cnt[miu]
                        if self.method_cnt[miu] < 1:
                            miu_total = 0
                            break
                    if miu_total == 0:
                        continue
                    self.signals["signals"] = self.signals.loc[:, methods_in_use].sum(1)
                    for threshold_sell in range(1, miu_total + 1):
                        for threshold_buy in range(1, miu_total + 1):
                            results = backtest(
                                self.data,
                                self.signals,
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
                                    "logic": self.signals["logic"],
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
                self.signals[key][day],
                self.signals[key][day] / self.method_cnt[key] * 100,
            ]
        possibility["logic"] = self.signals["logic"][day]
        possibility["total"] = [
            self.signals["signals"][day],
            self.signals["signals"][day] / self.total_cnt * 100,
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


class Balance(KoreaInvestment):
    """한국투자증권의 국내 계좌 정보 조회 class

    Args:
        path (``Optional[str]``): ``secret.key`` 혹은 ``token.dat`` 이 포함된 경로

    Attributes:
        balance (``Dict[str, Any]``): 현재 보유 주식과 계좌의 금액 정보
        kor (``Optional[bool]``): 국내 여부

    Methods:
        __contains__:
            Args:
                item (``Any``): 보유 여부를 판단할 종목명

            Returns:
                ``bool``: 입력 종목명의 보유 여부

        __len__:
            Returns:
                ``int``: 보유 주식 종류의 수

        __getitem__:
            Args:
                idx (``int``): Index

            Returns:
                ``List[Union[int, float, str]]``: Index에 따른 주식의 매수 시점과 현재의 정보

        __call__:
            Returns:
                ``int``: 현재 보유 금액

    Examples:
        ``kor=True``:
            >>> balance = zz.quant.Balance()
            >>> "LG전자" in balance
            True
            >>> "삼성전자" in balance
            False
            >>> len(balance)
            1
            >>> balance[0]
            ['066570', 102200.0, 100200, 1, -1.95, -2000]
            >>> balance()
            1997997

        ``kor=False``:
            >>> balance = zz.quant.Balance(kor=False)
            >>> "아마존닷컴" in balance
            True
            >>> "삼성전자" in balance
            False
            >>> len(balance)
            1
            >>> balance[0]
            ['AMZN', 145.98, 146.5, 1, 0.36, 146.5]
            >>> balance()
            146.5
    """

    def __init__(self, path: Optional[str] = "./", kor: Optional[bool] = True) -> None:
        super().__init__(path)
        self.balance = {"stock": defaultdict(list)}
        self.kor = kor
        self.symbols = []
        response = self.get_balance(kor)
        if self.kor:
            for stock in response["output1"]:
                self.symbols.append(stock["prdt_name"])
                self.balance["stock"][stock["prdt_name"]] = [
                    stock["pdno"],  # 종목번호
                    float(stock["pchs_avg_pric"]),  # 매입평균가격 (매입금액 / 보유수량)
                    int(stock["prpr"]),  # 현재가
                    int(stock["hldg_qty"]),  # 보유수량
                    float(stock["evlu_pfls_rt"]),  # 평가손익율
                    int(stock["evlu_pfls_amt"]),  # 평가손익금액 (평가금액 - 매입금액)
                ]
            self.balance["cash"] = int(response["output2"][0]["nass_amt"])  # 순자산금액
        else:
            for stock in response["output1"]:
                self.symbols.append(stock["ovrs_item_name"])
                self.balance["stock"][stock["ovrs_item_name"]] = [
                    stock["ovrs_pdno"],  # 종목번호
                    float(stock["pchs_avg_pric"]),  # 매입평균가격 (매입금액 / 보유수량)
                    float(stock["now_pric2"]),  # 현재가
                    int(stock["ovrs_cblc_qty"]),  # 해외잔고수량
                    float(stock["evlu_pfls_rt"]),  # 평가손익율
                    float(stock["frcr_evlu_pfls_amt"]),  # 외화평가손익금액
                ]
            self.balance["cash"] = float(
                response["output2"]["tot_evlu_pfls_amt"]
            )  # 총평가손익금액

    def __contains__(self, item: Any) -> bool:
        return item in self.balance["stock"]

    def __len__(self) -> int:
        return len(self.balance["stock"])

    def __getitem__(self, idx: int) -> List[Union[int, float, str]]:
        return self.balance["stock"][self.symbols[idx]]

    def __call__(self) -> int:
        return self.balance["cash"]

    def items(self) -> ItemsView[str, List[Union[int, float, str]]]:
        """보유 주식의 반복문 사용을 위한 method

        Returns:
            ``ItemsView[str, List[Union[int, float, str]]]``: 보유 종목 code와 그에 따른 정보들

        Examples:
            >>> for k, v in balance.items():
            >>>     print(k, v)
        """
        return self.balance["stock"].items()

    def bought_symbols(self) -> List[str]:
        """보유 주식의 종목 code return

        Returns:
            ``List[str]``: 보유 주식의 종목 code들

        Examples:
            >>> balance.bought_symbols():
            ['066570']
        """
        return [value[0] for _, value in self.items()]

    def table(self) -> str:
        """현재 계좌의 상태를 image로 저장

        Returns:
            ``str``: 저장된 image의 절대 경로

        Examples:
            >>> balance.table()

            .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/290175594-7a372038-fe24-4527-a2e3-3c5efe445017.png
                :alt: Balance Table Result
                :align: center
                :width: 400px
        """
        if self() == 0:
            return None
        if self.kor:
            col = [
                "Purchase Price [￦]",
                "Current Price [￦]",
                "Quantity",
                "Profit and Loss (P&L) [%]",
                "Profit and Loss (P&L) [￦]",
            ]
        else:
            col = [
                "Purchase Price [$]",
                "Current Price [$]",
                "Quantity",
                "Profit and Loss (P&L) [%]",
                "Profit and Loss (P&L) [$]",
            ]
        row = []
        data = []
        purchase_total = 0
        current_total = 0
        for name, value in self.items():
            _, purchase, current, quantity, pal, pal_price = value
            row.append(name)
            data.append(
                [
                    _cash2str(purchase, self.kor),
                    _cash2str(current, self.kor),
                    quantity,
                    f"{pal}%",
                    _cash2str(pal_price, self.kor),
                ]
            )
            purchase_total += purchase * quantity
            current_total += current * quantity
        row.append("TOTAL")
        data.append(
            [
                _cash2str(purchase_total, self.kor),
                _cash2str(current_total, self.kor),
                "-",
                f"{(current_total-purchase_total)/purchase_total*100:.2f}%",
                f"{_cash2str(current_total - purchase_total, self.kor)}\n\n{_cash2str(self(), self.kor)}",
            ]
        )
        return table(
            data,
            col,
            row,
            title="balance",
            figsize=(16, int(1.2 * len(row))),
            dpi=100,
        )


class QuantSlackBot(SlackBot):
    """입력된 여러 종목에 대해 매수, 매도 signal을 판단하고 Slack으로 message와 graph를 전송하는 class

    Note:
        종목 code에 따른 종목명과 data를 불러오는 함수 ``_get_data`` 를 상속을 통해 정의해야 사용 가능

        .. code-block:: python

            def _get_data(self, symbol: str) -> Tuple[str, pd.core.frame.DataFrame]:
                title = data = None
                return title, data

    Args:
        symbols (``List[str]``): 종목 code들
        start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        top (``Optional[int]``): Experiment 과정에서 사용할 각 전략별 수
        report: (``Optional[bool]``): Experiment 결과 출력 여부
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
        >>> qsb = zz.quant.QuantSlackBot(symbols, token, channel)
        >>> qsb.index()

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/290177638-802e3118-439f-4cb8-972b-ded26734c50d.png
            :alt: Slack Bot Result
            :align: center
            :width: 800px
    """

    def __init__(
        self,
        symbols: List[str],
        start_day: Optional[str] = "",
        ohlc: Optional[str] = "",
        top: Optional[int] = 1,
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
    ) -> requests.models.Response:
        """``token`` 혹은 ``channel`` 이 입력되지 않을 시 전송 불가

        Args:
            message (``str``): 전송할 message
            codeblock (``Optional[bool]``): 전송되는 message의 스타일
            thread_ts (``Optional[str]``): 댓글을 전송할 thread의 timestamp

        Returns:
            ``requests.models.Response``: Slack Bot의 응답
        """
        if self.slack:
            return super().message(message, codeblock, thread_ts)
        return None

    def file(
        self, path: str, thread_ts: Optional[str] = None
    ) -> requests.models.Response:
        """``token`` 혹은 ``channel`` 이 입력되지 않을 시 전송 불가

        Args:
            path (``str``): 전송할 file 경로
            thread_ts (``Optional[str]``): 댓글을 전송할 thread의 timestamp

        Returns:
            ``requests.models.Response``: Slack Bot의 응답
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
        plt.subplot(3, 1, 1)
        if self.kor:
            dim = "₩"
        else:
            dim = "$"
        hist(
            {"Buy": quant.transaction["buy"], "Sell": quant.transaction["sell"]},
            xlab=f"매수/매도가 [{dim}]",
            ylab="",
            title="",
            save=False,
        )
        plt.subplot(3, 1, 2)
        hist(
            {"Profit": quant.transaction["profit"]},
            xlab="이율 [%]",
            ylab="",
            title="",
            save=False,
        )
        plt.subplot(3, 1, 3)
        hist(
            {"Period": quant.transaction["period"]},
            xlab="거래 기간 [일]",
            ylab="",
            title="",
            save=False,
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
        ] += f"\t:technologist: Signal Info: {today['total'][1]:.2f}% (`{int(today['total'][0])}/{int(quant.total_cnt)}`) → {logic[today['logic']]}\n"
        report["param"] += "> :information_desk_person: *Parameter Info*"
        for key in quant.methods:
            report[
                "main"
            ] += f"\t\t:hammer: {key.replace('_', ' ').upper()}: {today[key][1]:.2f}% (`{int(today[key][0])}/{int(quant.method_cnt[key])}`)\n"
            report[
                "param"
            ] += f"\n\t:hammer: {key.replace('_', ' ').upper()}: `{'`, `'.join(quant.exps_str[key])}`"
        report["main"] += "\t:memo: THRESHOLD:\n"
        report[
            "main"
        ] += f"\t\t:arrow_double_up: BUY: `{quant.threshold_buy}`\n\t\t:arrow_double_down: SELL: `{quant.threshold_sell}`"
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

    def _get_data(self, symbol: str) -> Tuple[str, pd.core.frame.DataFrame]:
        title = data = None
        return title, data

    def _run(self, args: List[str]) -> Tuple[Dict[str, str], Quant]:
        symbol, mode = args
        try:
            title, data = self._get_data(symbol)
            if len(data) < 20:
                return None, None
        except KeyError as error:
            response = self.message(f":x: '`{symbol}`' was not found")
            self.message(str(error), True, response.json()["ts"])
            self.message(traceback.format_exc(), True, response.json()["ts"])
            return None, None
        try:
            quant = Quant(title, data, ohlc=self.ohlc, top=self.top, report=self.report)
            today = quant()
        except IndexError as error:
            self.message(f":x: '`{symbol}`' ({title}): {data.index[0]} ({len(data)})")
            self.message(str(error), True, response.json()["ts"])
            self.message(traceback.format_exc(), True, response.json()["ts"])
            return None, None
        if mode == "Buy":
            positions = ["Buy"]
        elif mode != "NULL":
            positions = ["Buy", "Sell", "None"]
        if today["position"] in positions:
            return self._report(symbol, quant, today), quant
        return None, quant

    def _send(self, report: Dict[str, str]) -> None:
        if report is None:
            return
        response = self.message(report["main"])
        thread_ts = response.json()["ts"]
        self.file(report["candle"], thread_ts)
        response = self.message(report["backtest"], thread_ts=thread_ts)
        self.file(report["hist"], thread_ts)
        response = self.message(report["param"], thread_ts=thread_ts)

    def _analysis_update(
        self, methods: Tuple[str], exps: Dict[str, List[Dict[str, int]]]
    ) -> None:
        for method in methods:
            self.methods[method.replace("_", " ").upper()] += 1
        for strategy, counts in exps.items():
            if strategy not in self.exps.keys():
                self.exps[strategy] = [defaultdict(int) for _ in range(len(counts))]
            for idx, count in enumerate(counts):
                for param, cnt in count.items():
                    self.exps[strategy][idx][param] += cnt

    def _analysis_send(self) -> None:
        response = self.message("> :memo: Parameter Analysis")
        thread_ts = response.json()["ts"]
        path = barv(self.methods, "전략", "", "Strategy")
        self.file(path, thread_ts)
        for strategy, counts in self.exps.items():
            figure((18, 8))
            stg = True
            for idx, count in enumerate(counts):
                try:
                    plt.subplot(1, len(counts), idx + 1)
                    barh(count, "", "", "", save=False)
                except ValueError:
                    stg = False
                    print(f"'{strategy}' was not used: {count}")
            if stg:
                path = savefig(strategy, dpi=100)
                self.file(path, thread_ts)
            else:
                self.message(
                    f":no_bell: '{strategy}' was not used", thread_ts=thread_ts
                )

    def _inference(self, symbols: List[str], mode: str) -> None:
        start = time.time()
        if self.analysis:
            self.methods = defaultdict(int)
            self.exps = defaultdict(list)
        response = self.message(f"> :moneybag: Check {mode} Signals")
        self.message(", ".join(symbols), True, response.json()["ts"])
        if self.mp_num == 0 or self.mp_num >= len(symbols):
            for symbol in symbols:
                report, quant = self._run([symbol, mode])
                self._send(report)
                if self.analysis and quant is not None:
                    self._analysis_update(quant.methods, quant.exps_cnt)
        else:
            args = [[symbol, mode] for symbol in symbols]
            with mp.Pool(processes=self.mp_num) as pool:
                results = pool.map(self._run, args)
            for report, quant in results:
                self._send(report)
                if self.analysis and quant is not None:
                    self._analysis_update(quant.methods, quant.exps_cnt)
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


class QuantSlackBotKI(Balance, QuantSlackBot):
    """한국투자증권 API를 기반으로 입력된 여러 종목에 대해 매수, 매도 signal을 판단하고 Slack으로 message와 graph를 전송하는 class

    Args:
        symbols (``Optional[List[str]]``): 종목 code들
        start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        top (``Optional[int]``): Experiment 과정에서 사용할 각 전략별 수
        report: (``Optional[bool]``): Experiment 결과 출력 여부
        token (``Optional[str]``): Slack Bot의 token
        channel (``Optional[str]``): Slack Bot이 전송할 channel
        name (``Optional[str]``): Slack Bot의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Bot의 표시될 사진 (emoji)
        mp_num (``Optional[int]``): 병렬 처리에 사용될 process의 수 (``0``: 직렬 처리)
        analysis (``Optional[bool]``): 각 전략의 보고서 전송 여부
        kor (``Optional[bool]``): 국내 여부
        path (``Optional[str]``): ``secret.key`` 혹은 ``token.dat`` 이 포함된 경로

    Attributes:
        exps (``Dict[str, List[Dict[str, int]]]``): 각 전략에 따른 parameter 분포

    Examples:
        >>> qsb = zz.quant.QuantSlackBotKI(symbols, token=token, channel=channel)
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        start_day: Optional[str] = "",
        ohlc: Optional[str] = "",
        top: Optional[int] = 1,
        report: Optional[bool] = False,
        token: Optional[str] = None,
        channel: Optional[str] = None,
        name: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        mp_num: Optional[int] = 0,
        analysis: Optional[bool] = False,
        kor: Optional[bool] = True,
        path: Optional[str] = "./",
    ) -> None:
        Balance.__init__(self, path, kor)
        if symbols is None:
            symbols = []
        QuantSlackBot.__init__(
            self,
            symbols,
            start_day,
            ohlc,
            top,
            report,
            token,
            channel,
            name,
            icon_emoji,
            mp_num,
            analysis,
            kor,
        )
        symbols_bought = self.bought_symbols()
        for symbol in symbols_bought:
            if symbol in symbols:
                symbols.remove(symbol)
        self.symbols = symbols
        self.symbols_bought = symbols_bought

    def _get_data(self, symbol: str) -> Tuple[str, pd.core.frame.DataFrame]:
        response = self.get_ohlcv(symbol, start_day=self.start_day, kor=self.kor)
        title, data = self.response2ohlcv(response)
        return title, data

    def sell(self) -> None:
        """매도 signals 탐색

        한국투자증권의 잔고와 주식 보유 상황을 image로 변환하여 slack으로 전송 및 보유 중인 주식에 대해 매도 signals 탐색
        """
        path = self.table()
        if path is None:
            self.message("Balance: NULL", True)
            return None
        response = self.message("> :bank: Balance")
        self.file(path, response.json()["ts"])
        self._inference(self.symbols_bought, "Sell")
        return None


class QuantSlackBotFDR(QuantSlackBot):
    """`FinanceDataReader <https://github.com/FinanceData/FinanceDataReader>`_ module 기반으로 입력된 여러 종목에 대해 매수, 매도 signal을 판단하고 Slack으로 message와 graph를 전송하는 class

    Args:
        symbols (``Union[int, List[str]]``): 종목 code들 혹은 시가 총액 순위
        start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        top (``Optional[int]``): Experiment 과정에서 사용할 각 전략별 수
        report: (``Optional[bool]``): Experiment 결과 출력 여부
        token (``Optional[str]``): Slack Bot의 token
        channel (``Optional[str]``): Slack Bot이 전송할 channel
        name (``Optional[str]``): Slack Bot의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Bot의 표시될 사진 (emoji)
        mp_num (``Optional[int]``): 병렬 처리에 사용될 process의 수 (``0``: 직렬 처리)
        analysis (``Optional[bool]``): 각 전략의 보고서 전송 여부
        kor (``Optional[bool]``): 국내 여부

    Attributes:
        exps (``Dict[str, List[Dict[str, int]]]``): 각 전략에 따른 parameter 분포
        market (``pd.core.frame.DataFrame``): ``kor`` 에 따른 시장 목록

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

    def _get_data(self, symbol: str) -> Tuple[str, pd.core.frame.DataFrame]:
        if self.kor:
            title = self.market[self.market["Code"] == symbol].iloc[0, 1]
        else:
            title = self.market[self.market["Symbol"] == symbol].iloc[0, 1]
        data = fdr.DataReader(symbol, self.start_day)
        return title, data
