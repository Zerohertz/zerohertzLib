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
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, ItemsView, List, Optional, Tuple, Union

import FinanceDataReader as fdr
import pandas as pd
import requests
from matplotlib import pyplot as plt

from zerohertzLib.api import KoreaInvestment, SlackBot
from zerohertzLib.plot import barh, candle, figure, savefig, table

from .backtest import Experiments, backtest


class Quant(Experiments):
    """한 가지 종목에 대해 full factorial design 기반의 backtest를 수행하고 최적의 전략을 융합하는 class

    Args:
        title(``str``): 종목 이름
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        top (``Optional[int]``): Experiments 과정에서 사용할 각 전략별 수
        methods (``Optional[List[str]]``): 사용할 전략들의 함수명

    Attributes:
        signals (``pd.core.frame.DataFrame``): 융합된 전략의 signal
        params (`Dict[str, List[str]]`): 각 전략에 따른 paramter 문자열
        cnt (``Dict[str, int]``): 각 전략에 따른 수
        exps (``Dict[str, List[Dict[str, int]]]``): 각 전략에 따른 parameter 분포
        profit (``float``): 융합된 전략의 backtest profit
        threshold_buy (``int``): 융합된 전략의 매수 signal threshold
        threshold_sell (``int``): 융합된 전략의 매도 signal threshold
        methods (``Tuple[str]``): 융합된 전략명

    Examples:
        >>> qnt = zz.quant.Quant(title, data, top=3)
        >>> qnt.signals.columns
        Index(['moving_average', 'rsi', 'bollinger_bands', 'momentum', 'signals'], dtype='object')
        >>> qnt.params
        defaultdict(<class 'list'>, {'moving_average': ['10-80-150', '10-80-100', '10-80-200'], ...})
        >>> qnt.cnt
        defaultdict(<class 'int'>, {'moving_average': 3, 'rsi': 3, 'bollinger_bands': 3, 'momentum': 3})
        >>> qnt.exps
        defaultdict(None, {'moving_average': [defaultdict(<class 'int'>, {'10': 3}), defaultdict(<class 'int'>, {'80': 3}), ...]})
        >>> qnt.profit
        409.6687719932959
        >>> qnt.threshold_buy
        4
        >>> qnt.threshold_sell
        -6
        >>> qnt.methods
        ('rsi', 'bollinger_bands', 'momentum')
    """

    def __init__(
        self,
        title: str,
        data: pd.core.frame.DataFrame,
        ohlc: Optional[str] = "",
        top: Optional[int] = 1,
        methods: Optional[List[str]] = None,
    ) -> None:
        super().__init__(title, data, ohlc, False, True)
        self.signals = pd.DataFrame(index=data.index)
        self.params = defaultdict(list)
        self.cnt = defaultdict(int)
        self.exps = defaultdict()
        if methods is None:
            methods = [
                "moving_average",
                "rsi",
                "bollinger_bands",
                "momentum",
            ]
        # 선정한 전략들의 parameter 최적화
        self.cnt_total = 0
        for method in methods:
            if hasattr(self, method):
                self.signals[method] = 0
                profits, signals, params, exps = getattr(self, method)()
                exps_cnt = [defaultdict(int) for _ in range(len(exps[0]))]
                for profit, signal, param, exp in zip(
                    profits[:top], signals[:top], params[:top], exps[:top]
                ):
                    if profit > 0:
                        self.signals[method] += signal["signals"]
                        self.params[method].append(param)
                        for i, ex in enumerate(exp):
                            exps_cnt[i][str(ex)] += 1
                        self.cnt[method] += 1
                        self.cnt_total += 1
                self.exps[method] = exps_cnt
            else:
                raise AttributeError(f"'Quant' object has no attribute '{method}'")
        # 전략 간 조합 최적화
        if self.cnt_total >= 1:
            backtests = []
            for cnt in range(1, len(methods)):
                for methods_in_use in combinations(methods, cnt):
                    miu_total = 0
                    for miu in methods_in_use:
                        miu_total += self.cnt[miu]
                        if self.cnt[miu] < 1:
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
                                    "backtest": self.signals["backtest"],
                                }
                            )
            backtests.sort(key=lambda x: x["weighted_profit"], reverse=True)
            # 최적 융합 전략
            self.profit = backtests[0]["profit"]
            self.threshold_sell, self.threshold_buy = backtests[0]["threshold"]
            self.methods = backtests[0]["methods"]
            self.signals["signals"] = self.signals.loc[:, backtests[0]["methods"]].sum(
                1
            )
            self.signals["backtest"] = backtests[0]["backtest"]
            self.cnt_total = backtests[0]["total"]
            self.transaction = backtests[0]["transaction"]

    def run(self, day: Optional[str] = -1) -> Dict[str, list]:
        """입력된 날짜에 대해 분석 정보 return

        Args:
            day (``Optional[str]``): 분석할 날짜

        Returns:
            ``Dict[str, float]``: 각 전략에 따른 분석 정보 및 결론

        Examples:
            >>> qnt.run()
            defaultdict(<class 'list'>, {'rsi': [0, 0.0], 'bollinger_bands': [0, 0.0], 'momentum': [3.0, 100.0], 'total': [3.0, 33.33333333333333], 'position': 'None'})
            >>> qnt.run("20180425")
            defaultdict(<class 'list'>, {'rsi': [0, 0.0], 'bollinger_bands': [2, 66.66666666666666], 'momentum': [2.0, 66.66666666666666], 'total': [4.0, 44.44444444444444], 'position': 'Buy'})
            >>> qnt.run("1998-10-23")
            defaultdict(<class 'list'>, {'rsi': [0, 0.0], 'bollinger_bands': [0, 0.0], 'momentum': [0.0, 0.0], 'total': [0.0, 0.0], 'position': 'None'})
        """
        if self.cnt_total < 1:
            return {"position": "NULL"}
        if day != -1 and "-" not in day:
            day = day[:4] + "-" + day[4:6] + "-" + day[6:8]
        possibility = defaultdict(list)
        for key in self.methods:
            possibility[key] = [
                self.signals[key][day],
                self.signals[key][day] / self.cnt[key] * 100,
            ]
        possibility["total"] = [
            self.signals["signals"][day],
            self.signals["signals"][day] / self.cnt_total * 100,
        ]
        if self.threshold_buy <= possibility["total"][0]:
            possibility["position"] = "Buy"
        elif self.threshold_sell >= -possibility["total"][0]:
            possibility["position"] = "Sell"
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

    def _cash2str(self, cash: str) -> str:
        if self.kor:
            return f"{cash:,.0f}원"
        return f"${cash:,.2f}"

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
        """
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
                    self._cash2str(purchase),
                    self._cash2str(current),
                    quantity,
                    f"{pal}%",
                    self._cash2str(pal_price),
                ]
            )
            purchase_total += purchase * quantity
            current_total += current * quantity
        row.append("TOTAL")
        data.append(
            [
                self._cash2str(purchase_total),
                self._cash2str(current_total),
                "-",
                f"{(current_total-purchase_total)/purchase_total*100:.2f}%",
                f"{self._cash2str(current_total - purchase_total)}\n\n{self._cash2str(self())}",
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
        token (``Optional[str]``): Slack Bot의 token
        channel (``Optional[str]``): Slack Bot이 전송할 channel
        start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        top (``Optional[int]``): Experiments 과정에서 사용할 각 전략별 수
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

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/289048168-9c339478-4d61-4ac6-9ecd-e0b9433af264.png
            :alt: Slack Bot Result
            :align: center
            :width: 400px
    """

    def __init__(
        self,
        symbols: List[str],
        token: Optional[str] = None,
        channel: Optional[str] = None,
        start_day: Optional[str] = "",
        ohlc: Optional[str] = "",
        top: Optional[int] = 1,
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

    def message(
        self,
        message: str,
        codeblock: Optional[bool] = False,
    ) -> requests.models.Response:
        """``token`` 혹은 ``channel`` 이 입력되지 않을 시 전송 불가"""
        if self.slack:
            return super().message(message, codeblock)
        return None

    def file(self, path: str) -> requests.models.Response:
        """``token`` 혹은 ``channel`` 이 입력되지 않을 시 전송 불가"""
        if self.slack:
            return super().file(path)
        return None

    def _cash2str(self, cash: str) -> str:
        if self.kor:
            return f"{cash:,.0f}원"
        return f"${cash:,.2f}"

    def _report(self, name: str, quant: Quant, today: Dict[str, list]):
        report = ""
        if today["position"] == "Buy":
            report += f"> :chart_with_upwards_trend: [Buy Signal] *{name}*\n"
        elif today["position"] == "Sell":
            report += f"> :chart_with_downwards_trend: [Sell Signal] *{name}*\n"
        else:
            report += f"> :egg: [None Signal] *{name}*\n"
        report += f"\t:heavy_dollar_sign: SIGNAL's INFO: {today['total'][1]:.2f}% (`{int(today['total'][0])}/{int(quant.cnt_total)}`)\n"
        for key in quant.methods:
            report += f"\t:hammer: {key.replace('_', ' ').upper()}:\t{today[key][1]:.2f}%\t(`{int(today[key][0])}/{int(quant.cnt[key])}`)\t"
            report += f"`{'`, `'.join(quant.params[key])}`\n"
        report += "\t:memo: THRESHOLD:\n"
        report += f"\t\t:arrow_double_up: BUY: `{quant.threshold_buy}`\n\t\t:arrow_double_down: SELL: `{quant.threshold_sell}`\n"
        report += (
            f"*Backtest*\n\t:money_with_wings: Total Profit:\t{quant.profit:.2f}%\n"
        )
        report += f"\t:chart_with_upwards_trend: Total Buy:\t{self._cash2str(quant.transaction['buy'])}\n"
        report += f"\t:chart_with_downwards_trend: Total Sell:\t{self._cash2str(quant.transaction['sell'])}\n"
        transaction_price = [
            self._cash2str(price) for price in quant.transaction["price"]
        ]
        transaction_profit = [f"{price:.2f}%" for price in quant.transaction["profit"]]
        transaction_price = "```" + " -> ".join(transaction_price) + "```"
        transaction_profit = "```" + " -> ".join(transaction_profit) + "```"
        report += f"\t:bank: Transactions:\n{transaction_price}\n{transaction_profit}"
        return report

    def _get_data(self, symbol: str) -> Tuple[str, pd.core.frame.DataFrame]:
        title = data = None
        return title, data

    def _run(
        self, args: List[Union[str, str]]
    ) -> Tuple[str, str, Dict[str, List[Dict[str, int]]]]:
        symbol, mode = args
        try:
            title, data = self._get_data(symbol)
        except KeyError:
            self.message(f"'{symbol}' is not found")
            return None, None, None
        try:
            quant = Quant(title, data, ohlc=self.ohlc, top=self.top)
            today = quant.run()
        except IndexError:
            self.message(f"{title}: {data.index[0]} ({len(data)})")
            return None, None, None
        if mode == "Buy":
            positions = ["Buy"]
        elif mode != "NULL":
            positions = ["Buy", "Sell", "None"]
        if today["position"] in positions:
            path = candle(
                quant.data[-500:],
                quant.title,
                signals=quant.signals.iloc[-500:, :].loc[
                    :, [*quant.methods, "signals", "backtest"]
                ],
                dpi=100,
                threshold=(quant.threshold_sell, quant.threshold_buy),
            )
            return self._report(title, quant, today), path, quant.exps
        return None, None, quant.exps

    def _send(self, message: str, image: str) -> None:
        if message is None or image is None:
            return
        self.message(message)
        self.file(image)

    def _analysis_update(self, exps: Dict[str, List[Dict[str, int]]]) -> None:
        """
        exps (``Dict[str, List[Dict[str, int]]]``)
        self.exps (``Dict[str, List[Dict[str, int]]]``)
        """
        for strategy, counts in exps.items():
            if strategy not in self.exps.keys():
                self.exps[strategy] = [defaultdict(int) for _ in range(len(counts))]
            for idx, count in enumerate(counts):
                for param, cnt in count.items():
                    self.exps[strategy][idx][param] += cnt

    def _analysis_send(self) -> None:
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
                self.file(path)
            else:
                self.message(f"'{strategy}' was not used", codeblock=True)

    def _inference(self, symbols: List[str], mode: str) -> None:
        if self.mp_num == 0 or self.mp_num >= len(symbols):
            for symbol in symbols:
                message, image, exps = self._run([symbol, mode])
                self._send(message, image)
                if self.analysis and exps is not None:
                    self._analysis_update(exps)
        else:
            args = [[symbol, mode] for symbol in symbols]
            with mp.Pool(processes=self.mp_num) as pool:
                results = pool.map(self._run, args)
            for message, image, exps in results:
                self._send(message, image)
                if self.analysis and exps is not None:
                    self._analysis_update(exps)

    def buy(self) -> None:
        """매수 signals 탐색"""
        if self.analysis:
            self.exps = defaultdict(list)
        self.message("> Check Buy Signals")
        self.message(", ".join(self.symbols), codeblock=True)
        self._inference(self.symbols, "Buy")
        if self.analysis:
            self._analysis_send()
        self.message("Done!")

    def index(self) -> None:
        """모든 signals 탐색"""
        if self.analysis:
            self.exps = defaultdict(list)
        self.message("> Check Index Signals")
        self.message(", ".join(self.symbols), codeblock=True)
        self._inference(self.symbols, "Index")
        if self.analysis:
            self._analysis_send()
        self.message("Done!")


class QuantSlackBotKI(Balance, QuantSlackBot):
    """한국투자증권 API를 기반으로 입력된 여러 종목에 대해 매수, 매도 signal을 판단하고 Slack으로 message와 graph를 전송하는 class

    Args:
        symbols (``Optional[List[str]]``): 종목 code들
        token (``Optional[str]``): Slack Bot의 token
        channel (``Optional[str]``): Slack Bot이 전송할 channel
        path (``Optional[str]``): ``secret.key`` 혹은 ``token.dat`` 이 포함된 경로
        start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        top (``Optional[int]``): Experiments 과정에서 사용할 각 전략별 수
        name (``Optional[str]``): Slack Bot의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Bot의 표시될 사진 (emoji)
        mp_num (``Optional[int]``): 병렬 처리에 사용될 process의 수 (``0``: 직렬 처리)
        analysis (``Optional[bool]``): 각 전략의 보고서 전송 여부
        kor (``Optional[bool]``): 국내 여부

    Attributes:
        exps (``Dict[str, List[Dict[str, int]]]``): 각 전략에 따른 parameter 분포

    Examples:
        >>> qsb = zz.quant.QuantSlackBotKI(symbols, token, channel)
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        token: Optional[str] = None,
        channel: Optional[str] = None,
        path: Optional[str] = "./",
        start_day: Optional[str] = "",
        ohlc: Optional[str] = "",
        top: Optional[int] = 1,
        name: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        mp_num: Optional[int] = 0,
        analysis: Optional[bool] = False,
        kor: Optional[bool] = True,
    ) -> None:
        Balance.__init__(self, path, kor)
        if symbols is None:
            symbols = []
        QuantSlackBot.__init__(
            self,
            symbols,
            token,
            channel,
            start_day,
            ohlc,
            top,
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
        if self.analysis:
            self.exps = defaultdict(list)
        self.message("> Balance")
        path = self.table()
        self.file(path)
        self.message("> Check Sell Signals")
        self.message(", ".join(self.symbols_bought), codeblock=True)
        self._inference(self.symbols_bought, "Sell")
        if self.analysis:
            self._analysis_send()
        self.message("Done!")


class QuantSlackBotFDR(QuantSlackBot):
    """`FinanceDataReader <https://github.com/FinanceData/FinanceDataReader>`_ module 기반으로 입력된 여러 종목에 대해 매수, 매도 signal을 판단하고 Slack으로 message와 graph를 전송하는 class

    Args:
        symbols (``Union[int, List[str]]``): 종목 code들 혹은 시가 총액 순위
        token: Optional[str] = None,
        channel: Optional[str] = None,
        start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        top (``Optional[int]``): Experiments 과정에서 사용할 각 전략별 수
        name (``Optional[str]``): Slack Bot의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Bot의 표시될 사진 (emoji)
        mp_num (``Optional[int]``): 병렬 처리에 사용될 process의 수 (``0``: 직렬 처리)
        analysis (``Optional[bool]``): 각 전략의 보고서 전송 여부
        kor (``Optional[bool]``): 국내 여부

    Attributes:
        exps (``Dict[str, List[Dict[str, int]]]``): 각 전략에 따른 parameter 분포
        market (``pd.core.frame.DataFrame``): ``kor`` 에 따른 시장 목록

    Examples:
        >>> qsb = zz.quant.QuantSlackBotFDR(symbols, token, channel)
        >>> qsb = zz.quant.QuantSlackBotFDR(10, token, channel)
    """

    def __init__(
        self,
        symbols: Union[int, List[str]],
        token: Optional[str] = None,
        channel: Optional[str] = None,
        start_day: Optional[str] = "",
        ohlc: Optional[str] = "",
        top: Optional[int] = 1,
        name: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        mp_num: Optional[int] = 0,
        analysis: Optional[bool] = False,
        kor: Optional[bool] = True,
    ) -> None:
        QuantSlackBot.__init__(
            self,
            symbols,
            token,
            channel,
            start_day,
            ohlc,
            top,
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
