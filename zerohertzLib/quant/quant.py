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
from typing import Any, Dict, ItemsView, List, Optional, Tuple, Union

import FinanceDataReader as fdr
import pandas as pd
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
        backtest (``List[Tuple[Any]]``): 융합된 전략의 backtest 결과
        params (`Dict[str, List[str]]`): 각 전략에 따른 paramter 문자역
        exps (``Dict[str, List[Dict[str, int]]]``): 각 전략에 따른 parameter 분포

    Examples:
        >>> qnt = zz.quant.Quant(title, data, top=3)
        >>> qnt.backtest[0]
        (5.692595336450012, 4, 33.33333333333333, defaultdict(<class 'list'>, {'price': [2385.95, -2521.7725], 'profit': [5.385993383622044]}))
        >>> qnt.params
        defaultdict(<class 'list'>, {'moving_average': ['5-75-50', '5-80-50', '5-65-50'], ...})
        >>> qnt.exps
        defaultdict(None, {'moving_average': [defaultdict(<class 'int'>, {'5': 3}), defaultdict(<class 'int'>, {'75': 1, '80': 1, '65': 1}), ...], ...})
    """

    def __init__(
        self,
        title: str,
        data: pd.core.frame.DataFrame,
        ohlc: Optional[str] = "",
        top: Optional[int] = 1,
        methods: Optional[List[str]] = None,
    ) -> None:
        super().__init__(title, data, ohlc, False, False)
        self.data = data
        self.signals = pd.DataFrame(index=data.index)
        self.params = defaultdict(list)
        self.exps = defaultdict()
        self.cnt = defaultdict(int)
        self.cnt_total = 0
        if methods is None:
            methods = [
                "moving_average",
                "rsi",
                "bollinger_bands",
                "momentum",
            ]
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
        if self.cnt_total >= 1:
            self.cnt["total"] = self.cnt_total
            self.signals["signals"] = self.signals.sum(1)
            self.backtest = []
            for threshold in range(1, self.cnt_total + 1):
                results = backtest(self.data, self.signals, threshold=threshold)
                self.backtest.append(
                    (
                        results["profit"],
                        threshold,
                        threshold / self.cnt_total * 100,
                        results["transaction"],
                    )
                )
            self.backtest.sort(key=lambda x: x[0], reverse=True)
            self.threshold_abs = self.backtest[0][1]
            self.threshold_rel = self.backtest[0][2]

    def run(self, day: Optional[str] = -1) -> Dict[str, Any]:
        """입력된 날짜에 대해 분석 정보 return

        Args:
            day (``Optional[str]``): 분석할 날짜

        Returns:
            ``Dict[str, Any]``: 각 전략에 따른 분석 정보 및 결론

        Examples:
            >>> qnt.run()
            defaultdict(<class 'float'>, {'moving_average': 0.0, 'rsi': 0.0, 'bollinger_bands': 0.0, 'momentum': 0.0, 'total': 0.0, 'position': 'None'})
            >>> qnt.run("20231023")
            defaultdict(<class 'float'>, {'moving_average': -100.0, 'rsi': 0.0, 'bollinger_bands': 66.66666666666666, 'momentum': 0.0, 'total': -8.333333333333332, 'position': 'None'})
            >>> qnt.run("2023-04-21")
            defaultdict(<class 'float'>, {'moving_average': 100.0, 'rsi': 0.0, 'bollinger_bands': 0.0, 'momentum': 33.33333333333333, 'total': 33.33333333333333, 'position': 'Buy'})
        """
        if self.cnt_total < 1:
            return {"position": "NULL"}
        if day != -1 and "-" not in day:
            day = day[:4] + "-" + day[4:6] + "-" + day[6:8]
        possibility = defaultdict(float)
        for key, value in self.cnt.items():
            if key != "total" and value != 0:
                possibility[key] = self.signals[key][day] / value * 100
        possibility["total"] = self.signals["signals"][day] / self.cnt_total * 100
        if self.threshold_rel <= possibility["total"]:
            possibility["position"] = "Buy"
        elif self.threshold_rel <= -possibility["total"]:
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
    """

    def __init__(self, path: Optional[str] = "./") -> None:
        super().__init__(path)
        self.balance = {"stock": defaultdict(list)}
        self.symbols = []
        response = self.get_balance()
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

    def __contains__(self, item: Any) -> bool:
        return item in self.balance["stock"]

    def __len__(self) -> int:
        return len(self.balance["stock"])

    def __getitem__(self, idx: int) -> List[Union[int, float, str]]:
        return self.balance["stock"][self.symbols[idx]]

    def __call__(self) -> int:
        return self.balance["cash"]

    def _cash2str(self, cash: str) -> str:
        return f"{cash:,.0f}원"

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
        col = [
            "Purchase Price [￦]",
            "Current Price [￦]",
            "Quantity",
            "Profit and Loss (P&L) [%]",
            "Profit and Loss (P&L) [￦]",
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
                f"{(current_total-purchase_total)/purchase_total:.2f}%",
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
        token (``str``): Slack Bot의 token
        channel (``str``): Slack Bot이 전송할 channel
        start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
        top (``Optional[int]``): Experiments 과정에서 사용할 각 전략별 수
        name (``Optional[str]``): Slack Bot의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Bot의 표시될 사진 (emoji)
        mp_num (``Optional[int]``): 병렬 처리에 사용될 process의 수 (``0``: 직렬 처리)
        analysis (``Optional[bool]``): 각 전략의 보고서 전송 여부

    Attributes:
        exps (``Dict[str, List[Dict[str, int]]]``): 각 전략에 따른 parameter 분포

    Examples:
        >>> qsb = zz.quant.QuantSlackBot(symbols, token, channel)
        >>> qsb.index()

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/288455654-53146aff-bf04-411e-a932-954e90c81d97.png
            :alt: Slack Bot Result
            :align: center
            :width: 400px

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/288455675-90300812-c23f-4222-a18f-0c621b03a633.png
            :alt: Slack Bot Result
            :align: center
            :width: 400px
    """

    def __init__(
        self,
        symbols: List[str],
        token: str,
        channel: str,
        start_day: Optional[str] = "",
        top: Optional[int] = 1,
        name: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        mp_num: Optional[int] = 0,
        analysis: Optional[bool] = False,
    ) -> None:
        SlackBot.__init__(self, token, channel, name, icon_emoji)
        self.symbols = symbols
        self.start_day = start_day
        self.top = top
        if mp_num > mp.cpu_count():
            self.mp_num = mp.cpu_count()
        else:
            self.mp_num = mp_num
        self.analysis = analysis

    def _cash2str(self, cash: str) -> str:
        return f"{cash:,.0f}원"

    def _report(self, name: str, quant: Quant, today: Dict[str, Any]):
        report = ""
        if today["position"] == "Buy":
            report += f"> :chart_with_upwards_trend: [Buy Signal] *{name}*\n"
        elif today["position"] == "Sell":
            report += f"> :chart_with_downwards_trend: [Sell Signal] *{name}*\n"
        else:
            report += f"> :egg: [None Signal] *{name}*\n"
        for key, value in today.items():
            if key != "position":
                if key == "total":
                    report += f":heavy_dollar_sign: {key.replace('_', ' ').upper()}:\t{value:.2f}%\t({quant.cnt[key]})\n"
                else:
                    report += f":heavy_dollar_sign: {key.replace('_', ' ').upper()}:\t{value:.2f}%\t(`{'`, `'.join(quant.params[key])}`)\n"
        report += f":heavy_dollar_sign: THRESHOLD: `{quant.threshold_abs}`, `{quant.threshold_rel:.1f}%`\n"
        report += f"*Backtest*\n:white_check_mark: Total Profit:\t{quant.backtest[0][0]:.2f}%\n"
        transaction_price = [
            self._cash2str(price) for price in quant.backtest[0][3]["price"]
        ]
        transaction_profit = [
            f"{price:.2f}%" for price in quant.backtest[0][3]["profit"]
        ]
        transaction_price = "```" + " -> ".join(transaction_price) + "```"
        transaction_profit = "```" + " -> ".join(transaction_profit) + "```"
        report += f":white_check_mark: Transactions:\n{transaction_price}\n{transaction_profit}"
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
            quant = Quant(title, data, top=self.top)
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
                signals=quant.signals.iloc[-500:, :].loc[:, ["signals"]],
                dpi=100,
                threshold=quant.threshold_abs,
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
        symbols (``List[str]``): 종목 code들
        token (``str``): Slack Bot의 token
        channel (``str``): Slack Bot이 전송할 channel
        path (``Optional[str]``): ``secret.key`` 혹은 ``token.dat`` 이 포함된 경로
        start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
        top (``Optional[int]``): Experiments 과정에서 사용할 각 전략별 수
        name (``Optional[str]``): Slack Bot의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Bot의 표시될 사진 (emoji)
        mp_num (``Optional[int]``): 병렬 처리에 사용될 process의 수 (``0``: 직렬 처리)
        analysis (``Optional[bool]``): 각 전략의 보고서 전송 여부

    Attributes:
        exps (``Dict[str, List[Dict[str, int]]]``): 각 전략에 따른 parameter 분포

    Examples:
        >>> qsb = zz.quant.QuantSlackBotKI(symbols, token, channel)
    """

    def __init__(
        self,
        symbols: List[str],
        token: str,
        channel: str,
        path: Optional[str] = "./",
        start_day: Optional[str] = "",
        top: Optional[int] = 1,
        name: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        mp_num: Optional[int] = 0,
        analysis: Optional[bool] = False,
    ) -> None:
        Balance.__init__(self, path)
        QuantSlackBot.__init__(
            self,
            symbols,
            token,
            channel,
            start_day,
            top,
            name,
            icon_emoji,
            mp_num,
            analysis,
        )
        symbols_bought = self.bought_symbols()
        for symbol in symbols_bought:
            if symbol in symbols:
                symbols.remove(symbol)
        self.symbols = symbols
        self.symbols_bought = symbols_bought

    def _get_data(self, symbol: str) -> Tuple[str, pd.core.frame.DataFrame]:
        response = self.get_ohlcv(symbol, start_day=self.start_day)
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
        symbols (``List[str]``): 종목 code들
        token (``str``): Slack Bot의 token
        channel (``str``): Slack Bot이 전송할 channel
        start_day (``Optional[str]``): 조회 시작 일자 (``YYYYMMDD``)
        top (``Optional[int]``): Experiments 과정에서 사용할 각 전략별 수
        name (``Optional[str]``): Slack Bot의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Bot의 표시될 사진 (emoji)
        mp_num (``Optional[int]``): 병렬 처리에 사용될 process의 수 (``0``: 직렬 처리)
        analysis (``Optional[bool]``): 각 전략의 보고서 전송 여부

    Attributes:
        exps (``Dict[str, List[Dict[str, int]]]``): 각 전략에 따른 parameter 분포
        krx (``pd.core.frame.DataFrame``): KRX 상장 회사 목록

    Examples:
        >>> qsb = zz.quant.QuantSlackBotFDR(symbols, token, channel)
    """

    def __init__(
        self,
        symbols: List[str],
        token: str,
        channel: str,
        start_day: Optional[str] = "",
        top: Optional[int] = 1,
        name: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        mp_num: Optional[int] = 0,
        analysis: Optional[bool] = False,
    ) -> None:
        QuantSlackBot.__init__(
            self,
            symbols,
            token,
            channel,
            start_day,
            top,
            name,
            icon_emoji,
            mp_num,
            analysis,
        )
        self.krx = fdr.StockListing("KRX")

    def _get_data(self, symbol: str) -> Tuple[str, pd.core.frame.DataFrame]:
        title = self.krx[self.krx["Code"] == symbol].iloc[0, 2]
        data = fdr.DataReader(symbol, self.start_day)
        return title, data
