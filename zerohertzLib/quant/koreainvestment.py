# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import copy
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, ItemsView

import FinanceDataReader as fdr
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker

from zerohertzLib.api import KoreaInvestment
from zerohertzLib.plot import barv, figure, pie, savefig, table

from .quant import QuantBot
from .util import _cash2str


class Balance(KoreaInvestment):
    """한국투자증권의 국내 계좌 정보 조회 class

    Args:
        account_no: API 호출 시 사용할 계좌 번호
        path: `secret.key` 혹은 `token.dat` 이 포함된 경로
        kor: 국내 여부

    Attributes:
        balance: 현재 보유 주식과 계좌의 금액 정보

    Examples:
        `kor=True`:
            ```python
            >>> balance = zz.quant.Balance("00000000-00")
            >>> "LG전자" in balance
            True
            >>> "삼성전자" in balance
            False
            >>> len(balance)
            1
            >>> balance[0]
            ['066570', 102200.0, 100200, 1, -1.95, -2000]
            >>> balance()
            000
            ```
        `kor=False`:
            ```python
            >>> balance = zz.quant.Balance("00000000-00", kor=False)
            >>> "아마존닷컴" in balance
            True
            >>> "삼성전자" in balance
            False
            >>> len(balance)
            1
            >>> balance[0]
            ['META', 488.74, 510.92, 1, 4.53, 22.18]
            >>> balance()
            000.000
            ```
    """

    def __init__(self, account_no: str, path: str = "./", kor: bool = True) -> None:
        super().__init__(account_no, path)
        self.balance = {"stock": defaultdict(list)}
        self.kor = kor
        response = self.get_balance(kor)
        if self.kor:
            for stock in response["output1"]:
                if int(stock["hldg_qty"]) > 0:  # 보유수량
                    self.balance["stock"][stock["prdt_name"]] = [
                        stock["pdno"],  # 종목번호
                        float(
                            stock["pchs_avg_pric"]
                        ),  # 매입평균가격 (매입금액 / 보유수량)
                        int(stock["prpr"]),  # 현재가
                        int(stock["hldg_qty"]),  # 보유수량
                        float(stock["evlu_pfls_rt"]),  # 평가손익율
                        int(
                            stock["evlu_pfls_amt"]
                        ),  # 평가손익금액 (평가금액 - 매입금액)
                    ]
            self.balance["cash"] = int(response["output2"][0]["nass_amt"])  # 순자산금액
        else:
            for stock in response["output1"]:
                if int(float(stock["ccld_qty_smtl1"])) > 0:  # 체결수량합계
                    self.balance["stock"][stock["prdt_name"]] = [
                        stock["pdno"],  # 종목번호
                        float(stock["avg_unpr3"]),  # 평균단가
                        float(stock["ovrs_now_pric1"]),  # 해외현재가격
                        int(float(stock["ccld_qty_smtl1"])),  # 해외잔고수량
                        float(stock["evlu_pfls_rt1"]),  # 평가손익율
                        float(stock["evlu_pfls_amt2"]),  # 평가손익금액
                    ]
            self.balance["cash"] = (
                float(response["output3"]["evlu_amt_smtl_amt"])  # 평가금액합계금액
                + float(response["output3"]["frcr_use_psbl_amt"])  # 외화사용가능금액
                + float(response["output3"]["ustl_sll_amt_smtl"])  # 미결제매도금액합계
                - float(response["output3"]["ustl_buy_amt_smtl"])  # 미결제매수금액합계
            ) / self._exchange()
        self.balance["stock"] = dict(
            sorted(
                self.balance["stock"].items(),
                key=lambda item: item[1][1] * item[1][3],
                reverse=True,
            )
        )
        self.symbols = list(self.balance["stock"].keys())

    def __contains__(self, item: Any) -> bool:
        """
        Args:
            item: 보유 여부를 판단할 종목명

        Returns:
            입력 종목명의 보유 여부
        """
        return item in self.balance["stock"]

    def __len__(self) -> int:
        """
        Returns:
            보유 주식 종류의 수
        """
        return len(self.balance["stock"])

    def __getitem__(self, idx: int) -> list[int | float | str]:
        """
        Args:
            idx: Index

        Returns:
            Index에 따른 주식의 매수 시점과 현재의 정보
        """
        return self.balance["stock"][self.symbols[idx]]

    def __call__(self) -> int:
        """
        Returns:
            현재 보유 금액
        """
        return self.balance["cash"]

    def _exchange(self) -> float:
        """USD/KRW의 현재 시세

        Returns:
            USD/KRW의 현재 시세
        """
        now = datetime.now()
        data = fdr.DataReader("USD/KRW", now - timedelta(days=10))
        return float(data.Close[-1])

    def merge(self, balance: "Balance") -> None:
        """현재 계좌와 입력 계좌의 정보를 병합하는 function

        Args:
            balance: 병합될 계좌 정보

        Returns:
            현재 계좌에 정보 update

        Examples:
            >>> balance_1.merge(balance_2)
        """
        merged_balance = copy.deepcopy(balance.balance)
        if self.kor != balance.kor:
            exchange = self._exchange()
            if not self.kor:
                exchange = 1 / exchange
            for key, value in balance.items():
                merged_balance["stock"][key][1] = value[1] * exchange
                merged_balance["stock"][key][2] = value[2] * exchange
                merged_balance["stock"][key][-1] = value[-1] * exchange
            merged_balance["cash"] = balance.balance["cash"] * exchange
        for key, value in merged_balance["stock"].items():
            if key in self:
                (
                    _merged_code,
                    _merged_buy_price,
                    _merged_present_price,
                    _merged_cnt,
                    _merged_pandl_per,
                    _merged_pandl_abs,
                ) = self.balance["stock"][key]
                (
                    _tmp_code,
                    _tmp_buy_price,
                    _tmp_present_price,
                    _tmp_cnt,
                    _tmp_pandl_per,
                    _tmp_pandl_abs,
                ) = value
                assert _merged_code == _tmp_code
                _merged_buy_price = (
                    _merged_buy_price * _merged_cnt + _tmp_buy_price * _tmp_cnt
                ) / (_merged_cnt + _tmp_cnt)
                _merged_present_price = (_merged_present_price + _tmp_present_price) / 2
                _merged_cnt += _tmp_cnt
                _merged_pandl_abs = (
                    _merged_present_price - _merged_buy_price
                ) * _merged_cnt
                _merged_pandl_per = (
                    (_merged_present_price - _merged_buy_price)
                    / _merged_buy_price
                    * 100
                )
                self.balance["stock"][key] = [
                    _merged_code,
                    _merged_buy_price,
                    _merged_present_price,
                    _merged_cnt,
                    _merged_pandl_per,
                    _merged_pandl_abs,
                ]
            else:
                self.symbols.append(key)
                self.balance["stock"][key] = value
        self.balance["cash"] += merged_balance["cash"]
        self.balance["stock"] = dict(
            sorted(
                self.balance["stock"].items(),
                key=lambda item: item[1][1] * item[1][3],
                reverse=True,
            )
        )
        self.symbols = list(self.balance["stock"].keys())

    def items(self) -> ItemsView[str, list[int | float | str]]:
        """보유 주식의 반복문 사용을 위한 method

        Returns:
            보유 종목 code와 그에 따른 정보들

        Examples:
            >>> for k, v in balance.items():
            >>>     print(k, v)
        """
        return self.balance["stock"].items()

    def bought_symbols(self) -> list[str]:
        """보유 주식의 종목 code return

        Returns:
            보유 주식의 종목 code들

        Examples:
            >>> balance.bought_symbols():
            ['066570']
        """
        return [value[0] for _, value in self.items()]

    def table(self) -> str:
        """현재 계좌의 상태를 image로 저장

        Returns:
            저장된 image의 절대 경로

        Examples:
            >>> balance.table()
        """
        if self() == 0:
            return None
        if self.kor:
            col = [
                "Purchase Price [₩]",
                "Current Price [₩]",
                "Quantity",
                "Profit and Loss (P&L) [%]",
                "Profit and Loss (P&L) [₩]",
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
            _, purchase, current, quantity, pandl_per, pandl_abs = value
            row.append(name)
            data.append(
                [
                    _cash2str(purchase, self.kor),
                    _cash2str(current, self.kor),
                    quantity,
                    f"{pandl_per:.2f}%",
                    _cash2str(pandl_abs, self.kor),
                ]
            )
            purchase_total += purchase * quantity
            current_total += current * quantity
        row.append("TOTAL")
        if purchase_total == 0:
            pandl_total = 0
        else:
            pandl_total = (current_total - purchase_total) / purchase_total * 100
        data.append(
            [
                _cash2str(purchase_total, self.kor),
                _cash2str(current_total, self.kor),
                "-",
                f"{pandl_total:.2f}%",
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

    def pie(self) -> str:
        """현재 보유 종목을 pie chart로 시각화

        Returns:
            저장된 graph의 절대 경로

        Examples:
            >>> balance.pie()
        """
        if self() == 0:
            return None
        if self.kor:
            dim = "₩"
        else:
            dim = "$"
        data = defaultdict(float)
        data["Cash"] = 0
        for name, value in self.items():
            _, purchase, _, quantity, _, _ = value
            data[name] = purchase * quantity
        cash = self() - sum(data.values())
        data["Cash"] = max(data["Cash"], cash)
        return pie(data, dim, title="Portfolio", dpi=100, int_label=self.kor)

    def barv(self) -> str:
        """현재 보유 종목의 이익과 손실을 bar chart로 시각화

        Returns:
            저장된 graph의 절대 경로

        Examples:
            >>> balance.barv()
        """
        if self.kor:
            dim = "₩"
        else:
            dim = "$"
        data = {}
        for value in self:
            data[value[0]] = value[5]
        figure((30, 10))
        barv(
            data,
            xlab="",
            ylab=f"Profit and Loss (P&L) [{dim}]",
            title="",
            dim="",
            dimsize=16,
        )
        plt.gca().yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: format(int(x), ","))
        )
        return savefig("ProfitLoss", 100)


class QuantBotKI(Balance, QuantBot):
    """한국투자증권 API를 기반으로 입력된 여러 종목에 대해 매수, 매도 signal을 판단하고 Bot을 통해 message와 graph를 전송하는 class

    Args:
        account_no: API 호출 시 사용할 계좌 번호
        symbols: 종목 code들
        start_day: 조회 시작 일자 (`YYYYMMDD`)
        ohlc: 사용할 `data` 의 column 이름
        top: Experiment 과정에서 사용할 각 전략별 수
        methods: 사용할 전략들의 function명 및 parameters
        report: Experiment 결과 출력 여부
        token: Bot의 token (`xoxb-` prefix로 시작하면 `SlackBot`, 아니면 `DiscordBot`)
        channel: Bot이 전송할 channel
        name: Bot의 표시될 이름
        icon_emoji: Bot의 표시될 사진 (emoji)
        mp_num: 병렬 처리에 사용될 process의 수 (`0`: 직렬 처리)
        analysis: 각 전략의 보고서 전송 여부
        kor: 국내 여부
        path: `secret.key` 혹은 `token.dat` 이 포함된 경로

    Attributes:
        exps: 각 전략에 따른 parameter 분포

    Examples:
        >>> qbki = zz.quant.QuantBotKI("00000000-00", token=token, channel=channel)
    """

    def __init__(
        self,
        account_no: str,
        symbols: list[str] | None = None,
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
        path: str = "./",
    ) -> None:
        Balance.__init__(self, account_no, path, kor)
        if symbols is None:
            symbols = []
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
        self.symbols_bought = self.bought_symbols()

    def _get_data(self, symbol: str) -> tuple[str, pd.DataFrame]:
        response = self.get_ohlcv(symbol, start_day=self.start_day, kor=self.kor)
        title, data = self.response2ohlcv(response)
        return title, data

    def sell(self) -> None:
        """매도 signals 탐색

        한국투자증권의 잔고와 주식 보유 상황을 image로 변환하여 slack으로 전송 및 보유 중인 주식에 대해 매도 signals 탐색
        """
        path_balance, path_portfolio = self.table(), self.pie()
        if path_balance is None:
            self.bot.message("Balance: NULL", True)
            return None
        response = self.bot.message("> :bank: Balance")
        thread_id = self.bot.get_thread_id(response)
        self.bot.file(path_balance, thread_id=thread_id)
        self.bot.file(path_portfolio, thread_id=thread_id)
        self._inference(self.symbols_bought, "Sell")
        return None
