# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

"""
!!! hint "Quant"
    `FinanceDataReader` 및 `KoreaInvestment` 를 통해 수집한 data를 이용해 매수, 매도 signal을 포착하고 test하는 quant function 및 class들

!!! important
    `signals["signals"]` 부호의 의미

    - `+1`: 매수
    - `-1`: 매도
"""

from zerohertzLib.quant.backtest import Experiments, backtest, experiments
from zerohertzLib.quant.koreainvestment import Balance, QuantBotKI
from zerohertzLib.quant.methods import (
    bollinger_bands,
    macd,
    momentum,
    moving_average,
    rsi,
)
from zerohertzLib.quant.quant import Quant, QuantBot, QuantBotFDR

__all__ = [
    "Balance",
    "Experiments",
    "Quant",
    "QuantBot",
    "QuantBotFDR",
    "QuantBotKI",
    "backtest",
    "bollinger_bands",
    "experiments",
    "macd",
    "momentum",
    "moving_average",
    "rsi",
]
