"""
.. admonition:: Quant
    :class: hint

    ``KoreaInvestment`` 를 통해 수집한 data를 이용해 매수, 매도 signal을 포착하고 test하는 quant 함수 및 class들

.. important::

    ``signals["signals"]`` 부호의 의미

    - ``+1``: 매수
    - ``-1``: 매도
"""

from zerohertzLib.quant.backtest import Experiments, backtest, experiments
from zerohertzLib.quant.quant import (
    Balance,
    Quant,
    QuantSlackBot,
    QuantSlackBotFDR,
    QuantSlackBotKI,
)
from zerohertzLib.quant.strategies import bollinger_bands, momentum, moving_average, rsi

__all__ = [
    "moving_average",
    "backtest",
    "experiments",
    "rsi",
    "bollinger_bands",
    "momentum",
    "Experiments",
    "Quant",
    "QuantSlackBot",
    "QuantSlackBotKI",
    "Balance",
    "QuantSlackBotFDR",
]
