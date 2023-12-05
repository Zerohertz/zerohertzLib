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


from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd

from zerohertzLib.plot import candle

from .backtest import Experiments, backtest


class Quant(Experiments):
    """Full factorial design 기반의 backtest를 수행하고 최적의 전략을 융합하는 class

    Args:
        title(``str``): 종목 이름
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        ohlc (``Optional[str]``): 사용할 ``data`` 의 column 이름
        top (``Optional[int]``): Experiments 과정에서 사용할 각 전략별 수
        methods (``Optional[List[str]]``): 사용할 전략들의 함수명

    Attributes:
        backtest (``List[Tuple[Any]]``): 융합된 전략의 backtest 결과

    Examples:
        >>> qnt = zz.quant.Quant(title, data, top=3)
    """

    def __init__(
        self,
        title: str,
        data: pd.core.frame.DataFrame,
        ohlc: Optional[str] = "",
        top: Optional[int] = 1,
        methods: Optional[List[str]] = [
            "moving_average",
            "rsi",
            "bollinger_bands",
            "momentum",
        ],
    ) -> None:
        super().__init__(title, data, ohlc)
        self.data = data
        self.signals = pd.DataFrame(index=data.index)
        self.cnt = defaultdict(int)
        self.cnt_total = 0
        for method in methods:
            if hasattr(self, method):
                self.signals[method] = 0
                _, profits, signals = getattr(self, method)()
                for profit, signal in zip(profits[:top], signals[:top]):
                    if profit > 0:
                        self.signals[method] += signal["signals"]
                        self.cnt[method] += 1
                        self.cnt_total += 1
            else:
                raise AttributeError(f"'Quant' object has no attribute '{method}'")
        self.signals["signals"] = self.signals.sum(1)
        self.backtest = []
        for i in range(1, 11):
            results = backtest(
                self.data, self.signals, threshold=self.cnt_total * i / 10
            )
            self.backtest.append(
                (
                    i / 10 * 100,
                    results["profit"],
                    results["transaction"],
                )
            )
        self.backtest.sort(key=lambda x: x[1], reverse=True)
        candle(
            data,
            title,
            signals=self.signals.loc[:, ["signals"]],
            dpi=100,
            threshold=self.backtest[0][0] * self.cnt_total / 100,
        )

    def run(self, day: Optional[str] = -1) -> Dict[str, Any]:
        """입력된 날짜에 대해 분석 정보 return

        Args:
            day (``Optional[str]``): 분석할 날짜

        Returns:
            ``Dict[str, Any]``: 각 전략에 따른 분석 정보 및 결론

        Examples:
            >>> qnt.run()
            defaultdict(<class 'float'>, {'moving_average': 100.0, 'momentum': 100.0, 'total': 100.0, 'position': 'Buy'})
            >>> qnt.run("20231023")
            defaultdict(<class 'float'>, {'moving_average': 100.0, 'momentum': 0.0, 'total': 66.66666666666666, 'position': 'Buy'})
            >>> qnt.run("2023-09-22")
            defaultdict(<class 'float'>, {'moving_average': -100.0, 'momentum': 0.0, 'total': -66.66666666666666, 'position': 'Sell'})
        """
        if self.cnt_total <= 0:
            return {"total": 0}
        if day != -1 and "-" not in day:
            day = day[:4] + "-" + day[4:6] + "-" + day[6:8]
        possibility = defaultdict(float)
        for key, value in self.cnt.items():
            if value != 0:
                possibility[key] = self.signals[key][day] / value * 100
        possibility["total"] = self.signals["signals"][day] / self.cnt_total * 100
        if self.backtest[0][0] <= possibility["total"]:
            possibility["position"] = "Buy"
        elif self.backtest[0][0] >= possibility["total"]:
            possibility["position"] = "Sell"
        else:
            possibility["position"] = "None"
        return possibility
