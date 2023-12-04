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


from typing import Optional

import pandas as pd


def _rsi(data: pd.core.frame.DataFrame, window: int) -> float:
    """RSI (Relative Strength Index)를 계산하는 함수

    Args:
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        window (``int``): Window의 크기

    Returns:
        ``float``: RSI 값
    """
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.rolling(window=window).mean()
    avg_loss = down.abs().rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _bollinger_bands(
    data: pd.core.frame.DataFrame,
    window: Optional[int] = 20,
    num_std_dev: Optional[int] = 2,
) -> pd.core.frame.DataFrame:
    """Bollinger band 계산 함수

    Args:
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        window (``Optional[int]``): 이동 평균을 계산하기 위한 윈도우 크기. 기본값은 20.
        num_std_dev (``Optional[int]``): 표준편차의 배수. 기본값은 2.

    Returns:
        ``pd.core.frame.DataFrame``: Bollinger band
    """
    bands = pd.DataFrame(index=data.index)
    bands["middle_band"] = data.iloc[:, :4].mean(1).rolling(window=window).mean()
    std_dev = data.iloc[:, :4].mean(1).rolling(window=window).std()
    bands["upper_band"] = bands["middle_band"] + (std_dev * num_std_dev)
    bands["lower_band"] = bands["middle_band"] - (std_dev * num_std_dev)
    return bands
