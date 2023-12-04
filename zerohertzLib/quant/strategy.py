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

import numpy as np
import pandas as pd


def moving_average(
    data: pd.core.frame.DataFrame,
    short_window: Optional[int] = 15,
    long_window: Optional[int] = 50,
    ohlc: Optional[str] = "Open",
) -> pd.core.frame.DataFrame:
    """주어진 ``data`` 의 단기 및 장기 이동 평균을 계산 후 trading signal과 postion을 결정하는 함수

    Args:
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        short_window (``Optional[int]``: 단기 이동 평균을 계산하기 위한 window 크기
        long_window (``Optional[int]``): 장기 이동 평균을 계산하기 위한 widnow 크기
        ohlc (``Optional[str]``): 이동 평균을 계산할 때 사용할 ``data`` 의 column 이름

    Returns:
        ``pd.core.frame.DataFrame``: 각 날짜에 대한 signal (``"signals"``) 및 position (``"positions"``) 정보

    Examples:
        >>> zz.quant.moving_average(data)
                    signals    short_mavg     long_mavg  positions
        2022-12-05      0.0  60900.000000  60900.000000          0
        ...             ...           ...           ...        ...
        2023-12-04     -1.0  72226.666667  69790.000000          0
        [249 rows x 4 columns]
    """
    signals = pd.DataFrame(index=data.index)
    signals["signals"] = 0.0
    signals["short_mavg"] = (
        data[ohlc].rolling(window=short_window, min_periods=1).mean()
    )
    signals["long_mavg"] = data[ohlc].rolling(window=long_window, min_periods=1).mean()
    signals["signals"][short_window:] = np.where(
        signals["short_mavg"][short_window:] > signals["long_mavg"][short_window:],
        -1.0,
        1.0,
    )
    signals["positions"] = signals["signals"].diff()
    signals["positions"] = np.where(
        signals["positions"] < 0, -1, np.where(signals["positions"] > 0, 1, 0)
    )
    return signals
