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

from .util import _bollinger_bands, _rsi


def moving_average(
    data: pd.core.frame.DataFrame,
    short_window: Optional[int] = 15,
    long_window: Optional[int] = 50,
    ohlc: Optional[str] = "Open",
) -> pd.core.frame.DataFrame:
    """주어진 ``data`` 의 단기 및 장기 이동 평균을 계산 후 trading signal과 postion을 결정하는 함수

    Args:
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        short_window (``Optional[int]``): 단기 이동 평균을 계산하기 위한 window 크기
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

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/287730379-368fa075-70b5-4721-91ba-05e7fc579d99.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
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


def rsi(
    data: pd.core.frame.DataFrame,
    lower_bound: Optional[int] = 30,
    upper_bound: Optional[int] = 70,
    window: Optional[int] = 14,
    ohlc: Optional[str] = "Open",
) -> pd.core.frame.DataFrame:
    """RSI 기반 매수 및 매도 signal을 생성하는 함수

    Args:
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        lower_bound (``Optional[int]``): RSI 과매도 기준
        upper_bound (``Optional[int]``): RSI 과매수 기준
        window (``Optional[int]``): 이동 평균을 계산하기 위한 widnow 크기
        ohlc (``Optional[str]``): 이동 평균을 계산할 때 사용할 ``data`` 의 column 이름

    Returns:
        ``pd.core.frame.DataFrame``: 각 날짜에 대한 signal (``"signals"``) 및 position (``"positions"``) 정보

    Examples:
        >>> zz.quant.rsi(data)
                          RSI  signals  positions
        2022-12-05        NaN        0        NaN
        ...               ...      ...        ...
        2023-12-04  60.975610        0        0.0
        [249 rows x 3 columns]

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/287730363-456bbf0e-62f9-45c8-8025-7fe22588d780.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    signals = pd.DataFrame(index=data.index)
    signals["RSI"] = _rsi(data[ohlc], window)
    signals["signals"] = 0
    signals["signals"] = np.where(
        signals["RSI"] > upper_bound, 1, np.where(signals["RSI"] < lower_bound, -1, 0)
    )
    signals["positions"] = signals["signals"].diff()
    return signals


def bollinger_bands(
    data: pd.core.frame.DataFrame,
    window: Optional[int] = 20,
    num_std_dev: Optional[int] = 2,
    ohlc: Optional[str] = "Open",
) -> pd.core.frame.DataFrame:
    """Bollinger band 기반 매수 및 매도 signal을 생성하는 함수

    Args:
        data (``pd.core.frame.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        window (``Optional[int]``): 이동 평균을 계산하기 위한 widnow 크기
        num_std_dev (``Optional[int]``): 표준편차의 배수
        ohlc (``Optional[str]``): 이동 평균을 계산할 때 사용할 ``data`` 의 column 이름

    Returns:
        ``pd.core.frame.DataFrame``: 각 날짜에 대한 signal (``"signals"``) 및 position (``"positions"``) 정보

    Examples:
        >>> zz.quant.bollinger_bands(data)
                    middle_band    upper_band    lower_band  signals  positions
        2022-12-05          NaN           NaN           NaN        0          0
        ...                 ...           ...           ...      ...        ...
        2023-12-04      71822.5  73714.581839  69930.418161        0          0

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/287743057-aec042ef-6c0b-41b7-a85e-7cc6046a06af.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    signals = _bollinger_bands(data, window, num_std_dev)
    signals["signals"] = 0
    signals["signals"] = np.where(
        data[ohlc] < signals["lower_band"], -1, signals["signals"]
    )
    signals["signals"] = np.where(
        data[ohlc] > signals["upper_band"], 1, signals["signals"]
    )
    signals["positions"] = 0
    previous_signal = 0
    for i in range(len(signals)):
        current_signal = signals["signals"].iloc[i]
        if current_signal not in (previous_signal, 0):
            signals.loc[signals.index[i], "positions"] = current_signal
        previous_signal = current_signal
    return signals
