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

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .util import _bollinger_bands, _rsi


def moving_average(
    data: Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]],
    short_window: Optional[int] = 10,
    long_window: Optional[int] = 50,
    gap: Optional[int] = 500,
    ohlc: Optional[str] = "",
) -> Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]:
    """단기 및 장기 이동 평균 기반 매수 및 매도 signal을 생성하는 함수

    Note:
        Moving Average

        - Definition: 일정 기간 동안 평균화하여 추세 파악 및 noise 감소

    - 매수 신호 (``+1``): 단기 이동 평균이 장기 이동 평균보다 높을 때 생성 (상승 추세)
    - 매도 신호 (``-1``): 단기 이동 평균이 장기 이동 평균보다 낮을 때 생성 (하락 추세)

    Args:
        data (``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``): OHLCV (Open, High, Low, Close, Volume) data
        short_window (``Optional[int]``): 단기 이동 평균을 계산하기 위한 window 크기
        long_window (``Optional[int]``): 장기 이동 평균을 계산하기 위한 widnow 크기
        gap (``Optional[int]``): 이동 평균 간의 threshold를 산정하기 위한 평균 주가의 분모
        ohlc (``Optional[str]``): 이동 평균을 계산할 때 사용할 ``data`` 의 column 이름

    Returns:
        ``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``: 각 날짜에 대한 signal (``"signals"``) 정보

    Examples:
        >>> zz.quant.moving_average(data)
                    short_mavg  long_mavg  signals
        2022-12-05    60575.00   60575.00      0.0
        ...                ...        ...      ...
        2023-12-05    72255.00   69838.50      1.0
        [248 rows x 3 columns]

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/289050247-8407baa6-abb9-490a-abc6-37500a05ff22.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    if isinstance(data, list):
        signals = []
        for data_ in data:
            signals.append(moving_average(data_, short_window, long_window, gap, ohlc))
        return signals
    signals = pd.DataFrame(index=data.index)
    if ohlc == "":
        signals["short_mavg"] = (
            data.iloc[:, :4].mean(1).rolling(window=short_window, min_periods=1).mean()
        )
        signals["long_mavg"] = (
            data.iloc[:, :4].mean(1).rolling(window=long_window, min_periods=1).mean()
        )
    else:
        signals["short_mavg"] = (
            data[ohlc].rolling(window=short_window, min_periods=1).mean()
        )
        signals["long_mavg"] = (
            data[ohlc].rolling(window=long_window, min_periods=1).mean()
        )
    gap = data.iloc[:, :4].mean().mean() / gap
    signals["signals"] = 0.0
    prev = 0
    for i in range(len(signals)):
        short = signals["short_mavg"].iloc[i]
        long = signals["long_mavg"].iloc[i]
        if short > long + gap:
            if prev == -1:
                signals.loc[signals.index[i], "signals"] = 1
            prev = 1
        elif short + gap < long:
            if prev == 1:
                signals.loc[signals.index[i], "signals"] = -1
            prev = -1
    return signals


def rsi(
    data: Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]],
    lower_bound: Optional[int] = 20,
    upper_bound: Optional[int] = 80,
    window: Optional[int] = 21,
    ohlc: Optional[str] = "",
) -> Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]:
    r"""RSI 기반 매수 및 매도 signal을 생성하는 함수

    Note:
        RSI (Relative Strength Index)

        - Definition
            - :math:`RS = \frac{Average\ Gain}{Average\ Loss}`
            - :math:`RSI = 100 - \frac{100}{1+RS}`

        - Mean
            - ``-1`` → ``0``: 과매수 상태에서 중립 상태로 변화 (매도 position 청산)
            - ``0`` → ``-1``: 과매수 상태로의 진입 (새로운 매도 position)
            - ``+1`` → ``0``: 과매도 상태에서 중립 상태로 변화 (매수 position 청산)
            - ``0`` → ``+1``: 과매도 상태로의 진입 (새로운 매수 position)

    - 매수 신호 (``+1``): RSI 값이 ``lower_bound`` 보다 낮을 때 생성 (과매도 상태)
    - 매도 신호 (``-1``): RSI 값이 ``upper_bound`` 보다 높을 때 생성 (과매수 상태)

    Args:
        data (``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``): OHLCV (Open, High, Low, Close, Volume) data
        lower_bound (``Optional[int]``): RSI 과매도 기준
        upper_bound (``Optional[int]``): RSI 과매수 기준
        window (``Optional[int]``): 이동 평균을 계산하기 위한 widnow 크기
        ohlc (``Optional[str]``): RSI를 계산할 때 사용할 ``data`` 의 column 이름

    Returns:
        ``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``: 각 날짜에 대한 signal (``"signals"``) 정보

    Examples:
        >>> zz.quant.rsi(data)
                          RSI  signals
        2022-12-05        NaN        0
        ...               ...      ...
        2023-12-05  58.563536        0
        [248 rows x 2 columns]

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/289050264-6993d6ac-de6b-47f8-9657-897e5e66d2fd.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    if isinstance(data, list):
        signals = []
        for data_ in data:
            signals.append(rsi(data_, lower_bound, upper_bound, window, ohlc))
        return signals
    signals = pd.DataFrame(index=data.index)
    if ohlc == "":
        signals["RSI"] = _rsi(data.iloc[:, :4].mean(1), window)
    else:
        signals["RSI"] = _rsi(data[ohlc], window)
    signals["signals"] = 0
    signals["signals"] = np.where(
        signals["RSI"] > upper_bound, -1, np.where(signals["RSI"] < lower_bound, 1, 0)
    )
    return signals


def bollinger_bands(
    data: Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]],
    window: Optional[int] = 14,
    num_std_dev: Optional[float] = 2,
    ohlc: Optional[str] = "",
) -> Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]:
    """Bollinger band 기반 매수 및 매도 signal을 생성하는 함수

    Note:
        Bollinger Band

        - Definition: 1980년대에 John Bollinger에 의해 개발된 가격 변동성 및 추세 파악 기법
        - Mean
            - Middle Band: 기본적인 중간 가격 추세
            - Upper Band: Middle band에서 일정 표준편차 위에 위치 (과도한 상승 추세나 고평가 상태)
            - Lower Band: Middle band에서 일정 표준편차 아래에 위치 (과도한 하락 추세나 저평가 상태)

    - 매수 신호 (``+1``): 주가가 하단 Bollinger band (``lower_band``) 아래로 감소할 때 생성 (과매도 상태)
    - 매도 신호 (``-1``): 주가가 상단 Bollinger band (``upper_band``) 위로 상승할 때 생성 (과매수 상태)

    Args:
        data (``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``): OHLCV (Open, High, Low, Close, Volume) data
        window (``Optional[int]``): 이동 평균을 계산하기 위한 widnow 크기
        num_std_dev (``Optional[float]``): 표준편차의 배수
        ohlc (``Optional[str]``): 이동 평균을 계산할 때 사용할 ``data`` 의 column 이름

    Returns:
        ``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``: 각 날짜에 대한 signal (``"signals"``) 정보

    Examples:
        >>> zz.quant.bollinger_bands(data)
                     middle_band    upper_band    lower_band  signals
        2022-12-05           NaN           NaN           NaN        0
        ...                  ...           ...           ...      ...
        2023-12-05  72371.428571  73228.137390  71514.719753        0
        [248 rows x 4 columns]

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/288449546-8eb91fb4-c5d1-4aab-ae5d-928a04f1c159.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    if isinstance(data, list):
        signals = []
        for data_ in data:
            signals.append(bollinger_bands(data_, window, num_std_dev, ohlc))
        return signals
    signals = _bollinger_bands(data, window, num_std_dev)
    signals["signals"] = 0
    if ohlc == "":
        signals["signals"] = np.where(
            data.iloc[:, :4].mean(1) < signals["lower_band"], 1, signals["signals"]
        )
        signals["signals"] = np.where(
            data.iloc[:, :4].mean(1) > signals["upper_band"], -1, signals["signals"]
        )
    else:
        signals["signals"] = np.where(
            data[ohlc] < signals["lower_band"], 1, signals["signals"]
        )
        signals["signals"] = np.where(
            data[ohlc] > signals["upper_band"], -1, signals["signals"]
        )
    return signals


def momentum(
    data: Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]],
    avg_window: Optional[int] = 5,
    mnt_window: Optional[int] = 5,
    gap: Optional[int] = 75,
    ohlc: Optional[str] = "",
) -> Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]:
    """Momentum 기반 매수 및 매도 signal을 생성하는 함수

    Note:
        Momentum

        - Definition: ``data[ohlc].diff(window)`` 를 통해 ``window`` 일 전 가격 사이의 차이 계산
        - Mean
            - 양의 momentum: 가격 상승
            - 음의 momentum: 가격 하락
            - Momentum의 크기: 추세의 강도

    - 매수 신호 (``+1``): 주가 momentum이 양수일 때 생성 (상승 추세)
    - 매도 신호 (``-1``): 주가 momentum이 음수일 때 생성 (하락 추세)

    Args:
        data (``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``): OHLCV (Open, High, Low, Close, Volume) data
        avg_window (``Optional[int]``): 이동 평균을 계산하기 위한 widnow 크기
        mnt_window (``Optional[int]``): Momentum을 계산하기 위한 widnow 크기
        gap (``Optional[int]``): Momentum의 threshold를 산정하기 위한 평균 주가의 분모
        ohlc (``Optional[str]``): Momentum을 계산할 때 사용할 ``data`` 의 column 이름

    Returns:
        ``Union[pd.core.frame.DataFrame, List[pd.core.frame.DataFrame]]``: 각 날짜에 대한 signal (``"signals"``) 정보

    Examples:
        >>> zz.quant.momentum(data)
                    momentum  signals
        2022-12-05       NaN      0.0
        ...              ...      ...
        2023-12-05     190.0      0.0
        [248 rows x 2 columns]

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/289050257-64e5b1d5-7518-4161-9d40-777f6201cd6b.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    if isinstance(data, list):
        signals = []
        for data_ in data:
            signals.append(momentum(data_, avg_window, mnt_window, gap, ohlc))
        return signals
    signals = pd.DataFrame(index=data.index)
    if ohlc == "":
        signals["momentum"] = (
            data.iloc[:, :4].mean(1).rolling(window=avg_window).mean().diff(mnt_window)
        )
    else:
        signals["momentum"] = (
            data[ohlc].rolling(window=avg_window).mean().diff(mnt_window)
        )
    if gap != 0:
        gap = data.iloc[:, :4].mean().mean() / gap
    signals["signals"] = 0.0
    for i in range(len(signals)):
        mnt = signals["momentum"].iloc[i]
        if mnt > gap:
            signals.loc[signals.index[i], "signals"] = 1
        elif mnt < -gap:
            signals.loc[signals.index[i], "signals"] = -1
    return signals
