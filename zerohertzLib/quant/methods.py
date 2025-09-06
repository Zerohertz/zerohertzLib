# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import numpy as np
import pandas as pd

from .util import _bollinger_bands, _rsi


def moving_average(
    data: pd.DataFrame,
    short_window: int = 40,
    long_window: int = 80,
    threshold: float = 0.0,
    ohlc: str = "",
) -> pd.DataFrame:
    """단기 및 장기 이동 평균 기반 매수 및 매도 signal을 생성하는 function

    Note:
        Moving Average

        - Definition: 일정 기간 동안 평균화하여 추세 파악 및 noise 감소

    - 매수 신호: 단기 이동 평균이 장기 이동 평균보다 높을 때 생성 (상승 추세)
    - 매도 신호: 단기 이동 평균이 장기 이동 평균보다 낮을 때 생성 (하락 추세)

    Args:
        data: OHLCV (Open, High, Low, Close, Volume) data
        short_window: 단기 이동 평균을 계산하기 위한 window 크기
        long_window: 장기 이동 평균을 계산하기 위한 widnow 크기
        threshold: 신호를 발생 시킬 임계값
        ohlc: 이동 평균을 계산할 때 사용할 `data` 의 column 이름

    Returns:
        각 날짜에 대한 signal (`"signals"`) 정보

    Examples:
        >>> zz.quant.moving_average(data)
                    short_mavg    long_mavg  signals
        Date
        2022-01-03  139375.000  139375.0000        0
        ...                ...          ...      ...
        2023-12-19  102450.000  102337.1875        0
        [485 rows x 3 columns]

        ![Moving average example](../../../assets/quant/moving_average.png){ width="500" }
    """
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
    feature = signals["short_mavg"] - signals["long_mavg"]
    threshold = feature.abs().mean() * threshold
    signals["signals"] = 0
    signals.loc[feature > threshold, "signals"] = 1
    signals.loc[feature < -threshold, "signals"] = -1
    buy_signals = (signals["signals"] == 1) & (signals["signals"].shift(1) < 1)
    sell_signals = (signals["signals"] == -1) & (signals["signals"].shift(1) > -1)
    signals["signals"] = 0
    signals.loc[buy_signals, "signals"] = 1
    signals.loc[sell_signals, "signals"] = -1
    return signals


def rsi(
    data: pd.DataFrame,
    lower_bound: int = 20,
    upper_bound: int = 80,
    window: int = 30,
    ohlc: str = "",
) -> pd.DataFrame:
    r"""RSI 기반 매수 및 매도 signal을 생성하는 function

    Note:
        RSI (Relative Strength Index)

        - Definition
            - $RS = \frac{Average\ Gain}{Average\ Loss}$
            - $RSI = 100 - \frac{100}{1+RS}$

        - Mean
            - `-1` → `0`: 과매수 상태에서 중립 상태로 변화 (매도 position 청산)
            - `0` → `-1`: 과매수 상태로의 진입 (새로운 매도 position)
            - `+1` → `0`: 과매도 상태에서 중립 상태로 변화 (매수 position 청산)
            - `0` → `+1`: 과매도 상태로의 진입 (새로운 매수 position)

    - 매수 신호: RSI 값이 `lower_bound` 보다 낮을 때 생성 (과매도 상태)
    - 매도 신호: RSI 값이 `upper_bound` 보다 높을 때 생성 (과매수 상태)

    Args:
        data: OHLCV (Open, High, Low, Close, Volume) data
        lower_bound: RSI 과매도 기준
        upper_bound: RSI 과매수 기준
        window: 이동 평균을 계산하기 위한 widnow 크기
        ohlc: RSI를 계산할 때 사용할 `data` 의 column 이름

    Returns:
        각 날짜에 대한 signal (`"signals"`) 정보

    Examples:
        >>> zz.quant.rsi(data)
                          RSI  signals
        Date
        2022-01-03        NaN        0
        ...               ...      ...
        2023-12-19  35.671343        0
        [485 rows x 2 columns]

        ![RSI example](../../../assets/quant/rsi.png){ width="500" }
    """
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
    data: pd.DataFrame,
    window: int = 60,
    num_std_dev: float = 2.5,
    ohlc: str = "",
) -> pd.DataFrame:
    """Bollinger band 기반 매수 및 매도 signal을 생성하는 function

    Note:
        Bollinger Band

        - Definition: 1980년대에 John Bollinger에 의해 개발된 가격 변동성 및 추세 파악 기법
        - Mean
            - Middle Band: 기본적인 중간 가격 추세
            - Upper Band: Middle band에서 일정 표준편차 위에 위치 (과도한 상승 추세나 고평가 상태)
            - Lower Band: Middle band에서 일정 표준편차 아래에 위치 (과도한 하락 추세나 저평가 상태)

    - 매수 신호: 주가가 하단 Bollinger band (`lower_band`) 아래로 감소할 때 생성 (과매도 상태)
    - 매도 신호: 주가가 상단 Bollinger band (`upper_band`) 위로 상승할 때 생성 (과매수 상태)

    Args:
        data: OHLCV (Open, High, Low, Close, Volume) data
        window: 이동 평균을 계산하기 위한 widnow 크기
        num_std_dev: 표준편차의 배수
        ohlc: 이동 평균을 계산할 때 사용할 `data` 의 column 이름

    Returns:
        각 날짜에 대한 signal (`"signals"`) 정보

    Examples:
        >>> zz.quant.bollinger_bands(data)
                      middle_band     upper_band    lower_band  signals
        Date
        2022-01-03            NaN            NaN           NaN        0
        ...                   ...            ...           ...      ...
        2023-12-19  102771.666667  111527.577705  94015.755629        0
        [485 rows x 4 columns]

        ![Bollinger bands example](../../../assets/quant/bollinger_bands.png){ width="500" }
    """
    signals = _bollinger_bands(data, window, num_std_dev, ohlc)
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
    data: pd.DataFrame,
    window: int = 5,
    ohlc: str = "",
) -> pd.DataFrame:
    """Momentum 기반 매수 및 매도 signal을 생성하는 function

    Note:
        Momentum

        - Definition: `data[ohlc].diff(window)` 를 통해 `window` 일 전 가격 사이의 차이 계산
        - Mean
            - 양의 momentum: 가격 상승
            - 음의 momentum: 가격 하락
            - Momentum의 크기: 추세의 강도

    - 매수 신호: 주가 momentum이 양수일 때 생성 (상승 추세)
    - 매도 신호: 주가 momentum이 음수일 때 생성 (하락 추세)

    Args:
        data: OHLCV (Open, High, Low, Close, Volume) data
        window: Momentum을 계산하기 위한 widnow 크기
        ohlc: Momentum을 계산할 때 사용할 `data` 의 column 이름

    Returns:
        각 날짜에 대한 signal (`"signals"`) 정보

    Examples:
        >>> zz.quant.momentum(data)
                    momentum  signals
        Date
        2022-01-03       NaN        0
        ...              ...      ...
        2023-12-19     550.0        0
        [485 rows x 2 columns]

        ![Momentum example](../../../assets/quant/momentum.png){ width="500" }
    """
    signals = pd.DataFrame(index=data.index)
    if ohlc == "":
        signals["momentum"] = data.iloc[:, :4].mean(1).diff(window)
    else:
        signals["momentum"] = data[ohlc].diff(window)
    buy_signals = (signals["momentum"] > 0) & (signals["momentum"].shift(1) < 0)
    sell_signals = (signals["momentum"] < 0) & (signals["momentum"].shift(1) > 0)
    signals["signals"] = 0
    signals.loc[buy_signals, "signals"] = 1
    signals.loc[sell_signals, "signals"] = -1
    return signals


def macd(
    data: pd.DataFrame,
    n_fast: int = 12,
    n_signal: int = 9,
    ohlc: str = "",
) -> pd.DataFrame:
    """MACD 기반 매수 및 매도 signal을 생성하는 function

    Note:
        MACD (Moving Average Convergence Divergence)

        - Definition: 빠른 EMA (`n_fast`)와 느린 EMA (`n_slow`)의 차이
            - `n_slow = n_fast * 2`
        - Mean
            - EMA: 최근 가격에 더 많은 가중치를 두어 계산하는 이동 평균
            - Signal line: MACD의 추세를 평활화하여 추세의 방향과 강도를 파악

    - 매수 신호: MACD가 signal line 위로 상승할 때 생성 (상승 추세)
    - 매도 신호: MACD가 signal line 아래로 하락할 때 생성 (하락 추세)

    Args:
        data: OHLCV (Open, High, Low, Close, Volume) data
        n_fast: 빠른 EMA 계산을 위한 기간
        n_signal: MACD signal line 계산을 위한 기간
        ohlc: Momentum을 계산할 때 사용할 `data` 의 column 이름

    Returns:
        각 날짜에 대한 signal (`"signals"`) 정보

    Examples:
        >>> zz.quant.macd(data)
                           MACD  signals
        Date
        2022-01-03     0.000000        0
        ...                 ...      ...
        2023-12-19 -1950.006134        0
        [485 rows x 2 columns]

        ![MACD example](../../../assets/quant/macd.png){ width="500" }
    """
    n_slow = n_fast * 2
    signals = pd.DataFrame(index=data.index)
    if ohlc == "":
        data_ = data.iloc[:, :4].mean(1)
    else:
        data_ = data[ohlc]
    ema_fast = data_.ewm(span=n_fast, adjust=False).mean()
    ema_slow = data_.ewm(span=n_slow, adjust=False).mean()
    signals["MACD"] = 0
    signals["MACD"] = ema_fast - ema_slow
    signal = signals["MACD"].ewm(span=n_signal, adjust=False).mean()
    macd_diff = signals["MACD"] - signal
    signals["signals"] = 0
    signals.loc[macd_diff > 0, "signals"] = 1
    signals.loc[macd_diff < 0, "signals"] = -1
    buy_signals = (signals["signals"] > 0) & (signals["signals"].shift(1) < 0)
    sell_signals = (signals["signals"] < 0) & (signals["signals"].shift(1) > 0)
    signals["signals"] = 0
    signals.loc[buy_signals, "signals"] = 1
    signals.loc[sell_signals, "signals"] = -1
    return signals
