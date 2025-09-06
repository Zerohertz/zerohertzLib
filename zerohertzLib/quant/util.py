# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import pandas as pd


def _rsi(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """RSI (Relative Strength Index)를 계산하는 function

    Args:
        data: OHLCV (Open, High, Low, Close, Volume) data
        window: Window의 크기

    Returns:
        RSI 값
    """
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.rolling(window=window).mean()
    avg_loss = down.abs().rolling(window=window).mean()
    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _bollinger_bands(
    data: pd.DataFrame,
    window: int = 20,
    num_std_dev: int = 2,
    ohlc: str = "",
) -> pd.DataFrame:
    """Bollinger band 계산 function

    Args:
        data: OHLCV (Open, High, Low, Close, Volume) data
        window: 이동 평균을 계산하기 위한 윈도우 크기. 기본값은 20.
        num_std_dev: 표준편차의 배수. 기본값은 2.
        ohlc: 이동 평균을 계산할 때 사용할 `data` 의 column 이름

    Returns:
        Bollinger band
    """
    bands = pd.DataFrame(index=data.index)
    if ohlc == "":
        price_data = data.iloc[:, :4].mean(1)
        bands["middle_band"] = price_data.rolling(window=window).mean()
        std_dev = price_data.rolling(window=window).std()
    else:
        if ohlc not in data.columns:
            raise KeyError(f"Column '{ohlc}' not found in data")
        bands["middle_band"] = data[ohlc].rolling(window=window).mean()
        std_dev = data[ohlc].rolling(window=window).std()
    bands["upper_band"] = bands["middle_band"] + (std_dev * num_std_dev)
    bands["lower_band"] = bands["middle_band"] - (std_dev * num_std_dev)
    return bands


def _cash2str(
    cash: int | float,
    kor: bool,
) -> str:
    if kor:
        if cash < 0:
            return f"-₩{abs(cash):,.0f}"
        return f"₩{cash:,.0f}"
    if cash < 0:
        return f"-${abs(cash):,.2f}"
    return f"${cash:,.2f}"


def _seconds_to_hms(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours == 0:
        if minutes == 0:
            return f"{seconds}s"
        return f"{minutes}m {seconds}s"
    return f"{hours}h {minutes}m {seconds}s"


def _method2str(method: str) -> str:
    if "_" in method:
        return " ".join(word.capitalize() for word in method.split("_"))
    if method == "momentum":
        return "Momentum"
    return method.upper()
