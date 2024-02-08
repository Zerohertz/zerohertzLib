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


def _rsi(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """RSI (Relative Strength Index)를 계산하는 함수

    Args:
        data (``pd.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        window (``int``): Window의 크기

    Returns:
        ``pd.DataFrame``: RSI 값
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
    data: pd.DataFrame,
    window: Optional[int] = 20,
    num_std_dev: Optional[int] = 2,
    ohlc: Optional[str] = "",
) -> pd.DataFrame:
    """Bollinger band 계산 함수

    Args:
        data (``pd.DataFrame``): OHLCV (Open, High, Low, Close, Volume) data
        window (``Optional[int]``): 이동 평균을 계산하기 위한 윈도우 크기. 기본값은 20.
        num_std_dev (``Optional[int]``): 표준편차의 배수. 기본값은 2.
        ohlc (``Optional[str]``): 이동 평균을 계산할 때 사용할 ``data`` 의 column 이름

    Returns:
        ``pd.DataFrame``: Bollinger band
    """
    bands = pd.DataFrame(index=data.index)
    if ohlc == "":
        bands["middle_band"] = data.iloc[:, :4].mean(1).rolling(window=window).mean()
        std_dev = data.iloc[:, :4].mean(1).rolling(window=window).std()
    else:
        bands["middle_band"] = data[ohlc].rolling(window=window).mean()
        std_dev = data[ohlc].rolling(window=window).std()
    bands["upper_band"] = bands["middle_band"] + (std_dev * num_std_dev)
    bands["lower_band"] = bands["middle_band"] - (std_dev * num_std_dev)
    return bands


def _cash2str(
    cash: str,
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
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours == 0:
        if minutes == 0:
            return f"{seconds}s"
        return f"{minutes}m {seconds}s"
    return f"{hours}h {minutes}m {seconds}s"


def _method2str(method: str) -> str:
    if "_" in method:
        methods = method.split("_")
        for idx, met in enumerate(methods):
            methods[idx] = met[0].upper() + met[1:]
        return " ".join(methods)
    if "momentum" == method:
        return "Momentum"
    return method.upper()
