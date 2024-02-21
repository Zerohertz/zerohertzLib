"""
.. admonition:: Plot
    :class: hint

    다양한 변수들을 최대한 지정하지 않고 사용할 수 있는 시각화 함수들
"""

import os
from typing import Optional

from matplotlib import font_manager
from matplotlib import pyplot as plt

from zerohertzLib.plot.bar_chart import barh, barv, hist
from zerohertzLib.plot.pie import pie
from zerohertzLib.plot.plot import candle, plot
from zerohertzLib.plot.scatter import scatter
from zerohertzLib.plot.table import table
from zerohertzLib.plot.util import color, figure, savefig

FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts")


def font(kor: Optional[bool] = False, size: Optional[int] = 20) -> None:
    """``plot`` submodule 내 사용될 font 및 크기 설정

    Args:
        kor (``Optional[bool]``): 한국어 여부
        size (``Optional[int]``): Font의 크기

    Returns:
        ``None``: ``plt.rcParams`` 을 통한 전역적 설정
    """
    plt.rcParams["font.size"] = size
    # font_path = os.path.join(os.path.dirname(__file__), "fonts")
    if kor:
        font_manager.fontManager.addfont(
            os.path.join(FONT_PATH, "NotoSerifKR-Medium.otf")
        )
        plt.rcParams["font.family"] = "Noto Serif KR"
    else:
        font_manager.fontManager.addfont(os.path.join(FONT_PATH, "times.ttf"))
        plt.rcParams["font.family"] = "Times New Roman"


font()

__all__ = [
    "barv",
    "barh",
    "hist",
    "plot",
    "pie",
    "scatter",
    "color",
    "table",
    "savefig",
    "figure",
    "candle",
    "font",
]
