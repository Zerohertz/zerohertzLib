"""
.. admonition:: Plot
    :class: hint

    다양한 변수들을 최대한 지정하지 않고 사용할 수 있는 시각화 함수들
"""


from matplotlib import font_manager
from matplotlib import pyplot as plt

from zerohertzLib.plot.bar_chart import barh, barv, hist
from zerohertzLib.plot.pie import pie
from zerohertzLib.plot.plot import plot
from zerohertzLib.plot.scatter import scatter
from zerohertzLib.plot.util import color

plt.rcParams["font.size"] = 20
font_manager.fontManager.addfont(
    __file__.replace("__init__.py", "NotoSansKR-Medium.ttf")
)
plt.rcParams["font.family"] = "Noto Sans KR"

__all__ = ["barv", "barh", "hist", "plot", "pie", "scatter", "color"]
