import matplotlib.font_manager as font_manager
from matplotlib import pyplot as plt

from zerohertzLib.plot.bar import bar, barh, hist
from zerohertzLib.plot.pie import pie
from zerohertzLib.plot.plot import plot
from zerohertzLib.plot.scatter import scatter

plt.rcParams["font.size"] = 20
font_manager.fontManager.addfont(
    f"{__file__.replace('__init__.py', '')}NotoSansKR-Medium.ttf"
)
plt.rcParams["font.family"] = "Noto Sans KR"

__all__ = ["bar", "barh", "hist", "plot", "pie", "scatter"]
