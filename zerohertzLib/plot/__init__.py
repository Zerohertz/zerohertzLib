import matplotlib.font_manager as font_manager
from matplotlib import pyplot as plt

from zerohertzLib.plot.bar import bar, barh, hist
from zerohertzLib.plot.plot import plot

plt.rcParams["font.size"] = 20
font_manager.fontManager.addfont(
    f"{__file__.replace('__init__.py', '')}NotoSansKR-Medium.ttf"
)
plt.rcParams["font.family"] = "Noto Sans KR"

__all__ = ["bar", "hist", "plot"]
