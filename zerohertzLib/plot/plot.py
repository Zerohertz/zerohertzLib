from typing import Dict, List, Optional, Tuple, Union

import seaborn as sns
from matplotlib import pyplot as plt


def plot(
    x: List[Union[int, float]],
    y: Dict[str, List[Union[int, float]]],
    xlab: Optional[str] = "x축 [단위]",
    ylab: Optional[str] = "y축 [단위]",
    title: Optional[str] = "tmp",
    ratio: Optional[Tuple[int]] = (15, 10),
    dpi: Optional[int] = 300,
) -> None:
    """List와 Dictionary로 입력받은 데이터를 line chart로 시각화

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/280603766-22a0f42c-91b0-4f34-aa73-29de6fdbd4e9.png
        :alt: Visualzation Result
        :align: center

    Args:
        x (``List[Union[int, float]]``): 입력 데이터 (X축)
        y (``Dict[str, List[Union[int, float]]]``): 입력 데이터 (Y축)
        xlab (``Optional[str]``): Graph에 출력될 X축 label
        ylab (``Optional[str]``): Graph에 출력될 Y축 label
        title (``Optional[str]``): Graph에 표시될 제목 및 파일명
        ratio (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        ``None``: 현재 directory에 바로 graph 저장

    Examples:
        >>> zz.plot.plot([i for i in range(20)],{"테란": list(np.random.rand(20) * 10), "저그": list(np.random.rand(20) * 10 + 1), "프로토스": list(np.random.rand(20) * 10 + 2)}, xlab="시간 [초]", ylab="성적 [점]", title="Star Craft")
    """
    colors = sns.color_palette("husl", n_colors=len(y))
    plt.figure(figsize=ratio)
    # list(plt.Line2D.lineStyles.keys())
    linestyle = ["-", "--", "-.", ":"]
    # import matplotlib.markers as mmarkers
    # markers = list(mmarkers.MarkerStyle.markers.keys())
    marker = ["o", "v", "^", "s", "p", "*", "x"]
    for i, (k, v) in enumerate(y.items()):
        plt.plot(
            x,
            v,
            color=colors[i],
            linestyle=linestyle[i % len(linestyle)],
            linewidth=2,
            marker=marker[i],
            markersize=12,
            label=k,
        )
    plt.grid(zorder=0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title, fontsize=25)
    plt.legend()
    plt.savefig(
        f"{title.lower().replace(' ', '_')}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close("all")
