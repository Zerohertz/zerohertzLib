from typing import Dict, List, Optional, Tuple, Union

import seaborn as sns
from matplotlib import pyplot as plt


def scatter(
    data: Dict[str, List[List[Union[int, float]]]],
    size: Optional[float] = 36,
    xlab: Optional[str] = "x축 [단위]",
    ylab: Optional[str] = "y축 [단위]",
    title: Optional[str] = "tmp",
    ratio: Optional[Tuple[int]] = (15, 10),
    dpi: Optional[int] = 300,
) -> None:
    """Dictionary로 입력받은 데이터를 scatter plot으로 시각화

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/282639459-efca04cc-3c4a-42c5-b07d-e64705a5f791.png
        :alt: Visualzation Result
        :align: center

    Args:
        data (``Dict[str, List[List[Union[int, float]]]]``): 입력 데이터
        size (``Optional[int]``): Graph에 출력될 marker의 크기
        xlab (``Optional[str]``): Graph에 출력될 X축 label
        ylab (``Optional[str]``): Graph에 출력될 Y축 label
        title (``Optional[str]``): Graph에 표시될 제목 및 파일명
        ratio (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)

    Returns:
        ``None``: 현재 directory에 바로 graph 저장

    Examples:
        >>> zz.plot.scatter({"테란": [list(np.random.rand(200) * 10), list(np.random.rand(200) * 10)], "저그": [list(np.random.rand(200) * 5 - 1), list(np.random.rand(200) * 5 + 1)], "프로토스": [list(np.random.rand(200) * 10 + 3), list(np.random.rand(200) * 10 - 2)]}, size=400, xlab="비용 [미네랄]", ylab="전투력 [점]", title="Star Craft")
    """
    colors = sns.color_palette("husl", n_colors=len(data))
    plt.figure(figsize=ratio)
    # import matplotlib.markers as mmarkers
    # markers = list(mmarkers.MarkerStyle.markers.keys())
    marker = ["o", "v", "^", "s", "p", "*", "x"]
    for i, (k, v) in enumerate(data.items()):
        plt.scatter(
            v[0], v[1], s=size, color=colors[i], marker=marker[i], label=k, zorder=2
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
