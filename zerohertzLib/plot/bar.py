from typing import Dict, Tuple

import seaborn as sns
from matplotlib import pyplot as plt


def bar(
    data: Dict[str, int],
    xlab: str = "변수",
    ylab: str = "빈도 [단위]",
    title: str = "tmp",
    ratio: Tuple[int] = (15, 10),
    dpi: int = 300,
    rot: int = 0,
    per: bool = True,
) -> None:
    """Dictionary로 입력받은 데이터를 bar graph로 시각화

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/280595386-1c930639-762a-47b7-9ae1-1babf789803c.png
        :alt: Visualzation Result
        :align: center

    Args:
        data (``Dict[str, int]``): 입력 데이터
        xlab (``str``): Graph에 출력될 X축 label
        ylab (``str``): Graph에 출력될 Y축 label
        title (``str``): Graph에 표시될 제목 및 파일명
        ratio (``Tuple[int]``): Graph의 가로, 세로 길이
        dpi: (``int``): Graph 저장 시 DPI (Dots Per Inch)
        rot: (``int``): X축의 눈금 회전 각도
        per: (``bool``): 각 bar 상단에 percentage 표시 여부

    Returns:
        ``None``: 현재 directory에 바로 graph 저장

    Examples:
        >>> import zerohertzLib as zz
        >>> zz.plot.bar({"테란": 27, "저그": 40, "프로토스": 30}, xlab="종족", ylab="인구 [명]", title="Star Craft")
    """
    colors = sns.color_palette("husl", n_colors=len(data))
    plt.figure(figsize=ratio)
    bars = plt.bar(
        data.keys(),
        data.values(),
        color=colors,
        zorder=2,
    )
    plt.grid(zorder=0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(rotation=rot)
    plt.ylim([0, 1.1 * max(list(data.values()))])
    plt.title(title, fontsize=25)
    if per:
        SUM = sum(list(data.values()))
        for bar in bars:
            height = bar.get_height()
            percentage = (height / SUM) * 100
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
            )
    plt.savefig(
        f"{title.lower().replace(' ', '_')}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close("all")
