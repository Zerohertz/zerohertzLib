from typing import Dict, Optional, Tuple, Union

import seaborn as sns
from matplotlib import pyplot as plt


def pie(
    data: Dict[str, Union[int, float]],
    dim: Optional[str] = "",
    title: Optional[str] = "tmp",
    ratio: Optional[Tuple[int]] = (15, 10),
    dpi: Optional[int] = 300,
    int_label: Optional[bool] = True,
) -> None:
    """Dictionary로 입력받은 데이터를 pie chart로 시각화

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/282473748-bec83476-9ed6-4fe8-8f1a-2651344c1b7c.png
        :alt: Visualzation Result
        :align: center

    Args:
        data (``Dict[str, Union[int, float]]``): 입력 데이터
        dim: (``Optional[str]``): 입력 ``data`` 의 단위
        title (``Optional[str]``): Graph에 표시될 제목 및 파일명
        ratio (``Optional[Tuple[int]]``): Graph의 가로, 세로 길이
        dpi: (``Optional[int]``): Graph 저장 시 DPI (Dots Per Inch)
        int_label: (``Optional[bool]``): Label 내 수치의 소수점 표기 여부

    Returns:
        ``None``: 현재 directory에 바로 graph 저장

    Examples:
        >>> zz.plot.pie({"테란": 27, "저그": 40, "프로토스": 30}, dim="명", title="Star Craft")
    """
    colors = sns.color_palette("husl", n_colors=len(data))
    plt.figure(figsize=ratio)
    if int_label:
        if dim == "":
            labels = [f"{k} ({v:.0f})" for k, v in data.items()]
        else:
            labels = [f"{k} ({v:.0f} {dim})" for k, v in data.items()]
    else:
        if dim == "":
            labels = [f"{k} ({v:.2f})" for k, v in data.items()]
        else:
            labels = [f"{k} ({v:.2f} {dim})" for k, v in data.items()]
    plt.pie(
        data.values(),
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        radius=1,
        colors=colors,
        normalize=True,
    )
    plt.title(title, fontsize=25)
    plt.axis("equal")
    plt.savefig(
        f"{title.lower().replace(' ', '_')}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close("all")
