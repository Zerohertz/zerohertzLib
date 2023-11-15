import os
from typing import Optional

import zerohertzLib as zz


def _get_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    else:
        total = 0
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath):
                total += os.path.getsize(filepath)
            elif os.path.isdir(filepath):
                total += _get_size(filepath)
        return total


def storage(path: str, threshold: Optional[int] = 1) -> None:
    """지정한 경로에 존재하는 파일에 따른 용량을 pie graph로 시각화

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/282481985-15ce10ff-e4b1-4b6a-84ea-6e948b684e0c.png
        :alt: Visualzation Result
        :align: center

    Args:
        path (``str``): 입력 데이터
        threshold: (``Optional[int]``): Etc.로 분류될 임계값 (단위: %)

    Returns:
        ``None``: 현재 directory에 바로 graph 저장

    Examples:
        >>> zz.monitoring.storage(".")
    """
    sizes = {}
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        sizes[filename] = _get_size(filepath) / (1024**3)
    total_size = sum(sizes.values())
    etc = 0
    pop = []
    for k, v in sizes.items():
        if v / total_size * 100 <= threshold:
            etc += v
            pop.append(k)
    for p in pop:
        sizes.pop(p)
    sizes["Etc."] = etc
    data = dict(sorted(sizes.items(), key=lambda item: item[1], reverse=True))
    zz.plot.pie(data, "GB", os.path.abspath(path).split("/")[-1])
