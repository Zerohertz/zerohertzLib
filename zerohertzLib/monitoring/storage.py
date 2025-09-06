# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import os

from zerohertzLib.plot import pie


def _get_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for file_name in os.listdir(path):
        filepath = os.path.join(path, file_name)
        if os.path.isfile(filepath):
            total += os.path.getsize(filepath)
        elif os.path.isdir(filepath):
            total += _get_size(filepath)
    return total


def storage(path: str = ".", threshold: int = 1) -> str:
    """지정한 경로에 존재하는 file에 따른 용량을 pie graph로 시각화

    Args:
        path: 용량을 시각화할 경로
        threshold: Etc.로 분류될 임계값 (단위: %)

    Returns:
        저장된 graph의 절대 경로

    Examples:
        >>> zz.monitoring.storage(".")

        ![Storage monitoring example](../../../assets/monitoring/storage.png){ width="600" }
    """
    sizes = {}
    for file_name in os.listdir(path):
        filepath = os.path.join(path, file_name)
        sizes[file_name] = _get_size(filepath) / (1024**3)
    total_size = sum(sizes.values())
    etc = 0
    pops = []
    for key, value in sizes.items():
        if value / total_size * 100 <= threshold:
            etc += value
            pops.append(key)
    for pop in pops:
        sizes.pop(pop)
    sizes["Etc."] = etc
    data = dict(sorted(sizes.items(), key=lambda item: item[1], reverse=True))
    return pie(data, "GB", os.path.abspath(path).split("/")[-1])
