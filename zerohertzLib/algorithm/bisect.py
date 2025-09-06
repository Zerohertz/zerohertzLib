# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)


def bisect_right(sorted_list: list[int | float], value: int | float) -> int:
    """Binary Search (right)

    Args:
        sorted_list: 정렬된 list
        value: 찾고자하는 값

    Returns:
        `value` 값이 `sorted_list` 에 삽입될 때 index

    Examples:
        >>> zz.algorithm.bisect_right([1, 3, 5], 2.7)
        1
        >>> zz.algorithm.bisect_right([1, 3, 5], 3)
        2
        >>> zz.algorithm.bisect_right([1, 3, 5], 3.3)
        2
    """
    low, high = 0, len(sorted_list)
    while low < high:
        mid = (low + high) // 2
        if value < sorted_list[mid]:
            high = mid
        else:
            low = mid + 1
    return low


def bisect_left(sorted_list: list[int | float], value: int | float) -> int:
    """Binary Search (left)

    Args:
        sorted_list: 정렬된 list
        value: 찾고자하는 값

    Returns:
        `value` 값이 `sorted_list` 에 삽입될 때 index

    Examples:
        >>> zz.algorithm.bisect_left([1, 3, 5], 2.7)
        1
        >>> zz.algorithm.bisect_left([1, 3, 5], 3)
        1
        >>> zz.algorithm.bisect_left([1, 3, 5], 3.3)
        2
    """
    low, high = 0, len(sorted_list)
    while low < high:
        mid = (low + high) // 2
        if sorted_list[mid] < value:
            low = mid + 1
        else:
            high = mid
    return low
