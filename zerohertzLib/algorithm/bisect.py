"""
MIT License

Copyright (c) 2023 Hyogeun Oh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List, Union


def bisect_right(sorted_list: List[Union[int, float]], value: Union[int, float]) -> int:
    """Binary Search (right)

    Args:
        sorted_list (``List[Union[int, float]]``): 정렬된 list
        value (``Union[int, float]``): 찾고자하는 값

    Returns:
        ``int``: ``value`` 값이 ``sorted_list`` 에 삽입될 때 index

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


def bisect_left(sorted_list: List[Union[int, float]], value: Union[int, float]) -> int:
    """Binary Search (left)

    Args:
        sorted_list (``List[Union[int, float]]``): 정렬된 list
        value (``Union[int, float]``): 찾고자하는 값

    Returns:
        ``int``: ``value`` 값이 ``sorted_list`` 에 삽입될 때 index

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
