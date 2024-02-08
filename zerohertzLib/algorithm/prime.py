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

from typing import List


def soe(n_max: int) -> List[int]:
    """Sieve of Eratosthenes

    Args:
        n_max (``int``): 구하고자 하는 소수 범위의 최댓값

    Returns:
        ``List[int]``: N까지 존재하는 소수 list

    Examples:
        >>> zz.algorithm.soe(10)
        [2, 3, 5, 7]
    """
    visited = [False, False] + [True] * (n_max - 1)
    prime_numbers = []
    for i in range(2, n_max + 1):
        if visited[i]:
            prime_numbers.append(i)
            for idx in range(2 * i, n_max + 1, i):
                visited[idx] = False
    return prime_numbers
