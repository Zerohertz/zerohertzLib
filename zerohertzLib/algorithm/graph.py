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

from collections import deque
from typing import List


def bfs(maps: List[List[int]], start: int) -> List[int]:
    """BFS를 수행하기 위한 함수

    Args:
        maps (``List[List[int]]``): 입력 graph
        start (``int``): Graph의 시작 지점

    Returns:
        ``List[int]``: 방문 순서

    Examples:
        >>> zz.algorithm.bfs([[], [2, 3, 4], [1, 4], [1, 4], [1, 2, 3]], 1)
        [1, 2, 3, 4]
    """
    visit = [False for _ in range(len(maps))]
    results = []
    queue = deque()
    queue.append(start)
    visit[start] = True
    while queue:
        tmp = queue.popleft()
        results.append(tmp)
        for i in maps[tmp]:
            if not visit[i]:
                visit[i] = True
                queue.append(i)
    return results


def dfs(maps: List[List[int]], start: int) -> List[int]:
    """DFS를 수행하기 위한 함수

    Args:
        maps (``List[List[int]]``): 입력 graph
        start (``int``): Graph의 시작 지점

    Returns:
        ``List[int]``: 방문 순서

    Examples:
        >>> zz.algorithm.dfs([[], [2, 3, 4], [1, 4], [1, 4], [1, 2, 3]], 1)
        [1, 2, 4, 3]
    """
    visit = [False for _ in range(len(maps))]
    results = []

    def _dfs(start):
        visit[start] = True
        results.append(start)
        for i in maps[start]:
            if not visit[i]:
                _dfs(i)

    _dfs(start)
    return results
