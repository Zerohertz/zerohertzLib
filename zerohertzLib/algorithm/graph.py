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


import heapq
import sys
from collections import deque
from typing import List, Tuple


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


def floyd_warshall(graph: List[List[Tuple[int, int]]]) -> List[List[int]]:
    """Graph에서 모든 node 쌍 간의 최단 경로 거리 계산

    Note:
        Time Complexity: :math:`O(V^3)`

        - :math:`V`: Node의 수

    Args:
        graph (``List[List[Tuple[int, int]]]``): Index (간선의 시작 node)에 따른 간선의 도착 node와 가중치 정보

    Returns:
        ``List[int]``: 모든 node 쌍에 대한 최단 경로 거리

    Examples:
        >>> graph = [[(1, 4), (2, 2), (3, 7)], [(0, 1), (2, 5)], [(0, 2), (3, 4)], [(1, 3)]]
        >>> zz.algorithm.floyd_warshall(graph)
        [[0, 4, 2, 6], [1, 0, 3, 7], [2, 6, 0, 4], [4, 3, 6, 0]]
    """
    n = len(graph)
    distance = [[sys.maxsize for _ in range(n)] for _ in range(n)]
    for i in range(n):
        distance[i][i] = 0
        for j, dist in graph[i]:
            distance[i][j] = dist
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance[i][j] > distance[i][k] + distance[k][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
    return distance


def dijkstra(graph: List[List[Tuple[int, int]]], start: int) -> List[int]:
    """Graph에서 시작 node로부터 모든 다른 node까지의 최단 경로 거리 계산

    Note:
        Time Complexity: :math:`O((V+E)\log{V})`

        - :math:`V`: Node의 수
        - :math:`E`: 간선의 수

    Args:
        graph (``List[List[Tuple[int, int]]]``): Index (간선의 시작 node)에 따른 간선의 도착 node와 가중치 정보
        start (``int``): 최단 경로 거리가 계신될 시작 node

    Returns:
        ``List[int]``: ``start`` 에서 graph 내 모든 다른 node 까지의 최단 경로 거리

    Examples:
        >>> graph = [[(1, 4), (2, 2), (3, 7)], [(0, 1), (2, 5)], [(0, 2), (3, 4)], [(1, 3)]]
        >>> zz.algorithm.dijkstra(graph, 0)
        [0, 4, 2, 6]
        >>> zz.algorithm.dijkstra(graph, 1)
        [1, 0, 3, 7]
        >>> zz.algorithm.dijkstra(graph, 2)
        [2, 6, 0, 4]
        >>> zz.algorithm.dijkstra(graph, 3)
        [4, 3, 6, 0]
    """
    distance = [sys.maxsize for _ in range(len(graph))]
    distance[start] = 0
    queue = [(0, start)]
    while queue:
        dist, node = heapq.heappop(queue)
        if distance[node] < dist:
            continue
        for node_, dist_ in graph[node]:
            if dist + dist_ < distance[node_]:
                distance[node_] = dist + dist_
                heapq.heappush(queue, (dist + dist_, node_))
    return distance
