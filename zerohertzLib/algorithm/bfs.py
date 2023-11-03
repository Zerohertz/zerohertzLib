from collections import deque
from typing import List


def bfs(maps: List[List[int]], start: int) -> List[int]:
    """BFS를 수행하기 위한 함수

    Args:
        maps (``List[List[int]]``): 입력 그래프
        start (``int``): 그래프의 시작 지점

    Returns:
        ``List[int]``: 방문 순서

    Examples:
        >>> import zerohertzLib as zz
        >>> zz.algorithm.bfs([[], [2, 3, 4], [1, 4], [1, 4], [1, 2, 3]], 1)
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
