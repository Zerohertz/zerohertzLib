from typing import List


def dfs(maps: List[List[int]], start: int) -> List[int]:
    """DFS를 수행하기 위한 함수

    Args:
        maps (``List[List[int]]``): 입력 그래프
        start (``int``): 그래프의 시작 지점

    Returns:
        ``List[int]``: 방문 순서

    Examples:
        >>> import zerohertzLib as zz
        >>> zz.algorithm.dfs([[], [2, 3, 4], [1, 4], [1, 4], [1, 2, 3]], 1)
    """
    visit = [False for _ in range(len(maps))]
    results = []

    def DFS(start):
        visit[start] = True
        results.append(start)
        for i in maps[start]:
            if not visit[i]:
                DFS(i)

    DFS(start)
    return results
