from typing import List


def SoE(N: int) -> List[int]:
    """Sieve of Eratosthenes

    Args:
        N (``int``): 구하고자 하는 소수 범위의 최댓값

    Returns:
        ``List[int]``: N까지 존재하는 소수 list

    Examples:
        >>> zz.algorithm.SoE(10)
        [2, 3, 5, 7]
    """
    B = [False, False] + [True] * (N - 1)
    PN = []
    for i in range(2, N + 1):
        if B[i]:
            PN.append(i)
            for j in range(2 * i, N + 1, i):
                B[j] = False
    return PN
