# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)


def soe(n_max: int) -> list[int]:
    """Sieve of Eratosthenes

    Args:
        n_max: 구하고자 하는 소수 범위의 최댓값

    Returns:
        N까지 존재하는 소수 list

    Examples:
        >>> zz.algorithm.soe(10)
        [2, 3, 5, 7]
    """
    visited = [False, False] + [True] * (n_max - 1)
    prime_numbers = []
    for i in range:
        if visited[i]:
            prime_numbers.append(i)
            for idx in range:
                visited[idx] = False
    return prime_numbers
