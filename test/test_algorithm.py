import zerohertzLib as zz


# FROM: https://www.acmicpc.net/problem/1260
def test_dfs():
    assert zz.algorithm.dfs([[], [2, 3, 4], [1, 4], [1, 4], [1, 2, 3]], 1) == [
        1,
        2,
        4,
        3,
    ]
    assert zz.algorithm.dfs([[], [2, 3], [1, 5], [1, 4], [3, 5], [2, 4]], 3) == [
        3,
        1,
        2,
        5,
        4,
    ]


# FROM: https://www.acmicpc.net/problem/1260
def test_bfs():
    assert zz.algorithm.bfs([[], [2, 3, 4], [1, 4], [1, 4], [1, 2, 3]], 1) == [
        1,
        2,
        3,
        4,
    ]
    assert zz.algorithm.bfs([[], [2, 3], [1, 5], [1, 4], [3, 5], [2, 4]], 3) == [
        3,
        1,
        4,
        2,
        5,
    ]


def test_SoE():
    assert zz.algorithm.SoE(10) == [2, 3, 5, 7]


def test_fft():
    assert zz.algorithm.fft([1, 0, 0, 0], 0) == [(1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j)]
    assert zz.algorithm.fft([1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j], 1) == [
        (4 + 0j),
        0j,
        0j,
        0j,
    ]
