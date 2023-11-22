import zerohertzLib as zz


# FROM: https://www.acmicpc.net/problem/1260
def test_dfs_1():
    assert zz.algorithm.dfs([[], [2, 3, 4], [1, 4], [1, 4], [1, 2, 3]], 1) == [
        1,
        2,
        4,
        3,
    ]


def test_dfs_2():
    assert zz.algorithm.dfs([[], [2, 3], [1, 5], [1, 4], [3, 5], [2, 4]], 3) == [
        3,
        1,
        2,
        5,
        4,
    ]


# FROM: https://www.acmicpc.net/problem/1260
def test_bfs_1():
    assert zz.algorithm.bfs([[], [2, 3, 4], [1, 4], [1, 4], [1, 2, 3]], 1) == [
        1,
        2,
        3,
        4,
    ]


def test_bfs_2():
    assert zz.algorithm.bfs([[], [2, 3], [1, 5], [1, 4], [3, 5], [2, 4]], 3) == [
        3,
        1,
        4,
        2,
        5,
    ]


def test_SoE():
    assert zz.algorithm.soe(10) == [2, 3, 5, 7]


def test_fft():
    assert zz.algorithm.fft([1, 0, 0, 0]) == [(1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j)]


def test_fft_inv():
    assert zz.algorithm.fft([1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j], True) == [
        (4 + 0j),
        0j,
        0j,
        0j,
    ]
