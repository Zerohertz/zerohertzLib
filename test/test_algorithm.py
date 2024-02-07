import random

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


def test_soe():
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


def test_sort_int():
    arr = [random.randint(-100, 100) for _ in range(100)]
    sorted_arr = sorted(arr)
    assert sorted_arr == zz.algorithm.bubble_sort(arr.copy())
    assert sorted_arr == zz.algorithm.selection_sort(arr.copy())
    assert sorted_arr == zz.algorithm.insertion_sort(arr.copy())
    assert sorted_arr == zz.algorithm.merge_sort(arr.copy())
    assert sorted_arr == zz.algorithm.quick_sort(arr.copy())
    assert sorted_arr == zz.algorithm.heap_sort(arr.copy())


def test_sort_nat():
    arr = [random.randint(0, 100) for _ in range(100)]
    sorted_arr = sorted(arr)
    assert sorted_arr == zz.algorithm.counting_sort(arr.copy())
    assert sorted_arr == zz.algorithm.radix_sort(arr.copy())


# FROM: https://www.acmicpc.net/problem/1238
def test_floyd_warshall():
    graph = [[(1, 4), (2, 2), (3, 7)], [(0, 1), (2, 5)], [(0, 2), (3, 4)], [(1, 3)]]
    results = zz.algorithm.floyd_warshall(graph)
    for i, result in enumerate(results):
        assert result == zz.algorithm.dijkstra(graph, i)
        assert result == zz.algorithm.bellman_ford(graph, i)
    graph_no_cycle = [[(1, -2), (2, -3)], [(3, -5)], [(3, -1)], []]
    results = zz.algorithm.floyd_warshall(graph_no_cycle)
    for i, result in enumerate(results):
        assert result == zz.algorithm.bellman_ford(graph_no_cycle, i)
    graph_cycle = [[(1, -2), (2, 3)], [(2, -2)], [(0, -2), (3, 1)], []]
    assert zz.algorithm.floyd_warshall(graph_cycle) is None


def test_bellman_ford():
    graph = [[(1, 4), (2, 2), (3, 7)], [(0, 1), (2, 5)], [(0, 2), (3, 4)], [(1, 3)]]
    assert [0, 4, 2, 6] == zz.algorithm.bellman_ford(graph, 0)
    assert [1, 0, 3, 7] == zz.algorithm.bellman_ford(graph, 1)
    assert [2, 6, 0, 4] == zz.algorithm.bellman_ford(graph, 2)
    assert [4, 3, 6, 0] == zz.algorithm.bellman_ford(graph, 3)
    graph_cycle = [[(1, -2), (2, 3)], [(2, -2)], [(0, -2), (3, 1)], []]
    assert zz.algorithm.bellman_ford(graph_cycle, 0) is None
    assert zz.algorithm.bellman_ford(graph_cycle, 1) is None
    assert zz.algorithm.bellman_ford(graph_cycle, 2) is None
    # assert zz.algorithm.bellman_ford(graph_cycle, 3) is None


def test_dijkstra():
    graph = [[(1, 4), (2, 2), (3, 7)], [(0, 1), (2, 5)], [(0, 2), (3, 4)], [(1, 3)]]
    assert [0, 4, 2, 6] == zz.algorithm.dijkstra(graph, 0)
    assert [1, 0, 3, 7] == zz.algorithm.dijkstra(graph, 1)
    assert [2, 6, 0, 4] == zz.algorithm.dijkstra(graph, 2)
    assert [4, 3, 6, 0] == zz.algorithm.dijkstra(graph, 3)


def test_disjointset():
    disjointset = zz.algorithm.DisjointSet(5)
    disjointset.union(0, 1)
    disjointset.union(2, 3)
    disjointset.union(1, 2)
    assert disjointset.parent == [0, 0, 0, 2, 4]

    disjointset = zz.algorithm.DisjointSet(5, True)
    disjointset.union(0, 1)
    disjointset.union(2, 3)
    disjointset.union(1, 2)
    assert disjointset.parent == [0, 0, 0, 2, 4]

    disjointset = zz.algorithm.DisjointSetRank(5)
    disjointset.union(0, 1)
    disjointset.union(2, 3)
    disjointset.union(1, 2)
    assert disjointset.parent == [0, 0, 0, 2, 4]
    assert disjointset.rank == [2, 0, 1, 0, 0]

    disjointset = zz.algorithm.DisjointSetSize(5)
    disjointset.union(0, 1)
    disjointset.union(2, 3)
    disjointset.union(1, 2)
    assert disjointset.parent == [0, 0, 0, 2, 4]
    assert disjointset.size == [4, 1, 2, 1, 1]
    assert [disjointset.size[disjointset.find(i)] for i in range(5)] == [4, 4, 4, 4, 1]
