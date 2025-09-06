# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)


class DisjointSet:
    r"""Vanilla disjoint set

    Note:
        Time Complexity:

        - Without path compression:
            - Worst: $O(V)$
            - Average: $O(V)$
        - With path compression:
            - Worst: $O(\log{V})$
            - Average: $O(\alpha(V))$

        - $V$: Node의 수
        - $\alpha(V)$: [Ackermann function](https://ko.wikipedia.org/wiki/%EC%95%84%EC%BB%A4%EB%A7%8C_%ED%95%A8%EC%88%98) 의 역함수 ($O(\alpha(V))\simeq O(1)$)

    Args:
        size: Node의 수
        compression: Path compression 여부

    Attributes:
        parent: Node에 따른 부모 node의 index

    Examples:
        >>> disjointset = zz.algorithm.DisjointSet(5)
        >>> disjointset.union(0, 1)
        >>> disjointset.union(2, 3)
        >>> disjointset.union(1, 2)
        >>> disjointset.parent
        [0, 0, 0, 2, 4]
        >>> disjointset = zz.algorithm.DisjointSet(5, True)
        >>> disjointset.union(0, 1)
        >>> disjointset.union(2, 3)
        >>> disjointset.union(1, 2)
        >>> disjointset.parent
        [0, 0, 0, 2, 4]
    """

    def __init__(self, size: int, compression: bool = False) -> None:
        self.parent = list(range(size))
        self.compression = compression

    def find(self, node: int) -> int:
        """
        Args:
            node: 목표 node의 index

        Returns:
            목표 node에 대한 root node의 index
        """
        if not self.compression:
            while node != self.parent[node]:
                node = self.parent[node]
            return node
        if node != self.parent[node]:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1: int, node2: int) -> None:
        """
        Args:
            node1: 목표 node의 index
            node2: 목표 node의 index

        Returns:
            `self.parent` 에 root node의 index를 update
        """
        root1, root2 = self.find(node1), self.find(node2)
        if root1 != root2:
            self.parent[root2] = root1


class DisjointSetRank(DisjointSet):
    r"""Disjoint set (union by rank)

    Note:
        Time Complexity:

        - Worst: $O(\log{V})$
        - Average: $O(\alpha(V))$

        - $V$: Node의 수
        - $\alpha(V)$: [Ackermann function](https://ko.wikipedia.org/wiki/%EC%95%84%EC%BB%A4%EB%A7%8C_%ED%95%A8%EC%88%98) 의 역함수 ($O(\alpha(V))\simeq O(1)$)

    Args:
        size: Node의 수

    Attributes:
        parent: Node에 따른 부모 node의 index
        rank: Node에 따른 rank

    Examples:
        >>> disjointset = zz.algorithm.DisjointSetRank(5)
        >>> disjointset.union(0, 1)
        >>> disjointset.union(2, 3)
        >>> disjointset.union(1, 2)
        >>> disjointset.parent
        [0, 0, 0, 2, 4]
        >>> disjointset.rank
        [2, 0, 1, 0, 0]
    """

    def __init__(self, size: int) -> None:
        super().__init__(size, True)
        self.rank = [0 for _ in range(size)]

    def union(self, node1: int, node2: int) -> None:
        """
        Args:
            node1: 목표 node의 index
            node2: 목표 node의 index

        Returns:
            `self.parent` 에 root node의 index를 update
        """
        root1, root2 = self.find(node1), self.find(node2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


class DisjointSetSize(DisjointSet):
    r"""Disjoint set (union by size)

    Note:
        Time Complexity:

        - Worst: $O(\log{V})$
        - Average: $O(\alpha(V))$

        - $V$: Node의 수
        - $\alpha(V)$: [Ackermann function](https://ko.wikipedia.org/wiki/%EC%95%84%EC%BB%A4%EB%A7%8C_%ED%95%A8%EC%88%98) 의 역함수 ($O(\alpha(V))\simeq O(1)$)

    Args:
        size: Node의 수

    Attributes:
        parent: Node에 따른 부모 node의 index
        size: Node에 따른 size

    Examples:
        >>> disjointset = zz.algorithm.DisjointSetSize(5)
        >>> disjointset.union(0, 1)
        >>> disjointset.union(2, 3)
        >>> disjointset.union(1, 2)
        >>> disjointset.parent
        [0, 0, 0, 2, 4]
        >>> disjointset.size
        [4, 1, 2, 1, 1]
        >>> [disjointset.size[disjointset.find(i)] for i in range(5)]
        [4, 4, 4, 4, 1]
    """

    def __init__(self, size: int) -> None:
        super().__init__(size, True)
        self.size = [1 for _ in range(size)]

    def union(self, node1: int, node2: int) -> None:
        """
        Args:
            node1: 목표 node의 index
            node2: 목표 node의 index

        Returns:
            `self.parent` 에 root node의 index를 update
        """
        root1, root2 = self.find(node1), self.find(node2)
        if self.size[root1] < self.size[root2]:
            self.parent[root1] = root2
            self.size[root2] += self.size[root1]
        else:
            self.parent[root2] = root1
            self.size[root1] += self.size[root2]
