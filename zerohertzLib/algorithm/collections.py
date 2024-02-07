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

from typing import Optional


class DisjointSet:
    r"""Vanilla disjoint set

    Note:
        Time Complexity:
            - Without path compression:
                - Worst: :math:`O(V)`
                - Average: :math:`O(V)`
            - With path compression:
                - Worst: :math:`O(\log{V})`
                - Average: :math:`O(\alpha(V))`

        - :math:`V`: Node의 수
        - :math:`\alpha(V)`: `Ackermann function <https://ko.wikipedia.org/wiki/%EC%95%84%EC%BB%A4%EB%A7%8C_%ED%95%A8%EC%88%98>`_ 의 역함수 (:math:`O(\alpha(V))\simeq O(1)`)

    Args:
        size (``int``): Node의 수
        compression (``Optional[bool]``): Path compression 여부

    Attributes:
        parent (``List[int]``): Node에 따른 부모 node의 index

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

    def __init__(self, size: int, compression: Optional[bool] = False) -> None:
        self.parent = list(range(size))
        self.compression = compression

    def find(self, node: int) -> int:
        """
        Args:
            node (``int``): 목표 node의 index

        Returns:
            ``int``: 목표 node에 대한 root node의 index
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
            node1 (``int``): 목표 node의 index
            node2 (``int``): 목표 node의 index

        Returns:
            ``None``: ``self.parent`` 에 root node의 index를 update
        """
        root1, root2 = self.find(node1), self.find(node2)
        if root1 != root2:
            self.parent[root2] = root1


class DisjointSetRank(DisjointSet):
    r"""Disjoint set (union by rank)

    Note:
        Time Complexity:
            - Worst: :math:`O(\log{V})`
            - Average: :math:`O(\alpha(V))`

        - :math:`V`: Node의 수
        - :math:`\alpha(V)`: `Ackermann function <https://ko.wikipedia.org/wiki/%EC%95%84%EC%BB%A4%EB%A7%8C_%ED%95%A8%EC%88%98>`_ 의 역함수 (:math:`O(\alpha(V))\simeq O(1)`)

    Args:
        size (``int``): Node의 수

    Attributes:
        parent (``List[int]``): Node에 따른 부모 node의 index
        rank (``List[int]``): Node에 따른 rank

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
            node1 (``int``): 목표 node의 index
            node2 (``int``): 목표 node의 index

        Returns:
            ``None``: ``self.parent`` 에 root node의 index를 update
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
            - Worst: :math:`O(\log{V})`
            - Average: :math:`O(\alpha(V))`

        - :math:`V`: Node의 수
        - :math:`\alpha(V)`: `Ackermann function <https://ko.wikipedia.org/wiki/%EC%95%84%EC%BB%A4%EB%A7%8C_%ED%95%A8%EC%88%98>`_ 의 역함수 (:math:`O(\alpha(V))\simeq O(1)`)

    Args:
        size (``int``): Node의 수

    Attributes:
        parent (``List[int]``): Node에 따른 부모 node의 index
        size (``List[int]``): Node에 따른 size

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
            node1 (``int``): 목표 node의 index
            node2 (``int``): 목표 node의 index

        Returns:
            ``None``: ``self.parent`` 에 root node의 index를 update
        """
        root1, root2 = self.find(node1), self.find(node2)
        if self.size[root1] < self.size[root2]:
            self.parent[root1] = root2
            self.size[root2] += self.size[root1]
        else:
            self.parent[root2] = root1
            self.size[root1] += self.size[root2]
