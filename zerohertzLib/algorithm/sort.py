# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)


def bubble_sort(arr: list[int]) -> list[int]:
    """Bubble Sort Algorithm: 연속된 값들을 비교하여 가장 큰 값을 배열의 끝으로 이동시키는 방식으로 정렬

    Args:
        arr: 정렬할 정수 list

    Returns:
        오름차순으로 정렬된 list

    Examples:
        >>> zz.algorithm.bubble_sort([64, 34, 25, 12, 22, 11, 90])
        [11, 12, 22, 25, 34, 64, 90]
    """
    arr_len = len(arr)
    for i in range(arr_len):
        for j in range(0, arr_len - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def selection_sort(arr: list[int]) -> list[int]:
    """Selection Sort Algorithm: 배열에서 가장 작은 값을 찾아 해당 값을 배열의 앞부분으로 이동시키는 방식으로 정렬

    Args:
        arr: 정렬할 정수 list

    Returns:
        오름차순으로 정렬된 list

    Examples:
        >>> zz.algorithm.selection_sort([64, 34, 25, 12, 22, 11, 90])
        [11, 12, 22, 25, 34, 64, 90]
    """
    for i, _ in enumerate(arr):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[min_idx] > arr[j]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def insertion_sort(arr: list[int]) -> list[int]:
    """Insertion Sort Algorithm: 각 값들을 이미 정렬된 부분의 올바른 위치에 삽입하는 방식으로 정렬

    Args:
        arr: 정렬할 정수 list

    Returns:
        오름차순으로 정렬된 list

    Examples:
        >>> zz.algorithm.insertion_sort([64, 34, 25, 12, 22, 11, 90])
        [11, 12, 22, 25, 34, 64, 90]
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def merge_sort(arr: list[int]) -> list[int]:
    """Merge Sort Algorithm: 분할 정복 방법을 사용하여 배열을 절반으로 나누고, 각 부분을 정렬한 다음 합치는 방식으로 정렬

    Args:
        arr: 정렬할 정수 list

    Returns:
        오름차순으로 정렬된 list

    Examples:
        >>> zz.algorithm.merge_sort([64, 34, 25, 12, 22, 11, 90])
        [11, 12, 22, 25, 34, 64, 90]
    """
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        merge_sort(left)
        merge_sort(right)
        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
    return arr


def quick_sort(arr: list[int]) -> list[int]:
    """Quick Sort Algorithm: Pivot을 선택하여 이보다 작은 요소는 왼쪽, 큰 요소는 오른쪽에 위치시키는 방식으로 분할 정복을 사용하여 정렬

    Args:
        arr: 정렬할 정수 list

    Returns:
        오름차순으로 정렬된 list

    Examples:
        >>> zz.algorithm.quick_sort([64, 34, 25, 12, 22, 11, 90])
        [11, 12, 22, 25, 34, 64, 90]
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def _heapify(arr: list[int], n: int, i: int) -> None:
    """Helper function for Heap Sort: 주어진 node를 root로 하는 subtree를 heap 속성을 만족하도록 재구성

    Args:
        arr: 힙을 구성하는 list
        n: List의 크기
        i: 재구성할 subtree의 root node index
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[i] < arr[left]:
        largest = left
    if right < n and arr[largest] < arr[right]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)


def heap_sort(arr: list[int]) -> list[int]:
    """Heap Sort Algorithm: 배열 요소들을 heap으로 구성한 다음, 최대 heap 속성을 이용하여 정렬

    Args:
        arr: 정렬할 정수 list

    Returns:
        오름차순으로 정렬된 list

    Examples:
        >>> zz.algorithm.heap_sort([64, 34, 25, 12, 22, 11, 90])
        [11, 12, 22, 25, 34, 64, 90]
    """
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        _heapify(arr, i, 0)
    return arr


def counting_sort(arr: list[int]) -> list[int]:
    """Counting Sort Algorithm: 각 숫자의 개수를 세어 정렬

    Args:
        arr: 정렬할 정수 list

    Returns:
        오름차순으로 정렬된 list

    Examples:
        >>> zz.algorithm.counting_sort([64, 34, 25, 12, 22, 11, 90])
        [11, 12, 22, 25, 34, 64, 90]
    """
    max_val = max(arr) + 1
    count = [0] * max_val
    for arr_ in arr:
        count[arr_] += 1
    idx = 0
    for value in range(max_val):
        for _ in range(count[value]):
            arr[idx] = value
            idx += 1
    return arr


def _counting_sort_for_radix(arr: list[int], exp: int) -> None:
    """Helper function for Radix Sort: 기수 정렬을 위해 주어진 자릿수 (`exp`)에 따라 각 요소를 정렬

    Args:
        arr: 정렬할 정수 list
        exp: 현재 정렬할 자릿수
    """
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    for i in range(n):
        arr[i] = output[i]


def radix_sort(arr: list[int]) -> list[int]:
    """Radix Sort Algorithm: 각 자릿수에 대해 개별적으로 정렬

    Args:
        arr: 정렬할 정수 list

    Returns:
        오름차순으로 정렬된 list

    Examples:
        >>> zz.algorithm.radix_sort([64, 34, 25, 12, 22, 11, 90])
        [11, 12, 22, 25, 34, 64, 90]
    """
    max_val = max(arr)
    exp = 1
    while max_val / exp > 0:
        _counting_sort_for_radix(arr, exp)
        exp *= 10
    return arr
