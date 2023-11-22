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

from typing import Tuple

import numpy as np
from matplotlib.path import Path
from numpy.typing import DTypeLike, NDArray

from .util import _is_bbox


def _cwh2xyxy(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, y_0 = box[:2] - box[2:] / 2
    x_1, y_1 = box[:2] + box[2:] / 2
    return np.array([x_0, y_0, x_1, y_1], dtype=box.dtype)


def cwh2xyxy(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``NDArray[DTypeLike]``): ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.cwh2xyxy(np.array([20, 30, 20, 20]))
        array([10, 20, 30, 40])
        >>> zz.vision.cwh2xyxy(np.array([[20, 30, 20, 20], [50, 75, 40, 50]]))
        array([[ 10,  20,  30,  40],
               [ 30,  50,  70, 100]])
    """
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if poly:
        raise ValueError("The 'cwh' must be of shape [4], [N, 4]")
    if multi:
        boxes = np.zeros((shape[0], 4), dtype=box.dtype)
        for i, box_ in enumerate(box):
            boxes[i] = _cwh2xyxy(box_)
        return boxes
    return _cwh2xyxy(box)


def _cwh2poly(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, y_0 = box[:2] - box[2:] / 2
    x_1, y_1 = box[:2] + box[2:] / 2
    return np.array([[x_0, y_0], [x_1, y_0], [x_1, y_1], [x_0, y_1]], dtype=box.dtype)


def cwh2poly(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``NDArray[DTypeLike]``): ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Examples:
        >>> zz.vision.cwh2poly(np.array([20, 30, 20, 20]))
        array([[10, 20],
               [30, 20],
               [30, 40],
               [10, 40]])
        >>> zz.vision.cwh2poly(np.array([[20, 30, 20, 20], [50, 75, 40, 50]]))
        array([[[ 10,  20],
                [ 30,  20],
                [ 30,  40],
                [ 10,  40]],
               [[ 30,  50],
                [ 70,  50],
                [ 70, 100],
                [ 30, 100]]])
    """
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if poly:
        raise ValueError("The 'cwh' must be of shape [4], [N, 4]")
    if multi:
        boxes = np.zeros((shape[0], 4, 2), dtype=box.dtype)
        for i, box_ in enumerate(box):
            boxes[i] = _cwh2poly(box_)
        return boxes
    return _cwh2poly(box)


def _xyxy2cwh(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, y_0, x_1, y_1 = box
    return np.array(
        [(x_0 + x_1) / 2, (y_0 + y_1) / 2, x_1 - x_0, y_1 - y_0], dtype=box.dtype
    )


def xyxy2cwh(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``NDArray[DTypeLike]``): ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.xyxy2cwh(np.array([10, 20, 30, 40]))
        array([20, 30, 20, 20])
        >>> zz.vision.xyxy2cwh(np.array([[10, 20, 30, 40], [30, 50, 70, 100]]))
        array([[20, 30, 20, 20],
               [50, 75, 40, 50]])
    """
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if poly:
        raise ValueError("The 'xyxy' must be of shape [4], [N, 4]")
    if multi:
        boxes = np.zeros((shape[0], 4), dtype=box.dtype)
        for i, box_ in enumerate(box):
            boxes[i] = _xyxy2cwh(box_)
        return boxes
    return _xyxy2cwh(box)


def _xyxy2poly(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, y_0, x_1, y_1 = box
    return np.array([[x_0, y_0], [x_1, y_0], [x_1, y_1], [x_0, y_1]], dtype=box.dtype)


def xyxy2poly(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``NDArray[DTypeLike]``): ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Examples:
        >>> zz.vision.xyxy2poly(np.array([10, 20, 30, 40]))
        array([[10, 20],
               [30, 20],
               [30, 40],
               [10, 40]])
        >>> zz.vision.xyxy2poly(np.array([[10, 20, 30, 40], [30, 50, 70, 100]]))
        array([[[ 10,  20],
                [ 30,  20],
                [ 30,  40],
                [ 10,  40]],
               [[ 30,  50],
                [ 70,  50],
                [ 70, 100],
                [ 30, 100]]])
    """
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if poly:
        raise ValueError("The 'xyxy' must be of shape [4], [N, 4]")
    if multi:
        boxes = np.zeros((shape[0], 4, 2), dtype=box.dtype)
        for i, box_ in enumerate(box):
            boxes[i] = _xyxy2poly(box_)
        return boxes
    return _xyxy2poly(box)


def _poly2cwh(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, x_1 = box[:, 0].min(), box[:, 0].max()
    y_0, y_1 = box[:, 1].min(), box[:, 1].max()
    return np.array(
        [(x_0 + x_1) / 2, (y_0 + y_1) / 2, x_1 - x_0, y_1 - y_0], dtype=box.dtype
    )


def poly2cwh(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``NDArray[DTypeLike]``): ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.poly2cwh(np.array([[10, 20], [30, 20], [30, 40], [10, 40]]))
        array([20, 30, 20, 20])
        >>> zz.vision.poly2cwh(np.array([[[10, 20], [30, 20], [30, 40], [10, 40]], [[30, 50], [70, 50], [70, 100], [30, 100]]]))
        array([[20, 30, 20, 20],
               [50, 75, 40, 50]])
    """
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if not poly:
        raise ValueError("The 'poly' must be of shape [4, 2], [N, 4, 2]")
    if multi:
        boxes = np.zeros((shape[0], 4), dtype=box.dtype)
        for i, box_ in enumerate(box):
            boxes[i] = _poly2cwh(box_)
        return boxes
    return _poly2cwh(box)


def _poly2xyxy(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, x_1 = box[:, 0].min(), box[:, 0].max()
    y_0, y_1 = box[:, 1].min(), box[:, 1].max()
    return np.array([x_0, y_0, x_1, y_1], dtype=box.dtype)


def poly2xyxy(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``NDArray[DTypeLike]``): ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.poly2xyxy(np.array([[10, 20], [30, 20], [30, 40], [10, 40]]))
        array([10, 20, 30, 40])
        >>> zz.vision.poly2xyxy(np.array([[[10, 20], [30, 20], [30, 40], [10, 40]], [[30, 50], [70, 50], [70, 100], [30, 100]]]))
        array([[ 10,  20,  30,  40],
               [ 30,  50,  70, 100]])
    """
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if not poly:
        raise ValueError("The 'poly' must be of shape [4, 2], [N, 4, 2]")
    if multi:
        boxes = np.zeros((shape[0], 4), dtype=box.dtype)
        for i, box_ in enumerate(box):
            boxes[i] = _poly2xyxy(box_)
        return boxes
    return _poly2xyxy(box)


def poly2mask(poly: NDArray[DTypeLike], shape: Tuple[int]) -> NDArray[bool]:
    """다각형 좌표를 입력받아 mask로 변환

    Args:
        poly (``NDArray[DTypeLike]``): Mask의 꼭짓점 좌표 (``[N, 2]``)
        shape (``Tuple[int]``): 출력될 mask의 shape ``(H, W)``

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, C]``)

    Examples:
        >>> poly = np.array([[10, 10], [20, 10], [30, 40], [20, 60], [10, 20]])
        >>> mask = zz.vision.poly2mask(poly, (70, 100))
        >>> mask.shape
        (70, 100)
        >>> mask.dtype
        dtype('bool')

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284488846-9237c52b-d181-447c-95da-f67aa8fb2854.png
            :alt: Visualzation Result
            :align: center
            :width: 300px
    """
    poly = Path(poly)
    pts_x, pts_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    pts_x, pts_y = pts_x.flatten(), pts_y.flatten()
    points = np.vstack((pts_x, pts_y)).T
    grid = poly.contains_points(points)
    mask = grid.reshape(shape)
    return mask
