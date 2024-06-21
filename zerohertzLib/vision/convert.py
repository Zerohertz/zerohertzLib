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

import base64
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from matplotlib.path import Path
from numpy.typing import DTypeLike, NDArray

from .util import _is_bbox


def _list2np(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    if isinstance(box, list):
        return np.array(box)
    return box


def _cwh2xyxy(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, y_0 = box[:2] - box[2:] / 2
    x_1, y_1 = box[:2] + box[2:] / 2
    return np.array([x_0, y_0, x_1, y_1], dtype=box.dtype)


def cwh2xyxy(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.cwh2xyxy([20, 30, 20, 20])
        array([10, 20, 30, 40])
        >>> zz.vision.cwh2xyxy(np.array([[20, 30, 20, 20], [50, 75, 40, 50]]))
        array([[ 10,  20,  30,  40],
               [ 30,  50,  70, 100]])
    """
    box = _list2np(box)
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


def cwh2poly(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Examples:
        >>> zz.vision.cwh2poly([20, 30, 20, 20])
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
    box = _list2np(box)
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


def xyxy2cwh(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.xyxy2cwh([10, 20, 30, 40])
        array([20, 30, 20, 20])
        >>> zz.vision.xyxy2cwh(np.array([[10, 20, 30, 40], [30, 50, 70, 100]]))
        array([[20, 30, 20, 20],
               [50, 75, 40, 50]])
    """
    box = _list2np(box)
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


def xyxy2poly(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Examples:
        >>> zz.vision.xyxy2poly([10, 20, 30, 40])
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
    box = _list2np(box)
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


def poly2cwh(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.poly2cwh([[10, 20], [30, 20], [30, 40], [10, 40]])
        array([20, 30, 20, 20])
        >>> zz.vision.poly2cwh(np.array([[[10, 20], [30, 20], [30, 40], [10, 40]], [[30, 50], [70, 50], [70, 100], [30, 100]]]))
        array([[20, 30, 20, 20],
               [50, 75, 40, 50]])
    """
    box = _list2np(box)
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


def poly2xyxy(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.poly2xyxy([[10, 20], [30, 20], [30, 40], [10, 40]])
        array([10, 20, 30, 40])
        >>> zz.vision.poly2xyxy(np.array([[[10, 20], [30, 20], [30, 40], [10, 40]], [[30, 50], [70, 50], [70, 100], [30, 100]]]))
        array([[ 10,  20,  30,  40],
               [ 30,  50,  70, 100]])
    """
    box = _list2np(box)
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


def _poly2mask(poly: NDArray[DTypeLike], shape: Tuple[int]) -> NDArray[bool]:
    poly = Path(poly)
    pts_x, pts_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    pts_x, pts_y = pts_x.flatten(), pts_y.flatten()
    points = np.vstack((pts_x, pts_y)).T
    grid = poly.contains_points(points)
    mask = grid.reshape(shape)
    return mask


def poly2mask(
    poly: Union[List[Union[int, float]], NDArray[DTypeLike], List[NDArray[DTypeLike]]],
    shape: Tuple[int],
) -> NDArray[bool]:
    """다각형 좌표를 입력받아 mask로 변환

    Args:
        poly (``Union[List[Union[int, float]], NDArray[DTypeLike], List[NDArray[DTypeLike]]]``): Mask의 꼭짓점 좌표 (``[M, 2]`` or ``[N, M, 2]``)
        shape (``Tuple[int]``): 출력될 mask의 shape ``(H, W)``

    Returns:
        ``NDArray[bool]``: 변환된 mask (``[H, W]`` or ``[N, H, W]``)

    Examples:
        >>> poly = [[10, 10], [20, 10], [30, 40], [20, 60], [10, 20]]
        >>> mask1 = zz.vision.poly2mask(poly, (70, 100))
        >>> mask1.shape
        (70, 100)
        >>> mask1.dtype
        dtype('bool')
        >>> poly = np.array(poly)
        >>> mask2 = zz.vision.poly2mask([poly, poly - 10, poly + 20], (70, 100))
        >>> mask2.shape
        (3, 70, 100)
        >>> mask2.dtype
        dtype('bool')

        .. image:: _static/examples/dynamic/vision.poly2mask.png
            :align: center
            :width: 300px
    """
    if (isinstance(poly, list) and isinstance(poly[0], np.ndarray)) or (
        isinstance(poly, np.ndarray) and len(poly.shape) == 3
    ):
        mks = []
        for _poly in poly:
            mks.append(_poly2mask(_poly, shape))
        mks = np.array(mks)
    else:
        mks = _poly2mask(_list2np(poly), shape)
    return mks


def poly2area(poly: Union[List[Union[int, float]], NDArray[DTypeLike]]) -> float:
    """다각형의 면적을 산출하는 함수

    Args:
        poly (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): 다각형 (``[N, 2]``)

    Returns:
        ``float``: 다각형의 면적

    Examples:
        >>> poly = [[10, 10], [20, 10], [30, 40], [20, 60], [10, 20]]
        >>> zz.vision.poly2area(poly)
        550.0
        >>> box = np.array([[100, 200], [1200, 200], [1200, 1000], [100, 1000]])
        >>> zz.vision.poly2area(box)
        880000.0
    """
    poly = _list2np(poly)
    pts_x = poly[:, 0]
    pts_y = poly[:, 1]
    return 0.5 * np.abs(
        np.dot(pts_x, np.roll(pts_y, 1)) - np.dot(pts_y, np.roll(pts_x, 1))
    )


def poly2ratio(poly: Union[List[Union[int, float]], NDArray[DTypeLike]]) -> float:
    """다각형의 bbox 대비 다각형의 면적 비율을 산출하는 함수

    Args:
        poly (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): 다각형 (``[N, 2]``)

    Returns:
        ``float``: 다각형의 bbox 대비 다각형의 비율

    Examples:
        >>> poly = [[10, 10], [20, 10], [30, 40], [20, 60], [10, 20]]
        >>> zz.vision.poly2ratio(poly)
        0.55
        >>> box = np.array([[100, 200], [1200, 200], [1200, 1000], [100, 1000]])
        >>> zz.vision.poly2ratio(box)
        1.0
    """
    poly_area = poly2area(poly)
    _, _, height, width = poly2cwh(poly)
    bbox_area = height * width
    return poly_area / bbox_area


def encode(img: NDArray[np.uint8], ext: Optional[str] = "png") -> str:
    """Base64 encoding

    Args:
        img (``NDArray[np.uint8]``): ``cv2.imread``로 읽어온 image
        ext (``Optional[str]``): 출력 파일의 확장자

    Returns:
        ``str``: Base64 encoding된 문자열

    >>> zz.vision.encode(img)
    'iVBORw0KGg...'
    """
    _, buffer = cv2.imencode(f".{ext}", img)
    return base64.b64encode(buffer).decode("utf-8")


def decode(img: str) -> NDArray[np.uint8]:
    """Base64 decoding

    Args:
        img (``str``): ``zz.vision.encode``로 encoding된 문자열

    Returns:
        ``NDArray[np.uint8]``: Base64 decoding image

    >>> zz.vision.decode(img).shape
    (802, 802, 3)
    """
    img_bytes = base64.b64decode(img)
    buffer = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
