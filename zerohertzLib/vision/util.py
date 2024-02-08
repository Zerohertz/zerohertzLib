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

from typing import List, Tuple, Union

import cv2
import numpy as np
from matplotlib.path import Path
from numpy.typing import DTypeLike, NDArray


def _cvt_bgra(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """cv2로 읽어온 image를 BGRA 채널로 전환

    Args:
        img (``NDArray[np.uint8]``): 입력 image (``[H, W, C]``)

    Returns:
        ``NDArray[np.uint8]``: BGRA image (``[H, W, 4]``)
    """
    shape = img.shape
    if len(shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _is_bbox(shape: Tuple[int]) -> Tuple[bool]:
    """Bbox 여부 검증

    Args:
        shape (``Tuple[int]``): Bbox의 `shape`

    Returns:
        ``bool``: 복수의 bbox 여부 및 format의 정보
    """
    if len(shape) == 1 and shape[0] == 4:
        # [cx, cy, w, h] or N * [x0, y0, x1, y1]
        multi = False
        poly = False
    elif len(shape) == 2:
        if shape[1] == 4:
            # N * [cx, cy, w, h] or N * [x0, y0, x1, y1]
            multi = True
            poly = False
        elif shape[0] >= 4 and shape[1] == 2:
            # [[x0, y0], [x1, y1], [x2, y2], [x3, y3], ...]
            multi = False
            poly = True
        else:
            raise ValueError(
                "The 'box' must be of shape [4], [N, 4], [4, 2], [N, 4, 2]"
            )
    elif len(shape) == 3 and shape[1] == 4 and shape[2] == 2:
        # N *[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        multi = True
        poly = True
    else:
        raise ValueError("The 'box' must be of shape [4], [N, 4], [4, 2], [N, 4, 2]")
    return multi, poly


def is_pts_in_poly(
    poly: NDArray[DTypeLike], pts: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> Union[bool, NDArray[bool]]:
    """지점들의 좌표 내 존재 여부 확인 함수

    Args:
        poly (``NDArray[DTypeLike]``): 다각형 (``[N, 2]``)
        pts (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): point (``[2]`` or ``[N, 2]``)

    Returns:
        ``Union[bool, NDArray[DTypeLike]]``: 입력 ``point`` 의 다각형 ``poly`` 내부 존재 여부

    Examples:
        >>> poly = np.array([[10, 10], [20, 10], [30, 40], [20, 60], [10, 20]])
        >>> zz.vision.is_pts_in_poly(poly, [20, 20])
        True
        >>> zz.vision.is_pts_in_poly(poly, [[20, 20], [100, 100]])
        array([ True, False])
        >>> zz.vision.is_pts_in_poly(poly, np.array([20, 20]))
        True
        >>> zz.vision.is_pts_in_poly(poly, np.array([[20, 20], [100, 100]]))
        array([ True, False])
    """
    poly = Path(poly)
    if isinstance(pts, list):
        if isinstance(pts[0], list):
            return poly.contains_points(pts)
        return poly.contains_point(pts)
    if isinstance(pts, np.ndarray):
        shape = pts.shape
        if len(shape) == 1:
            return poly.contains_point(pts)
        if len(shape) == 2:
            return poly.contains_points(pts)
        raise ValueError("The 'pts' must be of shape [2], [N, 2]")
    raise TypeError("The 'pts' must be 'list' or 'np.ndarray'")
