from typing import List, Tuple, Union

import numpy as np
from matplotlib.path import Path
from numpy.typing import DTypeLike, NDArray


def _isBbox(shape: Tuple[int]) -> Tuple[bool]:
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
            raise Exception("The 'box' must be of shape [4], [N, 4], [4, 2], [N, 4, 2]")
    elif len(shape) == 3 and shape[1] == 4 and shape[2] == 2:
        # N *[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        multi = True
        poly = True
    else:
        raise Exception("The 'box' must be of shape [4], [N, 4], [4, 2], [N, 4, 2]")
    return multi, poly


def isPtsInPoly(
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
        >>> zz.vision.isPtsInPoly(poly, [20, 20])
        True
        >>> zz.vision.isPtsInPoly(poly, [[20, 20], [100, 100]])
        array([ True, False])
        >>> zz.vision.isPtsInPoly(poly, np.array([20, 20]))
        True
        >>> zz.vision.isPtsInPoly(poly, np.array([[20, 20], [100, 100]]))
        array([ True, False])
    """
    poly = Path(poly)
    if isinstance(pts, list):
        if isinstance(pts[0], list):
            return poly.contains_points(pts)
        else:
            return poly.contains_point(pts)
    elif isinstance(pts, np.ndarray):
        shape = pts.shape
        if len(shape) == 1:
            return poly.contains_point(pts)
        elif len(shape) == 2:
            return poly.contains_points(pts)
        else:
            raise Exception("The 'pts' must be of shape [2], [N, 2]")
    else:
        raise Exception("The 'pts' must be 'list' or 'np.ndarray'")
