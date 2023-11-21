from typing import Tuple

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
    # [cx, cy, w, h]
    if len(shape) == 1 and shape[0] == 4:
        multi = False
        poly = False
    elif len(shape) == 2:
        # N * [cx, cy, w, h]
        if shape[1] == 4:
            multi = True
            poly = False
        # [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        elif shape[0] == 4 and shape[1] == 2:
            multi = False
            poly = True
        else:
            raise Exception("The 'box' must be of shape [4], [N, 4], [4, 2], [N, 4, 2]")
        # [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    elif len(shape) == 3 and shape[1] == 4 and shape[2] == 2:
        multi = True
        poly = True
    else:
        raise Exception("The 'box' must be of shape [4], [N, 4], [4, 2], [N, 4, 2]")
    return multi, poly


def _cwh2xyxy(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x0, y0 = box[:2] - box[2:] / 2
    x1, y1 = box[:2] + box[2:] / 2
    return np.array([x0, y0, x1, y1], dtype=box.dtype)


def cwh2xyxy(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``NDArray[DTypeLike]``): ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[x0, y0, y1, y2]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.cwh2xyxy(np.array([20, 30, 20, 20]))
        array([10, 20, 30, 40])
        >>> zz.vision.cwh2xyxy(np.array([[20, 30, 20, 20], [50, 75, 40, 50]]))
        array([[ 10,  20,  30,  40],
               [ 30,  50,  70, 100]])
    """
    shape = box.shape
    multi, poly = _isBbox(shape)
    if poly:
        raise Exception("The 'cwh' must be of shape [4], [N, 4]")
    if multi:
        boxes = np.zeros((shape[0], 4), dtype=box.dtype)
        for i, b in enumerate(box):
            boxes[i] = _cwh2xyxy(b)
        return boxes
    else:
        return _cwh2xyxy(box)


def _cwh2poly(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x0, y0 = box[:2] - box[2:] / 2
    x1, y1 = box[:2] + box[2:] / 2
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=box.dtype)


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
    multi, poly = _isBbox(shape)
    if poly:
        raise Exception("The 'cwh' must be of shape [4], [N, 4]")
    if multi:
        boxes = np.zeros((shape[0], 4, 2), dtype=box.dtype)
        for i, b in enumerate(box):
            boxes[i] = _cwh2poly(b)
        return boxes
    else:
        return _cwh2poly(box)


def _xyxy2cwh(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x0, y0, x1, y1 = box
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dtype=box.dtype)


def xyxy2cwh(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``NDArray[DTypeLike]``): ``[x0, y0, y1, y2]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

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
    multi, poly = _isBbox(shape)
    if poly:
        raise Exception("The 'xyxy' must be of shape [4], [N, 4]")
    if multi:
        boxes = np.zeros((shape[0], 4), dtype=box.dtype)
        for i, b in enumerate(box):
            boxes[i] = _xyxy2cwh(b)
        return boxes
    else:
        return _xyxy2cwh(box)


def _xyxy2poly(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=box.dtype)


def xyxy2poly(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``NDArray[DTypeLike]``): ``[x0, y0, y1, y2]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

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
    multi, poly = _isBbox(shape)
    if poly:
        raise Exception("The 'xyxy' must be of shape [4], [N, 4]")
    if multi:
        boxes = np.zeros((shape[0], 4, 2), dtype=box.dtype)
        for i, b in enumerate(box):
            boxes[i] = _xyxy2poly(b)
        return boxes
    else:
        return _xyxy2poly(box)


def _poly2cwh(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x0, x1 = box[:, 0].min(), box[:, 0].max()
    y0, y1 = box[:, 1].min(), box[:, 1].max()
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dtype=box.dtype)


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
    multi, poly = _isBbox(shape)
    if not poly:
        raise Exception("The 'poly' must be of shape [4, 2], [N, 4, 2]")
    if multi:
        boxes = np.zeros((shape[0], 4), dtype=box.dtype)
        for i, b in enumerate(box):
            boxes[i] = _poly2cwh(b)
        return boxes
    else:
        return _poly2cwh(box)


def _poly2xyxy(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x0, x1 = box[:, 0].min(), box[:, 0].max()
    y0, y1 = box[:, 1].min(), box[:, 1].max()
    return np.array([x0, y0, x1, y1], dtype=box.dtype)


def poly2xyxy(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``NDArray[DTypeLike]``): ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[x0, y0, y1, y2]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.poly2xyxy(np.array([[10, 20], [30, 20], [30, 40], [10, 40]]))
        array([10, 20, 30, 40])
        >>> zz.vision.poly2xyxy(np.array([[[10, 20], [30, 20], [30, 40], [10, 40]], [[30, 50], [70, 50], [70, 100], [30, 100]]]))
        array([[ 10,  20,  30,  40],
               [ 30,  50,  70, 100]])
    """
    shape = box.shape
    multi, poly = _isBbox(shape)
    if not poly:
        raise Exception("The 'poly' must be of shape [4, 2], [N, 4, 2]")
    if multi:
        boxes = np.zeros((shape[0], 4), dtype=box.dtype)
        for i, b in enumerate(box):
            boxes[i] = _poly2xyxy(b)
        return boxes
    else:
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
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    grid = poly.contains_points(points)
    mask = grid.reshape(shape)
    return mask
