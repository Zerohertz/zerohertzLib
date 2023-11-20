import numpy as np
from numpy.typing import DTypeLike, NDArray


def _xyxy2xywh(
    box: NDArray[DTypeLike],
) -> NDArray[DTypeLike]:
    shape = box.shape
    assert shape[0] == 4
    assert shape[1] == 2
    w0, w1 = map(int, (np.min(box[:, 0]), np.max(box[:, 0])))
    h0, h1 = map(int, (np.min(box[:, 1]), np.max(box[:, 1])))
    cx = (w0 + w1) / 2
    cy = (h0 + h1) / 2
    w = w1 - w0
    h = h1 - h0
    return np.array([cx, cy, w, h], box.dtype)


def xyxy2xywh(
    box: NDArray[DTypeLike],
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``NDArray[DTypeLike]``): ``[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> box = np.array([[100, 200], [100, 1500], [1400, 1500], [1400, 200]])
        >>> zz.vision.xyxy2xywh(box)
        array([ 750,  850, 1300, 1300])
        >>> boxes = np.array([[[100, 200], [100, 1500], [1400, 1500], [1400, 200]],
                              [[200, 300], [200, 1600], [1500, 1600], [1500, 300]]])
        >>> zz.vision.xyxy2xywh(boxes)
        array([[ 750,  850, 1300, 1300],
              [ 850,  950, 1300, 1300]])
    """
    shape = box.shape
    if len(shape) == 2:
        return _xyxy2xywh(box)
    elif len(shape) == 3:
        assert shape[1] == 4
        assert shape[2] == 2
        converted_boxes = np.zeros((shape[0], 4), box.dtype)
        for i, b in enumerate(box):
            converted_boxes[i] = _xyxy2xywh(b)
        return converted_boxes
    else:
        raise Exception("The 'box' must be of shape [4, 2] or [N, 4, 2]")


def xywh2xyxy(
    box: NDArray[DTypeLike],
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``NDArray[DTypeLike]``): ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`` 로 구성된 bbox (``[4, 2]``)

    Examples:
        >>> box = np.array([850, 800, 1300, 1400])
        >>> zz.vision.xywh2xyxy(box)
        array([[ 200.,  100.],
               [1500.,  100.],
               [1500., 1500.],
               [ 200., 1500.]])
    """
    cx, cy, w, h = box
    xa = cx - w / 2
    xb = cx + w / 2
    ya = cy - h / 2
    yb = cy + h / 2
    x1, y1 = xa, ya
    x2, y2 = xb, ya
    x3, y3 = xb, yb
    x4, y4 = xa, yb
    return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
