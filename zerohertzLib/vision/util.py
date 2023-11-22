from typing import Tuple


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
