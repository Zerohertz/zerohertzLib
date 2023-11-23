"""
.. admonition:: Vision
    :class: hint

    다양한 image들을 handling하고 시각화하는 함수 및 class들

.. important::

    Bbox의 types

    - ``cwh``: ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)
    - ``xyxy``: ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)
    - ``poly``: ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)
"""

from zerohertzLib.vision.compare import before_after, grid, vert
from zerohertzLib.vision.convert import (
    cwh2poly,
    cwh2xyxy,
    poly2cwh,
    poly2mask,
    poly2xyxy,
    xyxy2cwh,
    xyxy2poly,
)
from zerohertzLib.vision.data import JsonImageLoader
from zerohertzLib.vision.gif import img2gif, vid2gif
from zerohertzLib.vision.util import is_pts_in_poly
from zerohertzLib.vision.visual import bbox, cutout, masks, pad, paste, text

__all__ = [
    "img2gif",
    "vid2gif",
    "before_after",
    "grid",
    "bbox",
    "masks",
    "text",
    "cwh2poly",
    "cwh2xyxy",
    "poly2cwh",
    "poly2mask",
    "poly2xyxy",
    "xyxy2cwh",
    "xyxy2poly",
    "cutout",
    "paste",
    "is_pts_in_poly",
    "JsonImageLoader",
    "vert",
    "pad",
]
