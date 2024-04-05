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

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import DTypeLike, NDArray
from PIL import Image

from .convert import _list2np, poly2mask
from .util import _cvt_bgra


def pad(
    img: NDArray[np.uint8],
    shape: Tuple[int],
    color: Optional[Tuple[int]] = (255, 255, 255),
    poly: Optional[NDArray[DTypeLike]] = None,
) -> Tuple[NDArray[np.uint8], Union[Tuple[float, int, int], NDArray[DTypeLike]]]:
    """입력 image를 원하는 shape로 resize 및 pad

    Args:
        img (``NDArray[np.uint8]``): 입력 image (``[H, W, C]``)
        shape (``Tuple[int]``): 출력의 shape ``(H, W)``
        color (``Optional[Tuple[int]]``): Padding의 색
        poly (``Optional[NDArray[DTypeLike]]``): Padding에 따라 변형될 좌표 (``[N, 2]``)

    Returns:
        ``Tuple[NDArray[np.uint8], Union[Tuple[float, int, int], NDArray[DTypeLike]]]``: 출력 image (``[H, W, C]``) 및 padding에 따른 정보 또는 변형된 좌표값

    Note:
        ``poly`` 를 입력하지 않을 시 ``(ratio, left, top)`` 가 출력되며 ``poly * ratio + (left, top)`` 와 같이 차후에 변환 가능

    Examples:
        GRAY:
            >>> img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            >>> res1 = cv2.resize(img, (500, 1000))
            >>> res1, _ = zz.vision.pad(res1, (1000, 1000), color=(0, 255, 0))

        BGR:
            >>> res2 = cv2.resize(img, (1000, 500))
            >>> res2, _ = zz.vision.pad(res2, (1000, 1000))

        BGRA:
            >>> img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            >>> res3 = cv2.resize(img, (500, 1000))
            >>> res3, _ = zz.vision.pad(res3, (1000, 1000), color=(0, 0, 255, 128))

        Poly:
            >>> poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
            >>> res4 = cv2.resize(img, (2000, 1000))
            >>> res4 = zz.vision.bbox(res4, poly, color=(255, 0, 0), thickness=20)
            >>> res4, poly = zz.vision.pad(res4, (1000, 1000), poly=poly)
            >>> res4 = zz.vision.bbox(res4, poly, color=(0, 0, 255))

        Transformation:
            >>> poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
            >>> res5 = cv2.resize(img, (2000, 1000))
            >>> res5 = zz.vision.bbox(res5, poly, color=(255, 0, 0), thickness=20)
            >>> res5, info = zz.vision.pad(res5, (1000, 1000), color=(128, 128, 128))
            >>> poly = poly * info[0] + info[1:]
            >>> res5 = zz.vision.bbox(res5, poly, color=(0, 0, 255))

        .. image:: _static/examples/dynamic/vision.pad.png
            :align: center
            :width: 700px
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4 and len(color) == 3:
        color = [*color, 255]
    img_height, img_width = img.shape[:2]
    tar_height, tar_width = shape
    if img_width / img_height > tar_width / tar_height:
        ratio = tar_width / img_width
        resize_width, resize_height = tar_width, int(img_height * ratio)
    elif img_width / img_height < tar_width / tar_height:
        ratio = tar_height / img_height
        resize_width, resize_height = int(img_width * ratio), tar_height
    else:
        ratio = 1
        (
            resize_width,
            resize_height,
        ) = (
            tar_width,
            tar_height,
        )
    img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    top, bottom = (
        (tar_height - resize_height) // 2,
        (tar_height - resize_height) // 2 + (tar_height - resize_height) % 2,
    )
    left, right = (
        (tar_width - resize_width) // 2,
        (tar_width - resize_width) // 2 + (tar_width - resize_width) % 2,
    )
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    if poly is None:
        return img, (ratio, left, top)
    return img, poly * ratio + (left, top)


def cutout(
    img: NDArray[np.uint8],
    poly: Union[List[Union[int, float]], NDArray[DTypeLike]],
    alpha: Optional[int] = 255,
    crop: Optional[bool] = True,
    background: Optional[int] = 0,
) -> NDArray[np.uint8]:
    """Image 내에서 지정한 좌표를 제외한 부분을 투명화

    Args:
        img (``NDArray[np.uint8]``): 입력 image (``[H, W, C]``)
        poly (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): 지정할 좌표 (``[N, 2]``)
        alpha (``Optional[int]``): 지정한 좌표 영역의 투명도
        crop (``Optional[bool]``): 출력 image의 Crop 여부
        background (``Optional[int]``): 지정한 좌표 외 배경의 투명도

    Returns:
        ``NDArray[np.uint8]``: 출력 image (``[H, W, 4]``)

    Examples:
        >>> poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
        >>> res1 = zz.vision.cutout(img, poly)
        >>> res2 = zz.vision.cutout(img, poly, 128, False)
        >>> res3 = zz.vision.cutout(img, poly, background=128)

        .. image:: _static/examples/dynamic/vision.cutout.png
            :align: center
            :width: 600px
    """
    shape = img.shape[:2]
    poly = _list2np(poly)
    poly = poly.astype(np.int32)
    x_0, x_1 = poly[:, 0].min(), poly[:, 0].max()
    y_0, y_1 = poly[:, 1].min(), poly[:, 1].max()
    mask = poly2mask(poly, shape)
    if background == 0:
        mask = (mask * alpha).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
        mask[mask == 0] = background
        mask[mask == 1] = alpha
    img = Image.fromarray(img)
    mask = Image.fromarray(mask)
    img.putalpha(mask)
    if crop:
        return np.array(img)[y_0:y_1, x_0:x_1, :]
    return np.array(img)


def transparent(
    img: NDArray[np.uint8],
    threshold: Optional[int] = 128,
    reverse: Optional[bool] = False,
) -> NDArray[np.uint8]:
    """입력 image에 대해 ``threshold`` 미만의 pixel들을 투명화

    Args:
        img (``NDArray[np.uint8]``): 입력 image (``[H, W, C]``)
        threshold (``Optional[int]``): Threshold
        reverse (``Optional[bool]``): ``threshold`` 이상의 pixel 투명화 여부

    Returns:
        ``NDArray[np.uint8]``: 출력 image (``[H, W, 4]``)

    Examples:
        >>> res1 = zz.vision.transparent(img)
        >>> res2 = zz.vision.transparent(img, reverse=True)

        .. image:: _static/examples/dynamic/vision.transparent.png
            :align: center
            :width: 600px
    """
    img = img.copy()
    img = _cvt_bgra(img)
    img_alpha = img[:, :, 3]
    img_bin = threshold > cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if reverse:
        img_alpha[~img_bin] = 0
    else:
        img_alpha[img_bin] = 0
    return img
