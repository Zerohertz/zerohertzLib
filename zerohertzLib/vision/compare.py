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

import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from .util import _cvt_bgra
from .visual import pad


def _rel2abs(
    x_0: float, y_0: float, x_1: float, y_1: float, width: int, height: int
) -> List[int]:
    return [
        int(x_0 * width / 100),
        int(y_0 * height / 100),
        int(x_1 * width / 100),
        int(y_1 * height / 100),
    ]


def before_after(
    before: NDArray[np.uint8],
    after: NDArray[np.uint8],
    area: Optional[List[Union[int, float]]] = None,
    per: Optional[bool] = True,
    quality: Optional[int] = 100,
    filename: Optional[str] = "tmp",
) -> None:
    """두 image를 비교하는 image 생성

    Args:
        before (``NDArray[np.uint8]``): 원본 image
        after (``NDArray[np.uint8]``): 영상 처리 혹은 모델 추론 후 image
        area: (``Optional[List[Union[int, float]]]``): 비교할 좌표 (``[x_0, y_0, x_1, y_1]``)
        per (``Optional[bool]``): ``area`` 의 백분율 여부
        quality (``Optional[int]``): 출력 image의 quality (단위: %)
        filename: (``Optional[str]``): 저장될 file의 이름

    Returns:
        ``None``: 현재 directory에 바로 image 저장

    Examples:

        BGR, GRAY:

        >>> after = cv2.GaussianBlur(before, (0, 0), 25)
        >>> after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        >>> zz.vision.before_after(before, after, quality=10)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284503831-44cbe7a2-c1a2-4d44-91bf-7f35c2f80d2e.png
            :alt: Visualzation Result
            :align: center
            :width: 300px

        BGR, Resize:

        >>> after = cv2.resize(before, (100, 100))
        >>> zz.vision.before_after(before, after, [20, 40, 30, 60])

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284503976-789c6f8d-1b98-4941-b528-523b2973e4b4.png
            :alt: Visualzation Result
            :align: center
            :width: 300px
    """
    before_shape = before.shape
    if area is None:
        if per:
            area = [0.0, 0.0, 100.0, 100.0]
        else:
            raise ValueError("'area' not provided while 'per' is False")
    if per:
        x_0, y_0, x_1, y_1 = _rel2abs(*area, *before_shape[:2])
    else:
        x_0, y_0, x_1, y_1 = area
    before = _cvt_bgra(before)
    before_shape = before.shape
    after = _cvt_bgra(after)
    after_shape = after.shape
    if not before_shape == after_shape:
        after = cv2.resize(after, before_shape[:2][::-1])
        after_shape = after.shape
    before, after = before[x_0:x_1, y_0:y_1, :], after[x_0:x_1, y_0:y_1, :]
    before_shape = before.shape
    height, width, channel = before_shape
    palette = np.zeros((height, 2 * width, channel), dtype=np.uint8)
    palette[:, :width, :] = before
    palette[:, width:, :] = after
    palette = cv2.resize(palette, (0, 0), fx=quality / 100, fy=quality / 100)
    cv2.imwrite(f"{filename}.png", palette)


def grid(
    *imgs: List[NDArray[np.uint8]],
    size: Optional[int] = 1000,
    color: Optional[Tuple[int]] = (255, 255, 255),
    filename: Optional[str] = "tmp",
) -> None:
    """여러 image를 입력받아 한 정방형 image로 병합

    Args:
        *imgs (``List[NDArray[np.uint8]]``): 입력 image
        size: (``Optional[int]``): 출력 image의 크기
        color: (``Optional[Tuple[int]]``): Padding의 색
        filename: (``Optional[str]``): 저장될 file의 이름

    Returns:
        ``None``: 현재 directory에 바로 image 저장

    Examples:
        >>> imgs = [cv2.resize(img, (random.randrange(300, 1000), random.randrange(300, 1000))) for _ in range(8)]
        >>> imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
        >>> imgs[3] = cv2.cvtColor(imgs[3], cv2.COLOR_BGR2BGRA)
        >>> zz.vision.grid(*imgs)
        >>> zz.vision.grid(*imgs, color=(0, 255, 0))
        >>> zz.vision.grid(*imgs, color=(0, 0, 0, 0))

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/285098735-3b259a4b-3b26-4d50-9cec-8ef8458bf5b5.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    cnt = math.ceil(math.sqrt(len(imgs)))
    length = size // cnt
    size = int(length * cnt)
    palette = np.full((size, size, 4), 0, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        d_y, d_x = divmod(idx, cnt)
        x_0, y_0, x_1, y_1 = (
            d_x * length,
            d_y * length,
            (d_x + 1) * length,
            (d_y + 1) * length,
        )
        img = _cvt_bgra(img)
        palette[y_0:y_1, x_0:x_1, :] = pad(img, (length, length), color)
    cv2.imwrite(f"{filename}.png", palette)


def vert(
    *imgs: List[NDArray[np.uint8]],
    height: int = 1000,
    filename: Optional[str] = "tmp",
):
    """여러 image를 입력받아 한 가로 image로 병합

    Args:
        *imgs (``List[NDArray[np.uint8]]``): 입력 image
        height: (``Optional[int]``): 출력 image의 높이
        filename: (``Optional[str]``): 저장될 file의 이름

    Returns:
        ``None``: 현재 directory에 바로 image 저장

    Examples:
        >>> imgs = [cv2.resize(img, (random.randrange(300, 600), random.randrange(300, 600))) for _ in range(5)]
        >>> zz.vision.vert(*imgs)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284879452-d856fa8c-49a9-4a64-83b9-b27ae4f45007.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    resized_imgs = []
    width = 0
    for img in imgs:
        shape = img.shape
        img = _cvt_bgra(img)
        if shape[0] != height:
            tar_width = int(height / shape[0] * shape[1])
            img = cv2.resize(img, (tar_width, height))
        else:
            tar_width = shape[1]
        width += tar_width
        resized_imgs.append(img)
    palette = np.full((height, width, 4), 255, dtype=np.uint8)
    width = 0
    for img in resized_imgs:
        img_height, img_width, _ = img.shape
        palette[:img_height, width : width + img_width, :] = img
        width += img_width
    cv2.imwrite(f"{filename}.png", palette)
