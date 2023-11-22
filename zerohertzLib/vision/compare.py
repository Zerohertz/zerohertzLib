import math
from typing import List, Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from .util import _cvtBGRA


def _rel2abs(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> List[int]:
    return [int(x0 * w / 100), int(y0 * h / 100), int(x1 * w / 100), int(y1 * h / 100)]


def before_after(
    before: NDArray[np.uint8],
    after: NDArray[np.uint8],
    area: Optional[List[Union[int, float]]] = None,
    per: Optional[bool] = True,
    quality: Optional[int] = 100,
    output_filename: Optional[str] = "tmp",
) -> None:
    """두 이미지를 비교하는 이미지 생성

    Args:
        before (``NDArray[np.uint8]``): 원본 이미지
        after (``NDArray[np.uint8]``): 영상 처리 혹은 모델 추론 후 이미지
        area: (``Optional[List[Union[int, float]]]``): 비교할 좌표 (``[x0, y0, x1, y1]``)
        per (``Optional[bool]``): ``area`` 의 백분율 여부
        quality (``Optional[int]``): 출력 이미지의 quality (단위: %)
        output_filename: (``Optional[str]``): 저장될 파일의 이름

    Returns:
        ``None``: 현재 directory에 바로 이미지 저장

    Examples:

        BGR, GRAY:

        >>> before = cv2.imread("test.jpg")
        >>> after = cv2.GaussianBlur(before, (0, 0), 25)
        >>> after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        >>> zz.vision.before_after(before, after, quality=10)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284503831-44cbe7a2-c1a2-4d44-91bf-7f35c2f80d2e.png
            :alt: Visualzation Result
            :align: center
            :width: 300px

        BGR, Resize:

        >>> before = cv2.imread("test.jpg")
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
        x0, y0, x1, y1 = _rel2abs(*area, *before_shape[:2])
    else:
        x0, y0, x1, y1 = area
    before = _cvtBGRA(before)
    before_shape = before.shape
    after = _cvtBGRA(after)
    after_shape = after.shape
    if not before_shape == after_shape:
        after = cv2.resize(after, before_shape[:2][::-1])
        after_shape = after.shape
    before, after = before[x0:x1, y0:y1, :], after[x0:x1, y0:y1, :]
    before_shape = before.shape
    H, W, C = before_shape
    palette = np.zeros((H, 2 * W, C), dtype=np.uint8)
    palette[:, :W, :] = before
    palette[:, W:, :] = after
    palette = cv2.resize(palette, (0, 0), fx=quality / 100, fy=quality / 100)
    cv2.imwrite(f"{output_filename}.png", palette)


def grid(
    *imgs: List[NDArray[np.uint8]],
    size: Optional[int] = 1000,
    output_filename: Optional[str] = "tmp",
) -> None:
    """여러 이미지를 입력받아 한 이미지로 병합

    Args:
        *imgs (``List[NDArray[np.uint8]]``): 입력 이미지
        size: (``Optional[int]``): 출력 이미지의 크기
        output_filename: (``Optional[str]``): 저장될 파일의 이름

    Returns:
        ``None``: 현재 directory에 바로 이미지 저장

    Examples:
        >>> tmp = cv2.imread("test.jpg")
        >>> imgs = [(tmp + np.random.rand(*tmp.shape)).astype(np.uint8) for _ in range(8)]
        >>> imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
        >>> zz.vision.grid(*imgs)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284504218-9859abdb-7fd3-47d1-a9c8-569f8c95d5b7.png
            :alt: Visualzation Result
            :align: center
            :width: 300px
    """
    cnt = math.ceil(math.sqrt(len(imgs)))
    length = size // cnt
    size = int(length * cnt)
    palette = np.full((size, size, 4), 255, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        y, x = divmod(idx, cnt)
        x0, y0, x1, y1 = x * length, y * length, (x + 1) * length, (y + 1) * length
        img = _cvtBGRA(img)
        H, W, _ = img.shape
        if H > W:
            h, w = length, int(length / H * W)
            gap = (length - w) // 2
            x0, y0, x1, y1 = (
                x * length + gap,
                y * length,
                x * length + gap + w,
                (y + 1) * length,
            )
        elif W > H:
            h, w = int(length / W * H), length
            gap = (length - h) // 2
            x0, y0, x1, y1 = (
                x * length,
                y * length + gap,
                (x + 1) * length,
                y * length + gap + h,
            )
        else:
            h = w = length
            x0, y0, x1, y1 = x * length, y * length, (x + 1) * length, (y + 1) * length
        img = cv2.resize(img, (w, h))
        palette[y0:y1, x0:x1, :] = img
    cv2.imwrite(f"{output_filename}.png", palette)


def vert(
    *imgs: List[NDArray[np.uint8]],
    height: int = 1000,
    output_filename: Optional[str] = "tmp",
):
    """여러 이미지를 입력받아 한 이미지로 병합

    Args:
        *imgs (``List[NDArray[np.uint8]]``): 입력 이미지
        height: (``Optional[int]``): 출력 이미지의 높이
        output_filename: (``Optional[str]``): 저장될 파일의 이름

    Returns:
        ``None``: 현재 directory에 바로 이미지 저장

    Examples:
        >>> tmp = cv2.imread("test.jpg")
        >>> imgs = [cv2.resize(tmp, (random.randrange(300, 600), random.randrange(300, 600))) for _ in range(5)]
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
        img = _cvtBGRA(img)
        if shape[0] != height:
            w = int(height / shape[0] * shape[1])
            img = cv2.resize(img, (w, height))
        else:
            w = shape[1]
        width += w
        resized_imgs.append(img)
    palette = np.full((height, width, 4), 255, dtype=np.uint8)
    width = 0
    for img in resized_imgs:
        h, w, _ = img.shape
        palette[:h, width : width + w, :] = img
        width += w
    cv2.imwrite(f"{output_filename}.png", palette)
