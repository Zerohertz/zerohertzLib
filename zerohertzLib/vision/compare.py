import math
from typing import List, Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray


def _rel2abs(x1: float, x2: float, y1: float, y2: float, w: int, h: int) -> List[int]:
    return [int(x1 * w / 100), int(x2 * w / 100), int(y1 * h / 100), int(y2 * h / 100)]


def before_after(
    before: NDArray[np.uint8],
    after: NDArray[np.uint8],
    area: Optional[List[Union[int, float]]] = None,
    per: Optional[bool] = True,
    quality: Optional[int] = 100,
    output_filename: Optional[str] = "tmp",
) -> None:
    """두 이미지를 비교하는 이미지 생성

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/282745206-fcddd725-e596-471d-b66c-27fe296ccb63.png
        :alt: Result
        :align: center

    Args:
        before (``NDArray[np.uint8]``): 원본 이미지
        after (``NDArray[np.uint8]``): 영상 처리 혹은 모델 추론 후 이미지
        area: (``Optional[List[Union[int, float]]]``): 비교할 좌표 (``[x1, x2, y1, y2]``)
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

        BGR, Resize:

        >>> before = cv2.imread("test.jpg")
        >>> after = cv2.resize(before, (100, 100))
        >>> zz.vision.before_after(before, after, [20, 40, 30, 60])
    """
    if area is None:
        if per:
            area = [0.0, 100.0, 0.0, 100.0]
        else:
            raise Exception("'area' not provided while 'per' is False")
    before_shape = before.shape
    if per:
        x1, x2, y1, y2 = _rel2abs(*area, *before_shape[:2])
    else:
        x1, x2, y1, y2 = area
    if len(before_shape) == 2:
        before = cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
        before_shape = before.shape
    after_shape = after.shape
    if len(after_shape) == 2:
        after = cv2.cvtColor(after, cv2.COLOR_GRAY2BGR)
        after_shape = after.shape
    if not before_shape == after_shape:
        after = cv2.resize(after, before_shape[:2][::-1])
        after_shape = after.shape
    before, after = before[x1:x2, y1:y2, :], after[x1:x2, y1:y2, :]
    before_shape = before.shape
    H, W, C = before_shape
    palette = np.zeros((H, 2 * W, C), dtype=np.uint8)
    palette[:, :W, :] = before
    palette[:, W:, :] = after
    palette = cv2.resize(palette, (0, 0), fx=quality / 100, fy=quality / 100)
    cv2.imwrite(f"{output_filename}.png", palette)


def grid(
    *imgs: List[NDArray[np.uint8]],
    size: int = 1000,
    output_filename: Optional[str] = "tmp",
) -> None:
    """여러 이미지를 입력받아 한 이미지로 병합

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/282752504-591cf407-c5bc-460b-99cf-0be569198855.png
        :alt: Result
        :align: center

    Args:
        *imgs (``List[NDArray[np.uint8]]``): 입력 이미지
        size: (``int``): 출력 이미지의 크기
        output_filename: (``Optional[str]``): 저장될 파일의 이름

    Returns:
        ``None``: 현재 directory에 바로 이미지 저장

    Examples:
        >>> tmp = cv2.imread("test.jpg")
        >>> imgs = [(tmp + np.random.rand(*tmp.shape)).astype(np.uint8) for _ in range(8)]
        >>> imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
        >>> zz.vision.grid(*imgs)
    """
    cnt = math.ceil(math.sqrt(len(imgs)))
    length = size // cnt
    size = int(length * cnt)
    palette = np.full((size, size, 3), 255, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        y, x = divmod(idx, cnt)
        x1, x2, y1, y2 = x * length, (x + 1) * length, y * length, (y + 1) * length
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        H, W, _ = img.shape
        if H > W:
            h, w = length, int(length / H * W)
            gap = (length - w) // 2
            x1, x2, y1, y2 = (
                x * length + gap,
                x * length + gap + w,
                y * length,
                (y + 1) * length,
            )
        elif W > H:
            h, w = int(length / W * H), length
            gap = (length - h) // 2
            x1, x2, y1, y2 = (
                x * length,
                (x + 1) * length,
                y * length + gap,
                y * length + gap + h,
            )
        else:
            h = w = length
            x1, x2, y1, y2 = x * length, (x + 1) * length, y * length, (y + 1) * length
        img = cv2.resize(img, (w, h))
        palette[y1:y2, x1:x2, :] = img
    cv2.imwrite(f"{output_filename}.png", palette)
