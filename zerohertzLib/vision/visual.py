from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import DTypeLike, NDArray
from PIL import Image, ImageDraw, ImageFont


def _cvtBGRA(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """cv2로 읽어온 이미지를 BGRA 채널로 전환

    Args:
        img (``NDArray[np.uint8]``): 입력 이미지 (``[H, W, C]``)

    Returns:
        ``NDArray[np.uint8]``: BGRA 이미지 (``[H, W, 4]``)
    """
    shape = img.shape
    if len(shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        return img


def bbox(
    img: NDArray[np.uint8],
    box: NDArray[DTypeLike],
    color: Tuple[int] = (0, 0, 255),
    thickness: int = 2,
) -> NDArray[np.uint8]:
    """Bbox 시각화

    Args:
        img (``NDArray[np.uint8]``): Input image (``[H, W, C]``)
        box (``NDArray[DTypeLike]``): 하나의 bbox (``[4, 2]``)
        color (``Tuple[int]``): bbox의 색
        thickness (``int``): bbox 선의 두께

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, C]``)
    """
    img = img.copy()
    shape = img.shape
    if len(shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif shape[2] == 4:
        color = (*color, 255)
    return cv2.polylines(
        img, [box.astype(np.int32)], isClosed=True, color=color, thickness=thickness
    )


def masks(
    img: NDArray[np.uint8],
    mks: NDArray[bool],
    color: Optional[Tuple[int]] = (0, 0, 255),
    class_list: Optional[List[Union[int, str]]] = None,
    class_color: Optional[Dict[Union[int, str], Tuple[int]]] = None,
    alpha: Optional[float] = 0.5,
) -> NDArray[np.uint8]:
    """Masks 시각화

    Args:
        img (``NDArray[np.uint8]``): 입력 이미지 (``[H, W, C]``)
        mks (``NDArray[bool]``): 입력 이미지 위에 병합할 ``N`` 개의 mask들 (``[N, H, W]``)
        color (``Optional[Tuple[int]]``): Mask의 색
        class_list (``Optional[List[Union[int, str]]]``): ``mks`` 의 index에 따른 class
        class_color (``Optional[Dict[Union[int, str], Tuple[int]]]``): Class에 따른 색 (``color`` 무시)
        alpha (``Optional[float]``): Mask의 투명도

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, C]``)
    """
    shape = img.shape
    if len(shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif shape[2] == 4:
        color = (*color, 255)
    overlay = img.copy()
    if class_list is not None and class_color is not None:
        assert mks.shape[0] == len(class_list)
        for mask, cls in zip(mks, class_list):
            overlay[mask] = class_color[cls]
    else:
        for mask in mks:
            overlay[mask] = color
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


def _paste(img: NDArray[np.uint8], target: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """``target`` 이미지를 ``img`` 위에 투명도를 포함하여 병합

    Args:
        img (``NDArray[np.uint8]``): 입력 이미지 (``[H, W, 4]``)
        target (``NDArray[np.uint8]``): 타겟 이미지 (``[H, W, 4]``)

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, 4]``)
    """
    alpha_overlay = target[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay
    for c in range(0, 3):
        img[:, :, c] = alpha_overlay * target[:, :, c] + alpha_background * img[:, :, c]
    return img


def _make_text(txt: str, shape: Tuple[int], color: Tuple[int]):
    """배경이 투명한 문자열 이미지 생성

    Args:
        txt (``str``): 입력 문자열
        shape (``Tuple[int]``): 출력 이미지의 shape
        color (``Tuple[int]``): 글씨의 색

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, 4]``)
    """
    size = (1000, 1000)
    palette = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(palette)
    font = ImageFont.truetype(
        __file__.replace("vision/visual.py", "plot/NotoSansKR-Medium.ttf"), 30
    )
    text_width, text_height = draw.textsize(txt, font=font)
    x, y = (size[0] - text_width) // 2, (size[1] - text_height) // 2
    w0, w1 = x, x + text_width
    h0, h1 = y, y + text_height
    draw.text((x, y), txt, font=font, fill=(*color, 255))
    palette = np.array(palette)[h0:h1, w0:w1, :]
    h, w, _ = palette.shape
    H, W = shape
    if w / h > W / H:
        palette = cv2.resize(
            palette, (W, int(h * W / w)), interpolation=cv2.INTER_LINEAR
        )
    elif w / h < W / H:
        palette = cv2.resize(
            palette, (int(w * H / h), H), interpolation=cv2.INTER_LINEAR
        )
    else:
        palette = cv2.resize(palette, (W, H), interpolation=cv2.INTER_LINEAR)
    h, w, _ = palette.shape
    top, bottom = (H - h) // 2, (H - h) // 2 + (H - h) % 2
    left, right = (W - w) // 2, (W - w) // 2 + (W - w) % 2
    palette = np.pad(
        palette,
        ((top, bottom), (left, right), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    return palette


def text(
    img: NDArray[np.uint8],
    box: NDArray[DTypeLike],
    txt: str,
    color: Optional[Tuple[int]] = (0, 0, 0),
) -> NDArray[np.uint8]:
    """Text 시각화

    Args:
        img (``NDArray[np.uint8]``): 입력 이미지 (``[H, W, C]``)
        box (``NDArray[DTypeLike]``): 문자열이 존재할 bbox (``[4, 2]``)
        txt (``str``): 이미지에 추가할 문자열
        color (``Optional[Tuple[int]``): bbox의 색

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, 4]``)
    """
    img = img.copy()
    img = _cvtBGRA(img)
    w0, w1 = map(int, (min(box[:, 0]), max(box[:, 0])))
    h0, h1 = map(int, (min(box[:, 1]), max(box[:, 1])))
    w, h = w1 - w0, h1 - h0
    txt = _make_text(txt, (h, w), color)
    img[h0:h1, w0:w1, :] = _paste(img[h0:h1, w0:w1, :], txt)
    return img
