from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import DTypeLike, NDArray
from PIL import Image, ImageDraw, ImageFont

from .convert import xywh2xyxy, xyxy2xywh


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


def _bbox(
    img: NDArray[np.uint8],
    box_xyxy: NDArray[DTypeLike],
    color: Tuple[int],
    thickness: int,
) -> NDArray[np.uint8]:
    """Bbox 시각화

    Args:
        img (``NDArray[np.uint8]``): Input image (``[H, W, C]``)
        box_xyxy (``NDArray[DTypeLike]``): 하나의 bbox (``[4, 2]``)
        color (``Tuple[int]``): bbox의 색
        thickness (``int``): bbox 선의 두께

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, C]``)
    """
    return cv2.polylines(
        img,
        [box_xyxy.astype(np.int32)],
        isClosed=True,
        color=color,
        thickness=thickness,
    )


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
        xyxy = False
    elif len(shape) == 2:
        # N * [cx, cy, w, h]
        if shape[1] == 4:
            multi = True
            xyxy = False
        # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        elif shape[0] == 4 and shape[1] == 2:
            multi = False
            xyxy = True
        else:
            raise Exception("The 'box' must be of shape [4], [N, 4], [4, 2], [N, 4, 2]")
    # N * [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    elif len(shape) == 3 and shape[1] == 4 and shape[2] == 2:
        multi = True
        xyxy = True
    else:
        raise Exception("The 'box' must be of shape [4], [N, 4], [4, 2], [N, 4, 2]")
    return multi, xyxy


def bbox(
    img: NDArray[np.uint8],
    box: NDArray[DTypeLike],
    color: Optional[Tuple[int]] = (0, 0, 255),
    thickness: Optional[int] = 2,
) -> NDArray[np.uint8]:
    """여러 Bbox 시각화

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284457018-1bf97633-f650-4d28-b955-1212cdd243ef.png
        :alt: Visualzation Result
        :align: center

    Args:
        img (``NDArray[np.uint8]``): Input image (``[H, W, C]``)
        box (``NDArray[DTypeLike]``): 하나 혹은 여러 개의 bbox (``[4]``, ``[N, 4]``, ``[4, 2]``, ``[N, 4, 2]``)
        color (``Optional[Tuple[int]]``): bbox의 색
        thickness (``Optional[int]``): bbox 선의 두께

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, C]``)

    Examples:
        >>> img = cv2.imread("test.jpg")
        >>> box = np.array([[100, 200], [100, 1500], [1400, 1500], [1400, 200]])
        >>> box.shape
        (4, 2)
        >>> zz.vision.bbox(img, box, thickness=10)
        >>> boxes = np.array([[850, 800, 1300, 1400], [300, 300, 300, 400], [1200, 1200, 300, 300]])
        >>> boxes.shape
        (3, 4)
        >>> zz.vision.bbox(img, boxes, (0, 255, 0), thickness=10)
    """
    img = img.copy()
    shape = img.shape
    if len(shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif shape[2] == 4:
        color = (*color, 255)
    shape = box.shape
    multi, xyxy = _isBbox(shape)
    if not xyxy:
        box = xywh2xyxy(box)
    if multi:
        for b in box:
            img = _bbox(img, b, color, thickness)
    else:
        img = _bbox(img, box, color, thickness)
    return img


def masks(
    img: NDArray[np.uint8],
    mks: NDArray[bool],
    color: Optional[Tuple[int]] = (0, 0, 255),
    class_list: Optional[List[Union[int, str]]] = None,
    class_color: Optional[Dict[Union[int, str], Tuple[int]]] = None,
    border: Optional[bool] = True,
    alpha: Optional[float] = 0.5,
) -> NDArray[np.uint8]:
    """Masks 시각화

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/283127171-6f6c0b60-ca62-48b7-91f9-dba899275a72.png
        :alt: Visualzation Result
        :align: center

    Args:
        img (``NDArray[np.uint8]``): 입력 이미지 (``[H, W, C]``)
        mks (``NDArray[bool]``): 입력 이미지 위에 병합할 ``N`` 개의 mask들 (``[N, H, W]``)
        color (``Optional[Tuple[int]]``): Mask의 색
        class_list (``Optional[List[Union[int, str]]]``): ``mks`` 의 index에 따른 class
        class_color (``Optional[Dict[Union[int, str], Tuple[int]]]``): Class에 따른 색 (``color`` 무시)
        border (``Optional[bool]``): Mask의 경계선 표시 여부
        alpha (``Optional[float]``): Mask의 투명도

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, C]``)

    Examples:
        >>> img = cv2.imread("test.jpg")
        >>> H, W, _ = img.shape
        >>> cnt = 30
        >>> mks = np.zeros((cnt, H, W), np.uint8)
        >>> for mask in mks:
        >>>     center_x = random.randint(0, W)
        >>>     center_y = random.randint(0, H)
        >>>     radius = random.randint(100, 400)
        >>>     cv2.circle(mask, (center_x, center_y), radius, (True), -1)
        >>> mks = mks.astype(bool)
        >>> zz.vision.masks(img, mks)
    """
    shape = img.shape
    if len(shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif shape[2] == 4:
        color = (*color, 255)
    overlay = img.copy()
    cumulative_mask = np.zeros(img.shape[:2], dtype=bool)
    for idx, mask in enumerate(mks):
        if class_list is not None and class_color is not None:
            color = class_color[class_list[idx]]
        overlapping = cumulative_mask & mask
        non_overlapping = mask & ~cumulative_mask
        cumulative_mask |= mask
        if overlapping.any():
            overlapping_color = overlay[overlapping].astype(np.float32)
            mixed_color = ((overlapping_color + color) / 2).astype(np.uint8)
            overlay[overlapping] = mixed_color
        if non_overlapping.any():
            overlay[non_overlapping] = color
        if border:
            edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
            overlay[edges > 0] = color
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


def _make_text(txt: str, shape: Tuple[int], color: Tuple[int]) -> NDArray[np.uint8]:
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
        __file__.replace("vision/visual.py", "plot/NotoSansKR-Medium.ttf"), 100
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


def _text(
    img: NDArray[np.uint8], box_xywh: NDArray[DTypeLike], txt: str, color: Tuple[int]
) -> NDArray[np.uint8]:
    """단일 text 시각화

    Args:
        img (``NDArray[np.uint8]``): 입력 이미지 (``[H, W, C]``)
        box_xywh (``NDArray[DTypeLike]``): 문자열이 존재할 bbox (``[4, 2]``)
        txt (``str``): 이미지에 추가할 문자열
        color (``Tuple[int]``): 문자의 색

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, 4]``)
    """
    w0, h0 = (box_xywh[:2] - box_xywh[2:] / 2).astype(np.int32)
    w1, h1 = (box_xywh[:2] + box_xywh[2:] / 2).astype(np.int32)
    w, h = w1 - w0, h1 - h0
    txt = _make_text(txt, (h, w), color)
    img[h0:h1, w0:w1, :] = _paste(img[h0:h1, w0:w1, :], txt)
    return img


def text(
    img: NDArray[np.uint8],
    box: NDArray[DTypeLike],
    txt: Union[str, List[str]],
    color: Optional[Tuple[int]] = (0, 0, 0),
    vis: Optional[bool] = False,
) -> NDArray[np.uint8]:
    """Text 시각화

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284456581-a5ffb8e0-059d-4145-ace4-dbb073780612.png
        :alt: Visualzation Result
        :align: center

    Args:
        img (``NDArray[np.uint8]``): 입력 이미지 (``[H, W, C]``)
        box (``NDArray[DTypeLike]``): 문자열이 존재할 bbox (``[4]``, ``[N, 4]``, ``[4, 2]``, ``[N, 4, 2]``)
        txt (``str``): 이미지에 추가할 문자열
        color (``Optional[Tuple[int]]``): 문자의 색
        vis (``Optional[bool]``): 문자 영역의 시각화 여부

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, 4]``)

    Examples:
        >>> img = cv2.imread("test.jpg")
        >>> box = np.array([[100, 200], [100, 1500], [1400, 1500], [1400, 200]])
        >>> box.shape
        (4, 2)
        >>> zz.vision.text(img, box, "먼지야")
        >>> boxes = np.array([[850, 800, 1300, 1400], [300, 300, 300, 400], [1200, 1200, 300, 300]])
        >>> boxes.shape
        (3, 4)
        >>> zz.vision.text(img, boxes, ["먼지야", "먼지야", "먼지야"], vis=True)
    """
    img = img.copy()
    img = _cvtBGRA(img)
    shape = box.shape
    multi, xyxy = _isBbox(shape)
    if xyxy:
        box_xyxy = box
        box_xywh = xyxy2xywh(box)
    else:
        box_xyxy = xywh2xyxy(box)
        box_xywh = box
    if multi:
        if not shape[0] == len(txt):
            raise Exception("'box.shape[0]' and 'len(txt)' must be equal")
        for b_xyxy, b_xywh, t in zip(box_xyxy, box_xywh, txt):
            img = _text(img, b_xywh, t, color)
            if vis:
                img = _bbox(img, b_xyxy, (0, 0, 255, 255), 2)
    else:
        img = _text(img, box_xywh, txt, color)
        if vis:
            img = _bbox(img, box_xyxy, (0, 0, 255, 255), 2)
    return img
