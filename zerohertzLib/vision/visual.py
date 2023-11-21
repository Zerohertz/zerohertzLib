from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import DTypeLike, NDArray
from PIL import Image, ImageDraw, ImageFont

from .convert import _isBbox, cwh2poly, poly2cwh, poly2mask


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


def bbox(
    img: NDArray[np.uint8],
    box: NDArray[DTypeLike],
    color: Optional[Tuple[int]] = (0, 0, 255),
    thickness: Optional[int] = 2,
) -> NDArray[np.uint8]:
    """여러 Bbox 시각화

    Args:
        img (``NDArray[np.uint8]``): Input image (``[H, W, C]``)
        box (``NDArray[DTypeLike]``): 하나 혹은 여러 개의 bbox (``[4]``, ``[N, 4]``, ``[4, 2]``, ``[N, 4, 2]``)
        color (``Optional[Tuple[int]]``): bbox의 색
        thickness (``Optional[int]``): bbox 선의 두께

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, C]``)

    Examples:
        >>> img = cv2.imread("test.jpg")
        >>> box = np.array([[100, 200], [100, 1000], [1200, 1000], [1200, 200]])
        >>> box.shape
        (4, 2)
        >>> zz.vision.bbox(img, box, thickness=10)
        >>> boxes = np.array([[250, 200, 100, 100], [600, 600, 800, 200], [900, 300, 300, 400]])
        >>> boxes.shape
        (3, 4)
        >>> zz.vision.bbox(img, boxes, (0, 255, 0), thickness=10)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284566751-ec443fc2-6b71-4ba3-a770-590fa873e944.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    img = img.copy()
    shape = img.shape
    if len(shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif shape[2] == 4:
        color = (*color, 255)
    shape = box.shape
    multi, poly = _isBbox(shape)
    if not poly:
        box = cwh2poly(box)
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
        >>>     radius = random.randint(30, 200)
        >>>     cv2.circle(mask, (center_x, center_y), radius, (True), -1)
        >>> mks = mks.astype(bool)
        >>> zz.vision.masks(img, mks)
        >>> cls = [i for i in range(cnt)]
        >>> class_list = [cls[random.randint(0, 2)] for _ in range(cnt)]
        >>> class_color = {}
        >>> for c in cls:
        >>>     class_color[c] = [random.randint(0, 255) for _ in range(3)]
        >>> zz.vision.masks(img, mks, class_list=class_list, class_color=class_color)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284525631-53c2be9d-e7a6-45eb-bf5e-a08a6bb09e1c.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
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
    x0, x1 = x, x + text_width
    y0, y1 = y, y + text_height
    draw.text((x, y), txt, font=font, fill=(*color, 255))
    palette = np.array(palette)[y0:y1, x0:x1, :]
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
    x0, y0 = (box_xywh[:2] - box_xywh[2:] / 2).astype(np.int32)
    x1, y1 = (box_xywh[:2] + box_xywh[2:] / 2).astype(np.int32)
    w, h = x1 - x0, y1 - y0
    txt = _make_text(txt, (h, w), color)
    img[y0:y1, x0:x1, :] = _paste(img[y0:y1, x0:x1, :], txt)
    return img


def text(
    img: NDArray[np.uint8],
    box: NDArray[DTypeLike],
    txt: Union[str, List[str]],
    color: Optional[Tuple[int]] = (0, 0, 0),
    vis: Optional[bool] = False,
) -> NDArray[np.uint8]:
    """Text 시각화

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
        >>> box = np.array([[100, 200], [100, 1000], [1200, 1000], [1200, 200]])
        >>> box.shape
        (4, 2)
        >>> zz.vision.text(img, box, "먼지야")
        >>> boxes = np.array([[250, 200, 100, 100], [600, 600, 800, 200], [900, 300, 300, 400]])
        >>> boxes.shape
        (3, 4)
        >>> zz.vision.text(img, boxes, ["먼지야", "먼지야", "먼지야"], vis=True)

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284566305-fe9d1be6-b506-4140-bca9-db2a210f333c.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    img = img.copy()
    img = _cvtBGRA(img)
    shape = box.shape
    multi, poly = _isBbox(shape)
    if poly:
        box_xyxy = box
        box_xywh = poly2cwh(box)
    else:
        box_xyxy = cwh2poly(box)
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


def cutout(
    img: NDArray[np.uint8],
    poly: NDArray[DTypeLike],
    alpha: Optional[int] = 255,
    crop: Optional[bool] = True,
) -> NDArray[np.uint8]:
    """이미지 내에서 지정한 좌표를 제외한 부분을 투명화

    Args:
        img (``NDArray[np.uint8]``): 입력 이미지 (``[H, W, C]``)
        poly (``NDArray[DTypeLike]``): 지정할 좌표 (``[N, 2]``)
        alpha (``Optional[int]``): 지정한 좌표 영역의 투명도
        crop (``Optional[bool]``): 출력 이미지의 Crop 여부

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, 4]``)

    Examples:
        >>> img = cv2.imread("test.jpg")
        >>> poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
        >>> zz.vision.cutout(img, poly)
        >>> zz.vision.cutout(img, poly, 128, False)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284564607-9df73033-f99e-4b54-a321-2a47a6c89a32.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    shape = img.shape[:2]
    poly = poly.astype(np.int32)
    x0, x1 = poly[:, 0].min(), poly[:, 0].max()
    y0, y1 = poly[:, 1].min(), poly[:, 1].max()
    mask = poly2mask(poly, shape)
    mask = (mask * alpha).astype(np.uint8)
    img = Image.fromarray(img)
    mask = Image.fromarray(mask)
    img.putalpha(mask)
    if crop:
        return np.array(img)[y0:y1, x0:x1, :]
    else:
        return np.array(img)


def paste(
    img: NDArray[np.uint8],
    target: NDArray[np.uint8],
    box: List[int],
    resize: Optional[bool] = False,
    vis: Optional[bool] = False,
) -> NDArray[np.uint8]:
    """``target`` 이미지를 ``img`` 위에 투명도를 포함하여 병합

    Note:
        `PIL.Image.paste` 를 `numpy` 와 `cv2` 기반으로 구현

        >>> img = Image.open("test.png").convert("RGBA")
        >>> target = Image.open("target.png").convert("RGBA")
        >>> img.paste(target, (0, 0), target)

    Args:
        img (``NDArray[np.uint8]``): 입력 이미지 (``[H, W, C]``)
        target (``NDArray[np.uint8]``): 타겟 이미지 (``[H, W, 4]``)
        box (``List[int]``): 병합될 영역
        resize (``Optional[bool]``): 타겟 이미지의 resize 여부
        vis (``Optional[bool]``): 지정한 영역 (``box``)의 시각화 여부

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, 4]``)

    Examples:
        >>> img = cv2.imread("test.jpg")
        >>> poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
        >>> target = zz.vision.cutout(img, poly, 200)
        >>> zz.vision.paste(img, target, [200, 200, 1000, 800], False, True)
        >>> zz.vision.paste(img, target, [200, 200, 1000, 800], True, True)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284563666-2b7f33a9-5c1d-454e-afd8-9b4d3606b4e4.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    x0, y0, x1, y1 = box
    H, W = y1 - y0, x1 - x0
    img = img.copy()
    img = _cvtBGRA(img)
    h, w = target.shape[:2]
    if resize:
        target = cv2.resize(target, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        if w / h > W / H:
            target = cv2.resize(
                target, (W, int(h * W / w)), interpolation=cv2.INTER_LINEAR
            )
        elif w / h < W / H:
            target = cv2.resize(
                target, (int(w * H / h), H), interpolation=cv2.INTER_LINEAR
            )
        else:
            target = cv2.resize(target, (W, H), interpolation=cv2.INTER_LINEAR)
        h, w, _ = target.shape
        top, bottom = (H - h) // 2, (H - h) // 2 + (H - h) % 2
        left, right = (W - w) // 2, (W - w) // 2 + (W - w) % 2
        target = np.pad(
            target,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=((0, 0), (0, 0), (0, 0)),
        )
    img[y0:y1, x0:x1, :] = _paste(img[y0:y1, x0:x1, :], target)
    if vis:
        box = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
        img = _bbox(img, box, (0, 0, 255, 255), 2)
    return img
