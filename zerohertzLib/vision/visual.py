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

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import DTypeLike, NDArray
from PIL import Image, ImageDraw, ImageFont

from .convert import _list2np, cwh2poly, cwh2xyxy, poly2cwh, poly2mask
from .transform import pad
from .util import _cvt_bgra, _is_bbox


def _bbox(
    img: NDArray[np.uint8],
    box_poly: NDArray[DTypeLike],
    color: Tuple[int],
    thickness: int,
) -> NDArray[np.uint8]:
    """Bbox 시각화

    Args:
        img (``NDArray[np.uint8]``): Input image (``[H, W, C]``)
        box_poly (``NDArray[DTypeLike]``): 하나의 bbox (``[4, 2]``)
        color (``Tuple[int]``): bbox의 색
        thickness (``int``): bbox 선의 두께

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, C]``)
    """
    return cv2.polylines(
        img,
        [box_poly.astype(np.int32)],
        isClosed=True,
        color=color,
        thickness=thickness,
    )


def bbox(
    img: NDArray[np.uint8],
    box: Union[List[Union[int, float]], NDArray[DTypeLike]],
    color: Optional[Tuple[int]] = (0, 0, 255),
    thickness: Optional[int] = 2,
) -> NDArray[np.uint8]:
    """여러 Bbox 시각화

    Args:
        img (``NDArray[np.uint8]``): Input image (``[H, W, C]``)
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): 하나 혹은 여러 개의 bbox (``[4]``, ``[N, 4]``, ``[4, 2]``, ``[N, 4, 2]``)
        color (``Optional[Tuple[int]]``): bbox의 색
        thickness (``Optional[int]``): bbox 선의 두께

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, C]``)

    Examples:
        Bbox:

        >>> box = np.array([[100, 200], [100, 1000], [1200, 1000], [1200, 200]])
        >>> box.shape
        (4, 2)
        >>> res1 = zz.vision.bbox(img, box, thickness=10)

        Bboxes:

        >>> boxes = np.array([[250, 200, 100, 100], [600, 600, 800, 200], [900, 300, 300, 400]])
        >>> boxes.shape
        (3, 4)
        >>> res2 = zz.vision.bbox(img, boxes, (0, 255, 0), thickness=10)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284566751-ec443fc2-6b71-4ba3-a770-590fa873e944.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    box = _list2np(box)
    img = img.copy()
    shape = img.shape
    if len(shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif shape[2] == 4 and len(color) == 3:
        color = (*color, 255)
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if not poly:
        box = cwh2poly(box)
    if multi:
        for box_ in box:
            img = _bbox(img, box_, color, thickness)
    else:
        img = _bbox(img, box, color, thickness)
    return img


def masks(
    img: NDArray[np.uint8],
    mks: Optional[NDArray[bool]] = None,
    poly: Optional[NDArray[DTypeLike]] = None,
    color: Optional[Tuple[int]] = (0, 0, 255),
    class_list: Optional[List[Union[int, str]]] = None,
    class_color: Optional[Dict[Union[int, str], Tuple[int]]] = None,
    border: Optional[bool] = True,
    alpha: Optional[float] = 0.5,
) -> NDArray[np.uint8]:
    """Masks 시각화

    Args:
        img (``NDArray[np.uint8]``): 입력 image (``[H, W, C]``)
        mks (``Optional[NDArray[bool]]``): 입력 image 위에 병합할 mask (``[H, W]`` or ``[N, H, W]``)
        poly (``Optional[NDArray[DTypeLike]]``): 입력 image 위에 병합할 mask (``[N, 2]``)
        color (``Optional[Tuple[int]]``): Mask의 색
        class_list (``Optional[List[Union[int, str]]]``): ``mks`` 의 index에 따른 class
        class_color (``Optional[Dict[Union[int, str], Tuple[int]]]``): Class에 따른 색 (``color`` 무시)
        border (``Optional[bool]``): Mask의 경계선 표시 여부
        alpha (``Optional[float]``): Mask의 투명도

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, C]``)

    Examples:
        Mask (without class):

        >>> H, W, _ = img.shape
        >>> cnt = 30
        >>> mks = np.zeros((cnt, H, W), np.uint8)
        >>> for mask in mks:
        >>>     center_x = random.randint(0, W)
        >>>     center_y = random.randint(0, H)
        >>>     radius = random.randint(30, 200)
        >>>     cv2.circle(mask, (center_x, center_y), radius, (True), -1)
        >>> mks = mks.astype(bool)
        >>> res1 = zz.vision.masks(img, mks)

        Mask (with class):

        >>> cls = [i for i in range(cnt)]
        >>> class_list = [cls[random.randint(0, 2)] for _ in range(cnt)]
        >>> class_color = {}
        >>> for c in cls:
        >>>     class_color[c] = [random.randint(0, 255) for _ in range(3)]
        >>> res2 = zz.vision.masks(img, mks, class_list=class_list, class_color=class_color)

        Poly:

        >>> poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
        >>> res3 = zz.vision.masks(img, poly=poly)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284878547-c36cd4ff-2b36-4b0f-a125-89ed8380a456.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    shape = img.shape
    if len(shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif shape[2] == 4 and len(color) == 3:
        color = (*color, 255)
        if class_list is not None and class_color is not None:
            for key, value in class_color.items():
                if len(value) == 3:
                    class_color[key] = [*value, 255]
    if poly is not None:
        mks = poly2mask(poly, (shape[:2]))
    shape = mks.shape
    overlay = img.copy()
    cumulative_mask = np.zeros(img.shape[:2], dtype=bool)
    if len(shape) == 2:
        overlay[mks] = color
        if border:
            edges = cv2.Canny(mks.astype(np.uint8) * 255, 100, 200)
            overlay[edges > 0] = color
    elif len(shape) == 3:
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
    else:
        raise ValueError("The 'mks' must be of shape [H, W] or [N, H, W]")
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


def _paste(img: NDArray[np.uint8], target: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """``target`` image를 ``img`` 위에 투명도를 포함하여 병합

    Args:
        img (``NDArray[np.uint8]``): 입력 image (``[H, W, 4]``)
        target (``NDArray[np.uint8]``): Target image (``[H, W, 4]``)

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, 4]``)
    """
    alpha_overlay = target[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay
    for channel in range(0, 3):
        img[:, :, channel] = (
            alpha_overlay * target[:, :, channel]
            + alpha_background * img[:, :, channel]
        )
    return img


def _make_text(
    txt: str, shape: Tuple[int], color: Tuple[int], fontsize: int
) -> NDArray[np.uint8]:
    """배경이 투명한 문자열 image 생성

    Args:
        txt (``str``): 입력 문자열
        shape (``Tuple[int]``): 출력 image의 shape
        color (``Tuple[int]``): 문자의 색
        fontsize (``int``): 문자의 크기

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, 4]``)
    """
    size = (1000, 1000)
    palette = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(palette)
    font = ImageFont.truetype(
        __file__.replace("vision/visual.py", "plot/NotoSansKR-Medium.ttf"), fontsize
    )
    text_width, text_height = draw.textsize(txt, font=font)
    d_x, d_y = (size[0] - text_width) // 2, (size[1] - text_height) // 2
    if d_x < 0 or d_y < 0:
        raise ValueError("Input text 'txt' is too long")
    x_0, x_1 = d_x, d_x + text_width
    y_0, y_1 = d_y, d_y + text_height
    draw.text((d_x, d_y), txt, font=font, fill=(*color, 255))
    palette = np.array(palette)[y_0:y_1, x_0:x_1, :]
    return pad(palette, shape, (0, 0, 0, 0))


def _text(
    img: NDArray[np.uint8],
    box_cwh: NDArray[DTypeLike],
    txt: str,
    color: Tuple[int],
    fontsize: int,
) -> NDArray[np.uint8]:
    """단일 text 시각화

    Args:
        img (``NDArray[np.uint8]``): 입력 image (``[H, W, C]``)
        box_cwh (``NDArray[DTypeLike]``): 문자열이 존재할 bbox (``[4]``)
        txt (``str``): Image에 추가할 문자열
        color (``Tuple[int]``): 문자의 색
        fontsize (``int``): 문자의 크기

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, 4]``)
    """
    x_0, y_0, x_1, y_1 = cwh2xyxy(box_cwh).astype(np.int32)
    width, height = x_1 - x_0, y_1 - y_0
    txt = _make_text(txt, (height, width), color, fontsize)
    img[y_0:y_1, x_0:x_1, :] = _paste(img[y_0:y_1, x_0:x_1, :], txt)
    return img


def text(
    img: NDArray[np.uint8],
    box: Union[List[Union[int, float]], NDArray[DTypeLike]],
    txt: Union[str, List[str]],
    color: Optional[Tuple[int]] = (0, 0, 0),
    vis: Optional[bool] = False,
    fontsize: Optional[int] = 100,
) -> NDArray[np.uint8]:
    """Text 시각화

    Args:
        img (``NDArray[np.uint8]``): 입력 image (``[H, W, C]``)
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): 문자열이 존재할 bbox (``[4]``, ``[N, 4]``, ``[4, 2]``, ``[N, 4, 2]``)
        txt (``str``): Image에 추가할 문자열
        color (``Optional[Tuple[int]]``): 문자의 색
        vis (``Optional[bool]``): 문자 영역의 시각화 여부
        fontsize (``Optional[int]``): 문자의 크기

    Returns:
        ``NDArray[np.uint8]``: 시각화 결과 (``[H, W, 4]``)

    Examples:
        Bbox:

        >>> box = np.array([[100, 200], [100, 1000], [1200, 1000], [1200, 200]])
        >>> box.shape
        (4, 2)
        >>> res1 = zz.vision.text(img, box, "먼지야")

        Bboxes:

        >>> boxes = np.array([[250, 200, 100, 100], [600, 600, 800, 200], [900, 300, 300, 400]])
        >>> boxes.shape
        (3, 4)
        >>> res2 = zz.vision.text(img, boxes, ["먼지야", "먼지야", "먼지야"], vis=True)

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284566305-fe9d1be6-b506-4140-bca9-db2a210f333c.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    box = _list2np(box)
    img = img.copy()
    img = _cvt_bgra(img)
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if poly:
        box_poly = box
        box_cwh = poly2cwh(box)
    else:
        box_poly = cwh2poly(box)
        box_cwh = box
    if multi:
        if not shape[0] == len(txt):
            raise ValueError("'box.shape[0]' and 'len(txt)' must be equal")
        for b_poly, b_cwh, txt_ in zip(box_poly, box_cwh, txt):
            img = _text(img, b_cwh, txt_, color, fontsize)
            if vis:
                img = _bbox(img, b_poly, (0, 0, 255, 255), 2)
    else:
        img = _text(img, box_cwh, txt, color, fontsize)
        if vis:
            img = _bbox(img, box_poly, (0, 0, 255, 255), 2)
    return img


def paste(
    img: NDArray[np.uint8],
    target: NDArray[np.uint8],
    box: Union[List[Union[int, float]], NDArray[DTypeLike]],
    resize: Optional[bool] = False,
    vis: Optional[bool] = False,
    poly: Optional[NDArray[DTypeLike]] = None,
    alpha: Optional[int] = None,
    gaussian: Optional[int] = None,
) -> Union[NDArray[np.uint8], Tuple[NDArray[np.uint8], NDArray[DTypeLike]]]:
    """``target`` image를 ``img`` 위에 투명도를 포함하여 병합

    Note:
        ``PIL.Image.paste`` 를 ``numpy`` 와 ``cv2`` 기반으로 구현

        >>> img = Image.open("test.png").convert("RGBA")
        >>> target = Image.open("target.png").convert("RGBA")
        >>> img.paste(target, (0, 0), target)

    Args:
        img (``NDArray[np.uint8]``): 입력 image (``[H, W, C]``)
        target (``NDArray[np.uint8]``): Target image (``[H, W, 4]``)
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): 병합될 영역 (``xyxy`` 형식)
        resize (``Optional[bool]``): Target image의 resize 여부
        vis (``Optional[bool]``): 지정한 영역 (``box``)의 시각화 여부
        poly (``Optional[NDArray[DTypeLike]]``): 변형된 좌표 (``[N, 2]``)
        alpha (``Optional[int]``): ``target`` image의 투명도 변경
        gaussian (``Optional[int]``): 자연스러운 병합을 위해 ``target`` 의 alpha channel에 적용될 Gaussian blur의 kernel size

    Returns:
        ``Union[NDArray[np.uint8], Tuple[NDArray[np.uint8], NDArray[DTypeLike]]]``: 시각화 결과 (``[H, W, 4]``) 및 ``poly`` 입력 시 변형된 좌표값

    Examples:
        Without Poly:

        >>> poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
        >>> target = zz.vision.cutout(img, poly, 200)
        >>> res1 = zz.vision.paste(img, target, [200, 200, 1000, 800], resize=False, vis=True)
        >>> res2 = zz.vision.paste(img, target, [200, 200, 1000, 800], resize=True, vis=True, alpha=255)

        With Poly:

        >>> poly -= zz.vision.poly2xyxy(poly)[:2]
        >>> target = zz.vision.bbox(target, poly, color=(255, 0, 0), thickness=20)
        >>> res3, poly3 = zz.vision.paste(img, target, [200, 200, 1000, 800], resize=False, poly=poly)
        >>> poly3
        array([[300.        , 200.        ],
               [557.14285714, 200.        ],
               [900.        , 628.57142857],
               [557.14285714, 800.        ],
               [300.        , 542.85714286]])
        >>> res3 = zz.vision.bbox(res3, poly3)
        >>> res4, poly4 = zz.vision.paste(img, target, [200, 200, 1000, 800], resize=True, poly=poly)
        >>> poly4
        array([[ 200.        ,  200.        ],
               [ 542.85714286,  200.        ],
               [1000.        ,  628.57142857],
               [ 542.85714286,  800.        ],
               [ 200.        ,  542.85714286]])
        >>> res4 = zz.vision.bbox(res4, poly4)

        Gaussian Blur:

        >>> res5, poly5 = zz.vision.paste(img, target, [200, 200, 1000, 800], resize=True, poly=poly, gaussian=501)
        >>> res5 = zz.vision.bbox(res5, poly5)

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/285364676-27ae4292-0553-4561-a275-ea56c046d147.png
            :alt: Visualzation Result
            :align: center
            :width: 600px
    """
    x_0, y_0, x_1, y_1 = map(int, box)
    box_height, box_width = y_1 - y_0, x_1 - x_0
    img = img.copy()
    img = _cvt_bgra(img)
    target = target.copy()
    tar_height, tar_width = target.shape[:2]
    if alpha is not None:
        target[:, :, 3][0 < target[:, :, 3]] = alpha
    if gaussian is not None:
        invisible = target[:, :, 3] == 0
        pad_gaussian = gaussian * 3
        target_alpha = cv2.copyMakeBorder(
            target[:, :, 3],
            pad_gaussian,
            pad_gaussian,
            pad_gaussian,
            pad_gaussian,
            cv2.BORDER_CONSTANT,
        )
        target[:, :, 3] = cv2.GaussianBlur(target_alpha, (gaussian, gaussian), 0)[
            pad_gaussian:-pad_gaussian, pad_gaussian:-pad_gaussian
        ]
        target[:, :, 3][invisible] = 0
    if resize:
        target = cv2.resize(
            target, (box_width, box_height), interpolation=cv2.INTER_LINEAR
        )
        if poly is not None:
            poly = poly * (box_width / tar_width, box_height / tar_height) + (x_0, y_0)
    else:
        if poly is None:
            target = pad(target, (box_height, box_width), (0, 0, 0, 0))
        else:
            target, poly = pad(target, (box_height, box_width), (0, 0, 0, 0), poly)
            poly += (x_0, y_0)
    img[y_0:y_1, x_0:x_1, :] = _paste(img[y_0:y_1, x_0:x_1, :], target)
    if vis:
        box = np.array([[x_0, y_0], [x_0, y_1], [x_1, y_1], [x_1, y_0]])
        img = _bbox(img, box, (0, 0, 255, 255), 2)
    if poly is None:
        return img
    return img, poly
