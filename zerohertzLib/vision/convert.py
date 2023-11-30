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

import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from matplotlib.path import Path
from numpy.typing import DTypeLike, NDArray

from zerohertzLib.util import Json, write_json

from .util import _is_bbox


def _list2np(box: List[Any]) -> NDArray[DTypeLike]:
    if isinstance(box, list):
        return np.array(box)
    return box


def _cwh2xyxy(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, y_0 = box[:2] - box[2:] / 2
    x_1, y_1 = box[:2] + box[2:] / 2
    return np.array([x_0, y_0, x_1, y_1], dtype=box.dtype)


def cwh2xyxy(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.cwh2xyxy([20, 30, 20, 20])
        array([10, 20, 30, 40])
        >>> zz.vision.cwh2xyxy(np.array([[20, 30, 20, 20], [50, 75, 40, 50]]))
        array([[ 10,  20,  30,  40],
               [ 30,  50,  70, 100]])
    """
    box = _list2np(box)
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if poly:
        raise ValueError("The 'cwh' must be of shape [4], [N, 4]")
    if multi:
        boxes = np.zeros((shape[0], 4), dtype=box.dtype)
        for i, box_ in enumerate(box):
            boxes[i] = _cwh2xyxy(box_)
        return boxes
    return _cwh2xyxy(box)


def _cwh2poly(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, y_0 = box[:2] - box[2:] / 2
    x_1, y_1 = box[:2] + box[2:] / 2
    return np.array([[x_0, y_0], [x_1, y_0], [x_1, y_1], [x_0, y_1]], dtype=box.dtype)


def cwh2poly(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Examples:
        >>> zz.vision.cwh2poly([20, 30, 20, 20])
        array([[10, 20],
               [30, 20],
               [30, 40],
               [10, 40]])
        >>> zz.vision.cwh2poly(np.array([[20, 30, 20, 20], [50, 75, 40, 50]]))
        array([[[ 10,  20],
                [ 30,  20],
                [ 30,  40],
                [ 10,  40]],
               [[ 30,  50],
                [ 70,  50],
                [ 70, 100],
                [ 30, 100]]])
    """
    box = _list2np(box)
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if poly:
        raise ValueError("The 'cwh' must be of shape [4], [N, 4]")
    if multi:
        boxes = np.zeros((shape[0], 4, 2), dtype=box.dtype)
        for i, box_ in enumerate(box):
            boxes[i] = _cwh2poly(box_)
        return boxes
    return _cwh2poly(box)


def _xyxy2cwh(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, y_0, x_1, y_1 = box
    return np.array(
        [(x_0 + x_1) / 2, (y_0 + y_1) / 2, x_1 - x_0, y_1 - y_0], dtype=box.dtype
    )


def xyxy2cwh(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.xyxy2cwh([10, 20, 30, 40])
        array([20, 30, 20, 20])
        >>> zz.vision.xyxy2cwh(np.array([[10, 20, 30, 40], [30, 50, 70, 100]]))
        array([[20, 30, 20, 20],
               [50, 75, 40, 50]])
    """
    box = _list2np(box)
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if poly:
        raise ValueError("The 'xyxy' must be of shape [4], [N, 4]")
    if multi:
        boxes = np.zeros((shape[0], 4), dtype=box.dtype)
        for i, box_ in enumerate(box):
            boxes[i] = _xyxy2cwh(box_)
        return boxes
    return _xyxy2cwh(box)


def _xyxy2poly(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, y_0, x_1, y_1 = box
    return np.array([[x_0, y_0], [x_1, y_0], [x_1, y_1], [x_0, y_1]], dtype=box.dtype)


def xyxy2poly(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Examples:
        >>> zz.vision.xyxy2poly([10, 20, 30, 40])
        array([[10, 20],
               [30, 20],
               [30, 40],
               [10, 40]])
        >>> zz.vision.xyxy2poly(np.array([[10, 20, 30, 40], [30, 50, 70, 100]]))
        array([[[ 10,  20],
                [ 30,  20],
                [ 30,  40],
                [ 10,  40]],
               [[ 30,  50],
                [ 70,  50],
                [ 70, 100],
                [ 30, 100]]])
    """
    box = _list2np(box)
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if poly:
        raise ValueError("The 'xyxy' must be of shape [4], [N, 4]")
    if multi:
        boxes = np.zeros((shape[0], 4, 2), dtype=box.dtype)
        for i, box_ in enumerate(box):
            boxes[i] = _xyxy2poly(box_)
        return boxes
    return _xyxy2poly(box)


def _poly2cwh(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, x_1 = box[:, 0].min(), box[:, 0].max()
    y_0, y_1 = box[:, 1].min(), box[:, 1].max()
    return np.array(
        [(x_0 + x_1) / 2, (y_0 + y_1) / 2, x_1 - x_0, y_1 - y_0], dtype=box.dtype
    )


def poly2cwh(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[cx, cy, w, h]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.poly2cwh([[10, 20], [30, 20], [30, 40], [10, 40]])
        array([20, 30, 20, 20])
        >>> zz.vision.poly2cwh(np.array([[[10, 20], [30, 20], [30, 40], [10, 40]], [[30, 50], [70, 50], [70, 100], [30, 100]]]))
        array([[20, 30, 20, 20],
               [50, 75, 40, 50]])
    """
    box = _list2np(box)
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if not poly:
        raise ValueError("The 'poly' must be of shape [4, 2], [N, 4, 2]")
    if multi:
        boxes = np.zeros((shape[0], 4), dtype=box.dtype)
        for i, box_ in enumerate(box):
            boxes[i] = _poly2cwh(box_)
        return boxes
    return _poly2cwh(box)


def _poly2xyxy(box: NDArray[DTypeLike]) -> NDArray[DTypeLike]:
    x_0, x_1 = box[:, 0].min(), box[:, 0].max()
    y_0, y_1 = box[:, 1].min(), box[:, 1].max()
    return np.array([x_0, y_0, x_1, y_1], dtype=box.dtype)


def poly2xyxy(
    box: Union[List[Union[int, float]], NDArray[DTypeLike]]
) -> NDArray[DTypeLike]:
    """Bbox 변환

    Args:
        box (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): ``[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]`` 로 구성된 bbox (``[4, 2]`` or ``[N, 4, 2]``)

    Returns:
        ``NDArray[DTypeLike]``: ``[x0, y0, x1, y1]`` 로 구성된 bbox (``[4]`` or ``[N, 4]``)

    Examples:
        >>> zz.vision.poly2xyxy([[10, 20], [30, 20], [30, 40], [10, 40]])
        array([10, 20, 30, 40])
        >>> zz.vision.poly2xyxy(np.array([[[10, 20], [30, 20], [30, 40], [10, 40]], [[30, 50], [70, 50], [70, 100], [30, 100]]]))
        array([[ 10,  20,  30,  40],
               [ 30,  50,  70, 100]])
    """
    box = _list2np(box)
    shape = box.shape
    multi, poly = _is_bbox(shape)
    if not poly:
        raise ValueError("The 'poly' must be of shape [4, 2], [N, 4, 2]")
    if multi:
        boxes = np.zeros((shape[0], 4), dtype=box.dtype)
        for i, box_ in enumerate(box):
            boxes[i] = _poly2xyxy(box_)
        return boxes
    return _poly2xyxy(box)


def poly2mask(
    poly: Union[List[Union[int, float]], NDArray[DTypeLike]], shape: Tuple[int]
) -> NDArray[bool]:
    """다각형 좌표를 입력받아 mask로 변환

    Args:
        poly (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): Mask의 꼭짓점 좌표 (``[N, 2]``)
        shape (``Tuple[int]``): 출력될 mask의 shape ``(H, W)``

    Returns:
        ``NDArray[bool]``: 시각화 결과 (``[H, W, C]``)

    Examples:
        >>> poly = [[10, 10], [20, 10], [30, 40], [20, 60], [10, 20]]
        >>> mask = zz.vision.poly2mask(poly, (70, 100))
        >>> mask.shape
        (70, 100)
        >>> mask.dtype
        dtype('bool')

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284488846-9237c52b-d181-447c-95da-f67aa8fb2854.png
            :alt: Visualzation Result
            :align: center
            :width: 300px
    """
    poly = _list2np(poly)
    poly = Path(poly)
    pts_x, pts_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    pts_x, pts_y = pts_x.flatten(), pts_y.flatten()
    points = np.vstack((pts_x, pts_y)).T
    grid = poly.contains_points(points)
    mask = grid.reshape(shape)
    return mask


def poly2area(poly: Union[List[Union[int, float]], NDArray[DTypeLike]]) -> float:
    """다각형의 면적을 산출하는 함수

    Args:
        poly (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): 다각형 (``[N, 2]``)

    Returns:
        ``float``: 다각형의 면적

    Examples:
        >>> poly = [[10, 10], [20, 10], [30, 40], [20, 60], [10, 20]]
        >>> zz.vision.poly2area(poly)
        550.0
        >>> box = np.array([[100, 200], [1200, 200], [1200, 1000], [100, 1000]])
        >>> zz.vision.poly2area(box)
        880000.0
    """
    poly = _list2np(poly)
    pts_x = poly[:, 0]
    pts_y = poly[:, 1]
    return 0.5 * np.abs(
        np.dot(pts_x, np.roll(pts_y, 1)) - np.dot(pts_y, np.roll(pts_x, 1))
    )


def poly2ratio(poly: Union[List[Union[int, float]], NDArray[DTypeLike]]) -> float:
    """다각형의 bbox 대비 다각형의 면적 비율을 산출하는 함수

    Args:
        poly (``Union[List[Union[int, float]], NDArray[DTypeLike]]``): 다각형 (``[N, 2]``)

    Returns:
        ``float``: 다각형의 bbox 대비 다각형의 비율

    Examples:
        >>> poly = [[10, 10], [20, 10], [30, 40], [20, 60], [10, 20]]
        >>> zz.vision.poly2ratio(poly)
        0.55
        >>> box = np.array([[100, 200], [1200, 200], [1200, 1000], [100, 1000]])
        >>> zz.vision.poly2ratio(box)
        1.0
    """
    poly_area = poly2area(poly)
    _, _, height, width = poly2cwh(poly)
    bbox_area = height * width
    return poly_area / bbox_area


def labelstudio2yolo(
    label_studio_path: str, target_path: str, label: Dict[str, int]
) -> None:
    """Label Studio로 annotation한 JSON data를 YOLO format으로 변환하는 함수

    Args:
        label_studio_path (``str``): Label Studio annotation에 대한 JSON file 경로
        target_path (``str``): YOLO format data가 저장될 경로
        label (``Dict[str, int]``): Label Studio에서 사용한 label을 정수로 변환하는 dictionary

    Returns:
        ``None``: ``target_path`` 에 ``.txt`` file로 저장

    Examples:
        >>> zz.vision.labelstudio2yolo("cwh.json", "tmp", {"TRUE": 0, "FALSE": 1})
        >>> zz.vision.labelstudio2yolo("poly.json", "tmp", {"TRUE": 0, "FALSE": 1})
    """
    label_studio = Json(label_studio_path)
    os.makedirs(target_path, exist_ok=True)
    for ls in label_studio:
        img_file_name = "-".join(ls["data"]["image"].split("/")[-1].split("-")[1:])
        txt_file_name = ".".join(img_file_name.split(".")[:-1]) + ".txt"
        converted_ground_truth = []
        if len(ls["annotations"]) > 1:
            raise ValueError("The 'annotations' in the JSON file are plural")
        for res in ls["annotations"][0]["result"]:
            if res["type"] == "rectanglelabels":
                box_cwh = np.array(
                    [
                        res["value"]["x"] / 100,
                        res["value"]["y"] / 100,
                        res["value"]["width"] / 100,
                        res["value"]["height"] / 100,
                    ]
                )
                if len(res["value"]["rectanglelabels"]) > 1:
                    raise ValueError("The 'rectanglelabels' are plural")
                lab = label[res["value"]["rectanglelabels"][0]]
            elif res["type"] == "polygonlabels":
                box_poly = np.array(res["value"]["points"]) / 100
                box_cwh = poly2cwh(box_poly)
                if len(res["value"]["polygonlabels"]) > 1:
                    raise ValueError("The 'polygonlabels' are plural")
                lab = label[res["value"]["polygonlabels"][0]]
            converted_ground_truth.append(
                f"{lab} " + " ".join(map(str, box_cwh)) + "\n"
            )
        with open(os.path.join(target_path, txt_file_name), "w", encoding="utf-8") as f:
            f.writelines(converted_ground_truth)


def labelstudio2labelme(label_studio_path: str, target_path: str) -> None:
    """Label Studio로 annotation한 JSON data를 LabelMe format으로 변환하는 함수

    Args:
        label_studio_path (``str``): Label Studio annotation에 대한 JSON file 경로
        target_path (``str``): LabelMe format data가 저장될 경로

    Returns:
        ``None``: ``target_path`` 에 ``.json`` file로 저장

    Examples:
        >>> zz.vision.labelstudio2labelme("cwh.json", "tmp")
        >>> zz.vision.labelstudio2labelme("poly.json", "tmp")
    """
    label_studio = Json(label_studio_path)
    os.makedirs(target_path, exist_ok=True)
    for ls in label_studio:
        img_file_name = "-".join(ls["data"]["image"].split("/")[-1].split("-")[1:])
        json_file_name = ".".join(img_file_name.split(".")[:-1])
        converted_ground_truth = []
        if len(ls["annotations"]) > 1:
            raise ValueError("The 'annotations' in the JSON file are plural")
        for res in ls["annotations"][0]["result"]:
            width, height = res["original_width"], res["original_height"]
            if res["type"] == "rectanglelabels":
                box_cwh = np.array(
                    [
                        res["value"]["x"] * width / 100,
                        res["value"]["y"] * height / 100,
                        (res["value"]["x"] + res["value"]["width"]) * width / 100,
                        (res["value"]["y"] + res["value"]["height"]) * height / 100,
                    ]
                )
                box_poly = xyxy2poly(box_cwh)
                if len(res["value"]["rectanglelabels"]) > 1:
                    raise ValueError("The 'rectanglelabels' are plural")
                lab = res["value"]["rectanglelabels"][0]
            elif res["type"] == "polygonlabels":
                box_poly = np.array(res["value"]["points"]) * (
                    width / 100,
                    height / 100,
                )
                if len(res["value"]["polygonlabels"]) > 1:
                    raise ValueError("The 'polygonlabels' are plural")
                lab = res["value"]["polygonlabels"][0]
            converted_ground_truth.append(
                {"label": lab, "points": box_poly.tolist(), "shape_type": "polygon"}
            )
        write_json(
            {"shapes": converted_ground_truth},
            os.path.join(target_path, json_file_name),
        )
