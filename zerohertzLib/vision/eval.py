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

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import DTypeLike, NDArray
from shapely.geometry import Polygon

from zerohertzLib.plot import figure, plot, savefig, scatter


def iou(poly1: NDArray[DTypeLike], poly2: NDArray[DTypeLike]) -> float:
    """IoU (Intersection over Union)를 계산하는 함수

    Args:
        poly1 (``NDArray[DTypeLike]``): IoU를 계산할 polygon (``[S1, 2]``, ``[[x_0, y_0], [x_1, y_1], ...]``)
        poly2 (``NDArray[DTypeLike]``): IoU를 계산할 polygon (``[S2, 2]``, ``[[x_0, y_0], [x_1, y_1], ...]``)

    Returns:
        ``float``: IoU 값

    Examples:
        >>> poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> poly2 = poly1 + (5, 0)
        >>> poly2
        array([[ 5,  0],
               [15,  0],
               [15, 10],
               [ 5, 10]])
        >>> zz.vision.iou(poly1, poly2)
        0.3333333333333333
    """
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)
    return polygon1.intersection(polygon2).area / polygon1.union(polygon2).area


def _append(
    logs: Dict[str, List[Any]],
    instance: int,
    confidence: float,
    class_: Union[int, str],
    iou_: float,
    results: str,
    gt: NDArray[DTypeLike],
    inf: NDArray[DTypeLike],
) -> None:
    logs["instance"].append(instance)
    logs["confidence"].append(confidence)
    logs["class"].append(class_)
    logs["IoU"].append(iou_)
    logs["results"].append(results)
    if gt is None:
        logs["gt_x0"].append(None)
        logs["gt_y0"].append(None)
        logs["gt_x1"].append(None)
        logs["gt_y1"].append(None)
        logs["gt_x2"].append(None)
        logs["gt_y2"].append(None)
        logs["gt_x3"].append(None)
        logs["gt_y3"].append(None)
    else:
        logs["gt_x0"].append(gt[0][0])
        logs["gt_y0"].append(gt[0][1])
        logs["gt_x1"].append(gt[1][0])
        logs["gt_y1"].append(gt[1][1])
        logs["gt_x2"].append(gt[2][0])
        logs["gt_y2"].append(gt[2][1])
        logs["gt_x3"].append(gt[3][0])
        logs["gt_y3"].append(gt[3][1])
    if inf is None:
        logs["inf_x0"].append(None)
        logs["inf_y0"].append(None)
        logs["inf_x1"].append(None)
        logs["inf_y1"].append(None)
        logs["inf_x2"].append(None)
        logs["inf_y2"].append(None)
        logs["inf_x3"].append(None)
        logs["inf_y3"].append(None)
    else:
        logs["inf_x0"].append(inf[0][0])
        logs["inf_y0"].append(inf[0][1])
        logs["inf_x1"].append(inf[1][0])
        logs["inf_y1"].append(inf[1][1])
        logs["inf_x2"].append(inf[2][0])
        logs["inf_y2"].append(inf[2][1])
        logs["inf_x3"].append(inf[3][0])
        logs["inf_y3"].append(inf[3][1])


def evaluation(
    ground_truths: NDArray[DTypeLike],
    inferences: NDArray[DTypeLike],
    confidences: List[float],
    gt_classes: Optional[List[str]] = None,
    inf_classes: Optional[List[str]] = None,
    file_name: Optional[str] = None,
    threshold: Optional[float] = 0.5,
) -> pd.DataFrame:
    """단일 이미지 내 detection model의 추론 성능 평가

    Args:
        ground_truths (``NDArray[DTypeLike]``): Ground truth object들의 polygon (``[N, 4, 2]``, ``[[[x_0, y_0], [x_1, y_1], ...], ...]``)
        inferences (``NDArray[DTypeLike]``): Model이 추론한 각 object들의 polygon (``[M, 4, 2]``, ``[[[x_0, y_0], [x_1, y_1], ...], ...]``)
        confidences (``List[float]``): Model이 추론한 각 object들의 confidence(``[M]``)
        gt_classes (``Optional[List[str]]``): Ground truth object들의 class (``[N]``)
        inf_classes (``Optional[List[str]]``): Model이 추론한 각 object들의 class (``[M]``)
        file_name (``Optional[str]``): 평가 image의 이름
        threshold (``Optional[float]``): IoU의 threshold

    Note:
        - `N`: 한 이미지의 ground truth 내 존재하는 object의 수
        - `M`: 한 이미지의 inference 결과 내 존재하는 object의 수

        .. image:: _static/examples/static/vision.evaluation.png
            :align: center
            :width: 600px

    Returns:
        ``pd.DataFrame``: 단일 이미지의 model 성능 평가 결과

    Examples:
        >>> poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        >>> ground_truths = np.array([poly, poly + 20, poly + 40])
        >>> inferences = np.array([poly, poly + 19, poly + 80])
        >>> confidences = np.array([0.6, 0.7, 0.8])
        >>> zz.vision.evaluation(ground_truths, inferences, confidences, file_name="test.png")
          file_name  instance  confidence  class       IoU results  gt_x0  gt_y0  gt_x1  gt_y1  gt_x2  gt_y2  gt_x3  gt_y3  inf_x0  inf_y0  inf_x1  inf_y1  inf_x2  inf_y2  inf_x3  inf_y3
        0  test.png         0         0.8    0.0  0.000000      FP    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    80.0    80.0    90.0    80.0    90.0    90.0    80.0    90.0
        1  test.png         1         0.7    0.0  0.680672      TP   20.0   20.0   30.0   20.0   30.0   30.0   20.0   30.0    19.0    19.0    29.0    19.0    29.0    29.0    19.0    29.0
        2  test.png         2         0.6    0.0  1.000000      TP    0.0    0.0   10.0    0.0   10.0   10.0    0.0   10.0     0.0     0.0    10.0     0.0    10.0    10.0     0.0    10.0
        3  test.png         3         0.0    0.0  0.000000      FN   40.0   40.0   50.0   40.0   50.0   50.0   40.0   50.0     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
        >>> gt_classes = np.array(["cat", "dog", "cat"])
        >>> inf_classes = np.array(["cat", "dog", "cat"])
        >>> zz.vision.evaluation(ground_truths, inferences, confidences, gt_classes, inf_classes)
           instance  confidence class       IoU results  gt_x0  gt_y0  gt_x1  gt_y1  gt_x2  gt_y2  gt_x3  gt_y3  inf_x0  inf_y0  inf_x1  inf_y1  inf_x2  inf_y2  inf_x3  inf_y3
        0         0         0.8   cat  0.000000      FP    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    80.0    80.0    90.0    80.0    90.0    90.0    80.0    90.0
        1         1         0.6   cat  1.000000      TP    0.0    0.0   10.0    0.0   10.0   10.0    0.0   10.0     0.0     0.0    10.0     0.0    10.0    10.0     0.0    10.0
        2         2         0.0   cat  0.000000      FN   40.0   40.0   50.0   40.0   50.0   50.0   40.0   50.0     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
        3         3         0.7   dog  0.680672      TP   20.0   20.0   30.0   20.0   30.0   30.0   20.0   30.0    19.0    19.0    29.0    19.0    29.0    29.0    19.0    29.0
    """
    logs = defaultdict(list)
    if gt_classes is None and inf_classes is None:
        gt_classes = np.zeros(len(ground_truths))
        inf_classes = np.zeros(len(inferences))
    instance = 0
    for cls in set(gt_classes).union(set(inf_classes)):
        cls_gt = ground_truths[np.where(gt_classes == cls)]
        cls_inf = inferences[np.where(inf_classes == cls)]
        cls_conf = confidences[np.where(inf_classes == cls)]
        sorted_indices = np.argsort(-cls_conf)
        cls_inf = cls_inf[sorted_indices]
        cls_conf = cls_conf[sorted_indices]
        matched = set()
        for confidence, inf in zip(cls_conf, cls_inf):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(cls_gt):
                if gt_idx in matched:
                    continue
                iou_ = iou(gt, inf)
                if iou_ > best_iou:
                    best_iou = iou_
                    best_gt_idx = gt_idx
            if best_iou >= threshold:
                matched.add(best_gt_idx)
                _append(
                    logs,
                    instance,
                    confidence,
                    cls,
                    best_iou,
                    "TP",
                    cls_gt[best_gt_idx],
                    inf,
                )
                instance += 1
            else:
                _append(logs, instance, confidence, cls, 0.0, "FP", None, inf)
                instance += 1
        for gt_idx, gt in enumerate(cls_gt):
            if gt_idx not in matched:
                _append(logs, instance, 0.0, cls, 0.0, "FN", gt, None)
                instance += 1
    logs = pd.DataFrame(logs)
    if file_name is not None:
        logs["file_name"] = file_name
        logs = logs[["file_name"] + [col for col in logs.columns if col != "file_name"]]
    return logs


def meanap(logs: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """Detection model의 P-R curve 시각화 및 mAP 산출

    Args:
        logs (``pd.DataFrame``): ``zz.vision.evaluation`` 함수를 통해 평가된 결과

    Returns:
        ``Tuple[float, Dict[str, float]]``: mAP 값 및 class에 따른 AP 값 (시각화 결과는 ``prc_curve.png``, ``pr_curve.png`` 로 현재 directory에 저장)

    Examples:
        >>> logs1 = zz.vision.evaluation(ground_truths_1, inferences_1, confidences_1, gt_classes, inf_classes, file_name="test_1.png")
        >>> logs2 = zz.vision.evaluation(ground_truths_2, inferences_2, confidences_2, gt_classes, inf_classes, file_name="test_2.png")
        >>> logs = pd.concat([logs1, logs2], ignore_index=True)
        >>> zz.vision.meanap(logs)
        (0.7030629916206652, defaultdict(<class 'float'>, {'dog': 0.7177078883735305, 'cat': 0.6884180948677999}))

        .. image:: _static/examples/static/vision.meanap.png
            :align: center
            :width: 600px
    """
    logs = logs.sort_values(by="confidence", ascending=False)
    confidence_per_cls = defaultdict(list)
    recall_per_cls = defaultdict(list)
    precision_per_cls = defaultdict(list)
    pr_curve = defaultdict(list)
    aps = defaultdict(float)
    classes = set(logs["class"])
    for cls in classes:
        gt = len(
            logs[
                (logs["class"] == cls)
                & ((logs["results"] == "TP") | (logs["results"] == "FN"))
            ]
        )
        for confidence in set(logs[logs["class"] == cls]["confidence"]):
            true_positive = len(
                logs[
                    (logs["class"] == cls)
                    & (logs["confidence"] >= confidence)
                    & (logs["results"] == "TP")
                ]
            )
            false_positive = len(
                logs[
                    (logs["class"] == cls)
                    & (logs["confidence"] >= confidence)
                    & (logs["results"] == "FP")
                ]
            )
            if true_positive + false_positive == 0:
                precision = 0
            else:
                precision = true_positive / (true_positive + false_positive)
            if gt == 0:
                recall = 0
            else:
                recall = true_positive / gt  # (true_positive + false_negative)
            pr_curve[cls].append((recall, precision))
            confidence_per_cls[cls].append(confidence)
            recall_per_cls[cls].append(recall)
            precision_per_cls[cls].append(precision)
        pr_curve[cls] = sorted(pr_curve[cls])
        pr_curve[cls].insert(0, (0, pr_curve[cls][0][1]))
        for i in range(1, len(pr_curve[cls])):
            recall_diff = pr_curve[cls][i][0] - pr_curve[cls][i - 1][0]
            precision_max = max(precision[1] for precision in pr_curve[cls][i:])
            aps[cls] += recall_diff * precision_max
    map_ = sum(aps.values()) / len(aps)
    _prc_curve(confidence_per_cls, recall_per_cls, precision_per_cls, classes)
    _pr_curve(pr_curve, classes, map_)
    return map_, aps


def _prc_curve(
    confidence_per_cls: Dict[str, List[float]],
    recall_per_cls: Dict[str, List[float]],
    precision_per_cls: Dict[str, List[float]],
    classes: Set[str],
) -> None:
    data = {}
    if len(classes) == 1:
        cls = list(classes)[0]
        data["Recall"] = [confidence_per_cls[cls], recall_per_cls[cls]]
        data["Precision"] = [
            confidence_per_cls[cls],
            precision_per_cls[cls],
        ]
    else:
        for cls in classes:
            data[f"{cls}: Recall"] = [confidence_per_cls[cls], recall_per_cls[cls]]
            data[f"{cls}: Precision"] = [
                confidence_per_cls[cls],
                precision_per_cls[cls],
            ]
    scatter(
        data,
        "Confidence",
        "Recall & Precision",
        [-0.1, 1.1],
        [-0.1, 1.1],
        ncol=2,
        title="PRC Curve",
        markersize=6,
    )


def _pr_curve(
    pr_curve: Dict[str, List[Tuple[float]]], classes: Set[str], map_: float
) -> None:
    figure()
    if len(classes) == 1:
        recall, precision = [], []
        for recall_, precision_ in pr_curve[list(classes)[0]]:
            recall.append(recall_)
            precision.append(precision_)
        plot(
            recall,
            {"": precision},
            "Recall",
            "Precision",
            [-0.1, 1.1],
            [-0.1, 1.1],
            stacked=True,
            title=f"P-R Curve (mAP: {map_:.2f})",
            markersize=1,
            save=False,
        )
    else:
        for cls in classes:
            recall, precision = [], []
            for recall_, precision_ in pr_curve[cls]:
                recall.append(recall_)
                precision.append(precision_)
            plot(
                recall,
                {cls: precision},
                "Recall",
                "Precision",
                [-0.1, 1.1],
                [-0.1, 1.1],
                stacked=True,
                title=f"P-R Curve (mAP: {map_:.2f})",
                markersize=1,
                save=False,
            )
        plt.legend()
    savefig("pr_curve")
