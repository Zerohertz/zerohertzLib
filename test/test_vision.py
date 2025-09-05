# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import os
import random
import shutil

import cv2
import numpy as np
import pandas as pd

import zerohertzLib as zz

BOX_CWH = np.array([650, 600, 1100, 800])
BOX_XYXY = np.array([100, 200, 1200, 1000])
BOX_POLY = np.array([[100, 200], [1200, 200], [1200, 1000], [100, 1000]])
BOXES_CWH = np.array([[250, 200, 100, 100], [600, 600, 800, 200], [900, 300, 300, 400]])
BOXES_XYXY = np.array(
    [[200, 150, 300, 250], [200, 500, 1000, 700], [750, 100, 1050, 500]]
)
BOXES_POLY = np.array(
    [
        [[200, 150], [300, 150], [300, 250], [200, 250]],
        [[200, 500], [1000, 500], [1000, 700], [200, 700]],
        [[750, 100], [1050, 100], [1050, 500], [750, 500]],
    ]
)

tmp = os.path.dirname(__file__)
data = os.path.join(tmp, "data")


def test_img2gif():
    for i in range(5):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(f"{data}/{i}.png", img)
    zz.vision.img2gif(data, "IMG")
    assert "IMG.gif" in os.listdir()


def test_vid2gif():
    zz.vision.vid2gif(f"{data}/test.mov", "VID", quality=20)
    assert "VID.gif" in os.listdir()


def test_before_after_org():
    before = cv2.imread(f"{data}/test.jpg")
    after = cv2.GaussianBlur(before, (0, 0), 25)
    after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    zz.vision.before_after(before, after, quality=10, file_name="BA1")
    assert "BA1.png" in os.listdir()


def test_before_after_crop():
    before = cv2.imread(f"{data}/test.jpg")
    after = cv2.resize(before, (100, 100))
    after = cv2.cvtColor(after, cv2.COLOR_BGR2BGRA)
    zz.vision.before_after(before, after, [20, 40, 30, 60], file_name="BA2")
    assert "BA2.png" in os.listdir()


def test_grid_vertical():
    test = cv2.imread(f"{data}/test.jpg")
    test = cv2.resize(test, (200, 300))
    imgs = [(test + np.random.rand(*test.shape)).astype(np.uint8) for _ in range(8)]
    imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
    imgs[3] = cv2.cvtColor(imgs[3], cv2.COLOR_BGR2BGRA)
    zz.vision.grid(imgs, file_name="GRID_VERTICAL")
    assert "GRID_VERTICAL.png" in os.listdir()


def test_grid_horizontal():
    test = cv2.imread(f"{data}/test.jpg")
    test = cv2.resize(test, (300, 200))
    imgs = [(test + np.random.rand(*test.shape)).astype(np.uint8) for _ in range(8)]
    imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
    imgs[3] = cv2.cvtColor(imgs[3], cv2.COLOR_BGR2BGRA)
    zz.vision.grid(imgs, file_name="GRID_HORIZONTAL")
    assert "GRID_HORIZONTAL.png" in os.listdir()


def test_vert():
    test = cv2.imread(f"{data}/test.jpg")
    imgs = [
        cv2.resize(test, (random.randrange(300, 600), random.randrange(300, 600)))
        for _ in range(5)
    ]
    imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
    imgs[3] = cv2.cvtColor(imgs[3], cv2.COLOR_BGR2BGRA)
    zz.vision.vert(imgs, file_name="VERT")
    assert "VERT.png" in os.listdir()


def test_bbox_bgr_poly():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_POLY
    BGR = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_BGR_POLY.png", BGR)
    assert "BBOX_BGR_POLY.png" in os.listdir()


def test_bbox_bgra_poly():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_POLY
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    BGRA = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_BGRA_POLY.png", BGRA)
    assert "BBOX_BGRA_POLY.png" in os.listdir()


def test_bbox_gray_poly():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_POLY
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    GRAY = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_GRAY_POLY.png", GRAY)
    assert "BBOX_GRAY_POLY.png" in os.listdir()


def test_bboxes_bgr_poly():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_POLY
    BGR = zz.vision.bbox(img, boxes, thickness=10)
    cv2.imwrite("BBOXES_BGR_POLY.png", BGR)
    assert "BBOXES_BGR_POLY.png" in os.listdir()


def test_bboxes_bgra_poly():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_POLY
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    BGRA = zz.vision.bbox(img, boxes, thickness=10)
    cv2.imwrite("BBOXES_BGRA_POLY.png", BGRA)
    assert "BBOXES_BGRA_POLY.png" in os.listdir()


def test_bboxes_gray_poly():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_POLY
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    GRAY = zz.vision.bbox(img, boxes, thickness=10)
    cv2.imwrite("BBOXES_GRAY_POLY.png", GRAY)
    assert "BBOXES_GRAY_POLY.png" in os.listdir()


def test_bbox_bgr_cwh():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_CWH
    BGR = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_BGR_CWH.png", BGR)
    assert "BBOX_BGR_CWH.png" in os.listdir()


def test_bbox_bgra_cwh():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_CWH
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    BGRA = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_BGRA_CWH.png", BGRA)
    assert "BBOX_BGRA_CWH.png" in os.listdir()


def test_bbox_gray_cwh():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_CWH
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    GRAY = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_GRAY_CWH.png", GRAY)
    assert "BBOX_GRAY_CWH.png" in os.listdir()


def test_bboxes_bgr_cwh():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_CWH
    BGR = zz.vision.bbox(img, boxes, thickness=10)
    cv2.imwrite("BBOXES_BGR_CWH.png", BGR)
    assert "BBOXES_BGR_CWH.png" in os.listdir()


def test_bboxes_bgra_cwh():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_CWH
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    BGRA = zz.vision.bbox(img, boxes, thickness=10)
    cv2.imwrite("BBOXES_BGRA_CWH.png", BGRA)
    assert "BBOXES_BGRA_CWH.png" in os.listdir()


def test_bboxes_gray_cwh():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_CWH
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    GRAY = zz.vision.bbox(img, boxes, thickness=10)
    cv2.imwrite("BBOXES_GRAY_CWH.png", GRAY)
    assert "BBOXES_GRAY_CWH.png" in os.listdir()


def test_mask_bgr():
    img = cv2.imread(f"{data}/test.jpg")
    H, W, _ = img.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mks_ in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(30, 200)
        cv2.circle(mks_, (center_x, center_y), radius, (True), -1)
    mks = mks.astype(bool)
    BGR = zz.vision.mask(img, mks)
    cv2.imwrite("MASK_BGR.png", BGR)
    assert "MASK_BGR.png" in os.listdir()


def test_mask_bgra():
    img = cv2.imread(f"{data}/test.jpg")
    H, W, _ = img.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mks_ in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(30, 200)
        cv2.circle(mks_, (center_x, center_y), radius, (True), -1)
    mks = mks.astype(bool)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    BGRA = zz.vision.mask(img, mks, color=[random.randint(0, 255) for _ in range(3)])
    cv2.imwrite("MASK_BGRA.png", BGRA)
    assert "MASK_BGRA.png" in os.listdir()


def test_mask_poly():
    img = cv2.imread(f"{data}/test.jpg")
    H, W, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    cnt = 30
    poly = zz.vision.xyxy2poly(
        zz.vision.poly2xyxy((np.random.rand(cnt, 4, 2) * (W, H)))
    )
    BGRA = zz.vision.mask(
        img, poly=poly, color=[random.randint(0, 255) for _ in range(3)]
    )
    cv2.imwrite("MASK_POLY.png", BGRA)
    assert "MASK_POLY.png" in os.listdir()


def test_mask_gray_int():
    img = cv2.imread(f"{data}/test.jpg")
    H, W, _ = img.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mks_ in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(30, 200)
        cv2.circle(mks_, (center_x, center_y), radius, (True), -1)
    mks = mks.astype(bool)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cls = [i for i in range(cnt)]
    class_list = [cls[random.randint(0, 5)] for _ in range(cnt)]
    class_color = {}
    for c in cls:
        class_color[c] = [random.randint(0, 255) for _ in range(3)]
    GRAY = zz.vision.mask(
        img, mks, class_list=class_list, class_color=class_color, alpha=1
    )
    cv2.imwrite("MASK_GRAY_INT.png", GRAY)
    assert "MASK_GRAY_INT.png" in os.listdir()


def test_mask_bgra_str():
    img = cv2.imread(f"{data}/test.jpg")
    H, W, _ = img.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mks_ in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(30, 200)
        cv2.circle(mks_, (center_x, center_y), radius, (True), -1)
    mks = mks.astype(bool)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    cls = ["a", "b", "c"]
    class_list = [cls[random.randint(0, 2)] for _ in range(cnt)]
    class_color = {}
    for c in cls:
        class_color[c] = [random.randint(0, 255) for _ in range(3)]
    BGRA = zz.vision.mask(img, mks, class_list=class_list, class_color=class_color)
    cv2.imwrite("MASK_BGRA_STR.png", BGRA)
    assert "MASK_BGRA_STR.png" in os.listdir()


def test_text_bgr_poly():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_POLY
    BGR = zz.vision.text(img, box, "먼지야", vis=True)
    cv2.imwrite("TEXT_BGR_POLY.png", BGR)
    assert "TEXT_BGR_POLY.png" in os.listdir()


def test_text_bgra_poly():
    img = cv2.imread(f"{data}/test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    boxes = BOXES_POLY
    BGRA = zz.vision.text(
        img, boxes, ["오래오래", "오래오래", "오래오래"], (0, 255, 0), vis=True
    )
    cv2.imwrite("TEXT_BGRA_POLY.png", BGRA)
    assert "TEXT_BGRA_POLY.png" in os.listdir()


def test_text_gray_poly():
    img = cv2.imread(f"{data}/test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    box = BOX_POLY
    GRAY = zz.vision.text(img, box, "행복해라", vis=True)
    cv2.imwrite("TEXT_GRAY_POLY.png", GRAY)
    assert "TEXT_GRAY_POLY.png" in os.listdir()


def test_text_bgr_cwh():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_CWH
    BGR = zz.vision.text(img, boxes, ["먼지야", "먼지야", "먼지야"], vis=True)
    cv2.imwrite("TEXT_BGR_CWH.png", BGR)
    assert "TEXT_BGR_CWH.png" in os.listdir()


def test_text_bgra_cwh():
    img = cv2.imread(f"{data}/test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    boxes = BOXES_CWH
    BGRA = zz.vision.text(
        img, boxes, ["오래오래", "오래오래", "오래오래"], (0, 255, 0), vis=True
    )
    cv2.imwrite("TEXT_BGRA_CWH.png", BGRA)
    assert "TEXT_BGRA_CWH.png" in os.listdir()


def test_text_gray_cwh():
    img = cv2.imread(f"{data}/test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    box = BOX_CWH
    GRAY = zz.vision.text(img, box, "행복해라", vis=True)
    cv2.imwrite("TEXT_GRAY_CWH.png", GRAY)
    assert "TEXT_GRAY_CWH.png" in os.listdir()


def test_convert_cwh():
    assert (BOX_CWH == zz.vision.xyxy2cwh(BOX_XYXY)).all()
    assert (BOXES_CWH == zz.vision.xyxy2cwh(BOXES_XYXY)).all()
    assert (BOX_CWH == zz.vision.poly2cwh(BOX_POLY)).all()
    assert (BOXES_CWH == zz.vision.poly2cwh(BOXES_POLY)).all()


def test_convert_xyxy():
    assert (BOX_XYXY == zz.vision.cwh2xyxy(BOX_CWH)).all()
    assert (BOXES_XYXY == zz.vision.cwh2xyxy(BOXES_CWH)).all()
    assert (BOX_XYXY == zz.vision.poly2xyxy(BOX_POLY)).all()
    assert (BOXES_XYXY == zz.vision.poly2xyxy(BOXES_POLY)).all()


def test_convert_poly():
    assert (BOX_POLY == zz.vision.cwh2poly(BOX_CWH)).all()
    assert (BOXES_POLY == zz.vision.cwh2poly(BOXES_CWH)).all()
    assert (BOX_POLY == zz.vision.xyxy2poly(BOX_XYXY)).all()
    assert (BOXES_POLY == zz.vision.xyxy2poly(BOXES_XYXY)).all()


def test_is_pts_in_poly():
    assert zz.vision.is_pts_in_poly(BOX_POLY, [650, 600])
    assert zz.vision.is_pts_in_poly(BOX_POLY, [[450, 400], [850, 800]]).all()
    assert zz.vision.is_pts_in_poly(BOX_POLY, np.array([650, 600]))
    assert zz.vision.is_pts_in_poly(BOX_POLY, np.array([[450, 400], [850, 800]])).all()


def test_eval():
    num = 21
    false_negative = 1  # 미탐지
    false_positive = 50  # 오탐지
    false_positive -= false_negative
    inf = 14
    idx = num - false_negative
    classes = ["cat", "dog"]

    def _detection():
        ground_truths = zz.vision.cwh2poly(
            np.array(
                [
                    [random.randrange(50, 1950), random.randrange(50, 1950), 50, 50]
                    for _ in range(num)
                ]
            )
        )
        fp = random.randrange(false_positive)
        inferences = zz.vision.cwh2poly(
            np.array(
                [
                    [random.randrange(50, 1950), random.randrange(50, 1950), 50, 50]
                    for _ in range(num + fp)
                ]
            )
        )
        inferences[:idx] = (
            ground_truths[:idx]
            + np.random.randint(low=-inf, high=inf, size=ground_truths.shape)[:idx]
        )
        confidences = []
        gt_classes, inf_classes = [], []
        for ground_truth, inference in zip(ground_truths, inferences[:num]):
            iou_ = zz.vision.iou(ground_truth, inference)
            confidences.append(
                (iou_ * 2 + np.random.rand()) / 3 + (np.random.rand() - 1) / 10
            )
            cls = random.choice(classes)
            gt_classes.append(cls)
            inf_classes.append(cls)
        for _ in range(fp):
            confidences.append(np.random.rand() * 2 / 3)
            cls = random.choice(classes)
            inf_classes.append(cls)
        return (
            ground_truths,
            inferences.astype(int),
            np.array(confidences).clip(0, 1),
            np.array(gt_classes),
            np.array(inf_classes),
        )

    ground_truths, inferences, _, _, _ = _detection()
    img = np.full((2000, 2000, 3), 255, dtype=np.uint8)
    img = zz.vision.bbox(img, ground_truths, (255, 0, 0))
    img = zz.vision.bbox(img, inferences, (0, 0, 255))
    cv2.imwrite("bboxes.png", img)
    logs = None
    for i in range(20):
        ground_truths, inferences, confidences, _, _ = _detection()
        logs_ = zz.vision.evaluation(
            ground_truths,
            inferences,
            confidences,
            file_name=f"test_{i}.png",
            threshold=0.5,
        )
        if logs is None:
            logs = logs_
        else:
            logs = pd.concat([logs, logs_], ignore_index=True)
    zz.vision.meanap(logs)
    logs = None
    for i in range(20):
        ground_truths, inferences, confidences, gt_classes, inf_classes = _detection()
        logs_ = zz.vision.evaluation(
            ground_truths,
            inferences,
            confidences,
            gt_classes,
            inf_classes,
            file_name=f"test_{i}.png",
            threshold=0.5,
        )
        if logs is None:
            logs = logs_
        else:
            logs = pd.concat([logs, logs_], ignore_index=True)
    zz.vision.meanap(logs)


def test_cutout():
    test = cv2.imread(f"{data}/test.jpg")
    res1 = zz.vision.cutout(test, BOX_POLY)
    res2 = zz.vision.cutout(test, BOX_POLY, 128, False)
    res3 = zz.vision.cutout(test, BOX_POLY, background=128)
    zz.vision.vert([res1, res2, res3], file_name="CUTOUT")
    assert "CUTOUT.png" in os.listdir()


def test_transparent():
    test = cv2.imread(f"{data}/test.jpg")
    res1 = zz.vision.transparent(test)
    res2 = zz.vision.transparent(test, reverse=True)
    zz.vision.vert([res1, res2], file_name="TRANSPARENT")
    assert "TRANSPARENT.png" in os.listdir()


def test_paste():
    test = cv2.imread(f"{data}/test.jpg")
    poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
    target = zz.vision.cutout(test, poly, 200)
    res1 = zz.vision.paste(test, target, [200, 200, 1000, 800], resize=False, vis=True)
    res2 = zz.vision.paste(
        test, target, [200, 200, 1000, 800], resize=True, vis=True, alpha=255
    )
    poly -= zz.vision.poly2xyxy(poly)[:2]
    target = zz.vision.bbox(target, poly, color=(255, 0, 0), thickness=20)
    res3, poly3 = zz.vision.paste(
        test, target, [200, 200, 1000, 800], resize=False, poly=poly
    )
    res3 = zz.vision.bbox(res3, poly3)
    res4, poly4 = zz.vision.paste(
        test, target, [200, 200, 1000, 800], resize=True, poly=poly
    )
    res4 = zz.vision.bbox(res4, poly4)
    res5, poly5 = zz.vision.paste(
        test, target, [200, 200, 1000, 800], resize=True, poly=poly, gaussian=501
    )
    res5 = zz.vision.bbox(res5, poly5)
    zz.vision.vert([res1, res2, res3, res4, res5], file_name="PASTE")
    assert "PASTE.png" in os.listdir()


def test_ImageLoader():
    il = zz.vision.ImageLoader(data)
    assert isinstance(len(il), int)
    path, img = il[0]
    assert isinstance(path, str)
    assert isinstance(img, np.ndarray)

    il = zz.vision.ImageLoader(data, cnt=4)
    assert isinstance(len(il), int)
    path, img = il[0]
    assert isinstance(path, list)
    assert isinstance(path[0], str)
    assert isinstance(img, list)
    assert isinstance(img[0], np.ndarray)


def test_JsonImageLoader():
    jil = zz.vision.JsonImageLoader(
        f"{data}/annotation/mock/images", f"{data}/annotation/mock/labels", "name"
    )
    assert len(jil) == 1
    img, js = jil[0]
    assert isinstance(img, np.ndarray)
    assert isinstance(js, zz.util.Json)


def test_LabelStudio_no_json():
    ls = zz.vision.LabelStudio(data)
    assert isinstance(len(ls), int)
    path, annotation = ls[0]
    assert isinstance(path, str)
    assert isinstance(annotation, dict)

    ls.json()
    assert "data.json" in os.listdir(f"{data}/..")


def _test_LabelStudio_detection(path=None):
    if path is None:
        path = f"{data}/annotation/label-studio-detection.json"
    ls = zz.vision.LabelStudio(data, path)
    assert len(ls) == 1
    path, annotation = ls[0]
    assert isinstance(path, str)
    assert isinstance(annotation, dict)
    return ls


def _test_LabelStudio_segmentation(path=None):
    if path is None:
        path = f"{data}/annotation/label-studio-segmentation.json"
    ls = zz.vision.LabelStudio(data, path)
    assert len(ls) == 1
    path, annotation = ls[0]
    assert isinstance(path, str)
    assert isinstance(annotation, dict)
    return ls


def _test_YoloLoader_detection(path=None):
    vis_path = "yololoader-detection-vis"
    if path is None:
        path = f"{data}/annotation/yolo-detection"
    yolo = zz.vision.YoloLoader(
        f"{path}/images",
        f"{path}/labels",
        vis_path=vis_path,
        class_color=[(255, 0, 0)],
    )
    assert len(yolo) == 1

    img, class_list, objects = yolo[0]
    assert isinstance(img, np.ndarray)
    assert isinstance(class_list, list)
    assert isinstance(class_list[0], int)
    assert isinstance(objects, list)
    assert isinstance(objects[0], np.ndarray)
    assert "test.jpg" in os.listdir(vis_path)
    return yolo


def _test_YoloLoader_segmentation(path=None):
    vis_path = "yololoader-segmentation-vis"
    if path is None:
        path = f"{data}/annotation/yolo-segmentation"
    yolo = zz.vision.YoloLoader(
        f"{path}/images",
        f"{path}/labels",
        True,
        vis_path=vis_path,
        class_color=[(255, 0, 0)],
    )
    assert len(yolo) == 1

    img, class_list, objects = yolo[0]
    assert isinstance(img, np.ndarray)
    assert isinstance(class_list, list)
    assert isinstance(class_list[0], int)
    assert isinstance(objects, list)
    assert isinstance(objects[0], np.ndarray)
    assert "test.jpg" in os.listdir(vis_path)
    return yolo


def _test_CocoLoader_segmentation(path=None):
    vis_path = "cocoloader-segmentation-vis"
    if path is None:
        path = f"{data}/annotation/coco-segmentation/images"
    coco = zz.vision.CocoLoader(
        path,
        vis_path=vis_path,
        class_color={"Cat": (0, 0, 255)},
    )
    assert len(coco) == 1

    img, class_list, bboxes, polys = coco(0, False, True)
    assert isinstance(img, str)
    assert isinstance(class_list, list)
    assert isinstance(class_list[0], int)
    assert isinstance(bboxes, np.ndarray)
    assert isinstance(polys, list)
    assert isinstance(polys[0], np.ndarray)

    img, class_list, bboxes, polys = coco[0]
    assert isinstance(img, np.ndarray)
    assert isinstance(class_list, list)
    assert isinstance(class_list[0], str)
    assert isinstance(bboxes, np.ndarray)
    assert isinstance(polys, list)
    assert isinstance(polys[0], np.ndarray)
    assert "test.jpg" in os.listdir(vis_path)
    return coco


def test_LabelStudio_detection():
    ls = _test_LabelStudio_detection()

    classification_path = "label-studio-detection-classification"
    ls.classification(classification_path, {"Cat": "Dust"}, aug=10)
    assert "test_0_0.jpg" in os.listdir(os.path.join(classification_path, "Dust"))
    assert len(os.listdir(os.path.join(classification_path, "Dust"))) == 10

    labelme_path = "label-studio-detection-labelme"
    ls.labelme(labelme_path, {"Cat": "Dust"})
    assert "test.jpg" in os.listdir(os.path.join(labelme_path, "images"))
    assert "test.json" in os.listdir(os.path.join(labelme_path, "labels"))

    yolo_path = "label-studio-detection-yolo"
    ls.yolo(yolo_path, ["Cat"])
    assert "test.jpg" in os.listdir(os.path.join(yolo_path, "images"))
    assert "test.txt" in os.listdir(os.path.join(yolo_path, "labels"))
    _test_YoloLoader_detection(yolo_path)

    coco_path = "label-studio-detection-coco"
    ls.coco(coco_path, {"Cat": 1})
    assert f"{coco_path}.json" in os.listdir()


def test_LabelStudio_segmentation():
    ls = _test_LabelStudio_segmentation()

    classification_path = "label-studio-segmentation-classification"
    ls.classification(classification_path, rand=1, aug=10, shrink=False)
    assert "test_0_0.jpg" in os.listdir(os.path.join(classification_path, "Cat"))
    assert len(os.listdir(os.path.join(classification_path, "Cat"))) == 10

    labelme_path = "label-studio-segmentation-labelme"
    ls.labelme(labelme_path)
    assert "test.jpg" in os.listdir(os.path.join(labelme_path, "images"))
    assert "test.json" in os.listdir(os.path.join(labelme_path, "labels"))

    yolo_path = "label-studio-segmentation-yolo"
    ls.yolo(yolo_path)
    assert "test.jpg" in os.listdir(os.path.join(yolo_path, "images"))
    assert "test.txt" in os.listdir(os.path.join(yolo_path, "labels"))
    _test_YoloLoader_segmentation(yolo_path)

    coco_path = "label-studio-segmentation-coco"
    ls.coco(coco_path, {"Cat": 1})
    assert f"{coco_path}.json" in os.listdir()
    shutil.copytree(f"{data}/annotation/coco-segmentation/images", coco_path)
    _test_CocoLoader_segmentation(coco_path)


def test_YoloLoader_detection():
    yolo = _test_YoloLoader_detection()
    yolo.labelstudio()
    labelstudio_path = f"{data}/annotation/yolo-detection"
    assert "images.json" in os.listdir(labelstudio_path)
    _test_LabelStudio_detection(labelstudio_path)


def test_YoloLoader_segmentation():
    yolo = _test_YoloLoader_segmentation()
    yolo.labelstudio()
    labelstudio_path = f"{data}/annotation/yolo-segmentation"
    assert "images.json" in os.listdir(labelstudio_path)
    _test_LabelStudio_segmentation(labelstudio_path)


def test_CocoLoader_segmentation():
    coco = _test_CocoLoader_segmentation()

    yolo_path = "coco-detection-yolo"
    coco.yolo(yolo_path, ["Cat"], False)
    _test_YoloLoader_detection(yolo_path)

    yolo_path = "coco-segmentation-yolo"
    coco.yolo(yolo_path, ["Cat"], True)
    _test_YoloLoader_segmentation(yolo_path)
