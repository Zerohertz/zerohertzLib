import os
import random

import cv2
import numpy as np

import zerohertzLib as zz

tmp = os.path.dirname(__file__)
data = os.path.join(tmp, "data")

BOX_XYXY = np.array([[100, 200], [100, 1000], [1200, 1000], [1200, 200]])
BOX_XYWH = np.array([650, 600, 1100, 800])
BOXES_XYXY = np.array(
    [
        [[200, 150], [300, 150], [300, 250], [200, 250]],
        [[200, 500], [1000, 500], [1000, 700], [200, 700]],
        [[750, 100], [1050, 100], [1050, 500], [750, 500]],
    ]
)
BOXES_XYWH = np.array(
    [[250, 200, 100, 100], [600, 600, 800, 200], [900, 300, 300, 400]]
)


def test_img2gif():
    for i in range(5):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(f"{data}/{i}.png", img)
    zz.vision.img2gif(data, "img")
    assert "img.gif" in os.listdir()


def test_vid2gif():
    zz.vision.vid2gif(f"{data}/test.mov", "vid", quality=20)
    assert "vid.gif" in os.listdir()


def test_before_after_org():
    before = cv2.imread(f"{data}/test.jpg")
    after = cv2.GaussianBlur(before, (0, 0), 25)
    after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    zz.vision.before_after(before, after, quality=10, output_filename="ba1")
    assert "ba1.png" in os.listdir()


def test_before_after_crop():
    before = cv2.imread(f"{data}/test.jpg")
    after = cv2.resize(before, (100, 100))
    zz.vision.before_after(before, after, [20, 40, 30, 60], output_filename="ba2")
    assert "ba2.png" in os.listdir()


def test_grid_vertical():
    test = cv2.imread(f"{data}/test.jpg")
    test = cv2.resize(test, (200, 300))
    imgs = [(test + np.random.rand(*test.shape)).astype(np.uint8) for _ in range(8)]
    imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
    zz.vision.grid(*imgs, output_filename="grid_vertical")
    assert "grid_vertical.png" in os.listdir()


def test_grid_horizontal():
    test = cv2.imread(f"{data}/test.jpg")
    test = cv2.resize(test, (300, 200))
    imgs = [(test + np.random.rand(*test.shape)).astype(np.uint8) for _ in range(8)]
    imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
    zz.vision.grid(*imgs, output_filename="grid_horizontal")
    assert "grid_horizontal.png" in os.listdir()


def test_bbox_bgr_xyxy():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_XYXY
    BGR = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_BGR_XYXY.png", BGR)
    assert "BBOX_BGR_XYXY.png" in os.listdir()


def test_bbox_bgra_xyxy():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_XYXY
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    BGRA = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_BGRA_XYXY.png", BGRA)
    assert "BBOX_BGRA_XYXY.png" in os.listdir()


def test_bbox_gray_xyxy():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_XYXY
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    GRAY = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_GRAY_XYXY.png", GRAY)
    assert "BBOX_GRAY_XYXY.png" in os.listdir()


def test_bboxes_bgr_xyxy():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_XYXY
    BGR = zz.vision.bbox(img, boxes, thickness=10)
    cv2.imwrite("BBOXES_BGR_XYXY.png", BGR)
    assert "BBOXES_BGR_XYXY.png" in os.listdir()


def test_bboxes_bgra_xyxy():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_XYXY
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    BGRA = zz.vision.bbox(img, boxes, thickness=10)
    cv2.imwrite("BBOXES_BGRA_XYXY.png", BGRA)
    assert "BBOXES_BGRA_XYXY.png" in os.listdir()


def test_bboxes_gray_xyxy():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_XYXY
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    GRAY = zz.vision.bbox(img, boxes, thickness=10)
    cv2.imwrite("BBOXES_GRAY_XYXY.png", GRAY)
    assert "BBOXES_GRAY_XYXY.png" in os.listdir()


def test_bbox_bgr_xyxwh():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_XYWH
    BGR = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_BGR_XYWH.png", BGR)
    assert "BBOX_BGR_XYWH.png" in os.listdir()


def test_bbox_bgra_xywh():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_XYWH
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    BGRA = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_BGRA_XYWH.png", BGRA)
    assert "BBOX_BGRA_XYWH.png" in os.listdir()


def test_bbox_gray_xywh():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_XYWH
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    GRAY = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_GRAY_XYWH.png", GRAY)
    assert "BBOX_GRAY_XYWH.png" in os.listdir()


def test_bboxes_bgr_xywh():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_XYWH
    BGR = zz.vision.bbox(img, boxes, thickness=10)
    cv2.imwrite("BBOXES_BGR_XYWH.png", BGR)
    assert "BBOXES_BGR_XYWH.png" in os.listdir()


def test_bboxes_bgra_xywh():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_XYWH
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    BGRA = zz.vision.bbox(img, boxes, thickness=10)
    cv2.imwrite("BBOXES_BGRA_XYWH.png", BGRA)
    assert "BBOXES_BGRA_XYWH.png" in os.listdir()


def test_bboxes_gray_xywh():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_XYWH
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    GRAY = zz.vision.bbox(img, boxes, thickness=10)
    cv2.imwrite("BBOXES_GRAY_XYWH.png", GRAY)
    assert "BBOXES_GRAY_XYWH.png" in os.listdir()


def test_masks_bgr():
    img = cv2.imread(f"{data}/test.jpg")
    H, W, _ = img.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mask in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(30, 200)
        cv2.circle(mask, (center_x, center_y), radius, (True), -1)
    mks = mks.astype(bool)
    BGR = zz.vision.masks(img, mks)
    cv2.imwrite("MASK_BGR.png", BGR)
    assert "MASK_BGR.png" in os.listdir()


def test_masks_bgra():
    img = cv2.imread(f"{data}/test.jpg")
    H, W, _ = img.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mask in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(30, 200)
        cv2.circle(mask, (center_x, center_y), radius, (True), -1)
    mks = mks.astype(bool)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    BGRA = zz.vision.masks(img, mks, [random.randint(0, 255) for _ in range(3)])
    cv2.imwrite("MASK_BGRA.png", BGRA)
    assert "MASK_BGRA.png" in os.listdir()


def test_masks_gray_int():
    img = cv2.imread(f"{data}/test.jpg")
    H, W, _ = img.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mask in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(30, 200)
        cv2.circle(mask, (center_x, center_y), radius, (True), -1)
    mks = mks.astype(bool)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cls = [i for i in range(cnt)]
    class_list = [cls[random.randint(0, 2)] for _ in range(cnt)]
    class_color = {}
    for c in cls:
        class_color[c] = [random.randint(0, 255) for _ in range(3)]
    GRAY = zz.vision.masks(
        img, mks, class_list=class_list, class_color=class_color, alpha=1
    )
    cv2.imwrite("MASK_GRAY_INT.png", GRAY)
    assert "MASK_GRAY_INT.png" in os.listdir()


def test_masks_gray_str():
    img = cv2.imread(f"{data}/test.jpg")
    H, W, _ = img.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mask in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(30, 200)
        cv2.circle(mask, (center_x, center_y), radius, (True), -1)
    mks = mks.astype(bool)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cls = ["a", "b", "c"]
    class_list = [cls[random.randint(0, 2)] for _ in range(cnt)]
    class_color = {}
    for c in cls:
        class_color[c] = [random.randint(0, 255) for _ in range(3)]
    GRAY = zz.vision.masks(img, mks, class_list=class_list, class_color=class_color)
    cv2.imwrite("MASK_GRAY_STR.png", GRAY)
    assert "MASK_GRAY_STR.png" in os.listdir()


def test_text_bgr_xyxy():
    img = cv2.imread(f"{data}/test.jpg")
    box = BOX_XYXY
    BGR = zz.vision.text(img, box, "먼지야", vis=True)
    cv2.imwrite("TEXT_BGR_XYXY.png", BGR)
    assert "TEXT_BGR_XYXY.png" in os.listdir()


def test_text_bgra_xyxy():
    img = cv2.imread(f"{data}/test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    boxes = BOXES_XYXY
    BGRA = zz.vision.text(img, boxes, ["오래오래", "오래오래", "오래오래"], (0, 255, 0), vis=True)
    cv2.imwrite("TEXT_BGRA_XYXY.png", BGRA)
    assert "TEXT_BGRA_XYXY.png" in os.listdir()


def test_text_gray_xyxy():
    img = cv2.imread(f"{data}/test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    box = BOX_XYXY
    GRAY = zz.vision.text(img, box, "행복해라", vis=True)
    cv2.imwrite("TEXT_GRAY_XYXY.png", GRAY)
    assert "TEXT_GRAY_XYXY.png" in os.listdir()


def test_text_bgr_xywh():
    img = cv2.imread(f"{data}/test.jpg")
    boxes = BOXES_XYWH
    BGR = zz.vision.text(img, boxes, ["먼지야", "먼지야", "먼지야"], vis=True)
    cv2.imwrite("TEXT_BGR_XYWH.png", BGR)
    assert "TEXT_BGR_XYWH.png" in os.listdir()


def test_text_bgra_xywh():
    img = cv2.imread(f"{data}/test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    boxes = BOXES_XYWH
    BGRA = zz.vision.text(img, boxes, ["오래오래", "오래오래", "오래오래"], (0, 255, 0), vis=True)
    cv2.imwrite("TEXT_BGRA_XYWH.png", BGRA)
    assert "TEXT_BGRA_XYWH.png" in os.listdir()


def test_text_gray_xywh():
    img = cv2.imread(f"{data}/test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    box = BOX_XYWH
    GRAY = zz.vision.text(img, box, "행복해라", vis=True)
    cv2.imwrite("TEXT_GRAY_XYWH.png", GRAY)
    assert "TEXT_GRAY_XYXY.png" in os.listdir()


def test_xyxy2xywh_bbox():
    box = np.array(
        [
            [200, 100],
            [1500, 100],
            [1500, 1500],
            [200, 1500],
        ]
    )
    assert (zz.vision.xyxy2xywh(box) == np.array([850, 800, 1300, 1400])).all()


def test_xyxy2xywh_bboxes():
    boxes = np.array(
        [
            [[200, 100], [1500, 100], [1500, 1500], [200, 1500]],
            [[150, 100], [450, 100], [450, 500], [150, 500]],
            [[1050, 1050], [1350, 1050], [1350, 1350], [1050, 1350]],
        ]
    )
    assert (
        zz.vision.xyxy2xywh(boxes)
        == np.array(
            [[850, 800, 1300, 1400], [300, 300, 300, 400], [1200, 1200, 300, 300]]
        )
    ).all()


def test_xywh2xyxy_bbox():
    box = np.array([850, 800, 1300, 1400])
    assert (
        zz.vision.xywh2xyxy(box)
        == np.array(
            [
                [200, 100],
                [1500, 100],
                [1500, 1500],
                [200, 1500],
            ]
        )
    ).all()


def test_xywh2xyxy_bboxes():
    boxes = np.array(
        [[850, 800, 1300, 1400], [300, 300, 300, 400], [1200, 1200, 300, 300]]
    )
    assert (
        zz.vision.xywh2xyxy(boxes)
        == np.array(
            [
                [[200, 100], [1500, 100], [1500, 1500], [200, 1500]],
                [[150, 100], [450, 100], [450, 500], [150, 500]],
                [[1050, 1050], [1350, 1050], [1350, 1350], [1050, 1350]],
            ]
        )
    ).all()
