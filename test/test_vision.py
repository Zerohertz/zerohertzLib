import os
import random

import cv2
import numpy as np

import zerohertzLib as zz

tmp = os.path.dirname(__file__)
data = os.path.join(tmp, "data")

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
    zz.vision.before_after(before, after, quality=10, filename="BA1")
    assert "BA1.png" in os.listdir()


def test_before_after_crop():
    before = cv2.imread(f"{data}/test.jpg")
    after = cv2.resize(before, (100, 100))
    after = cv2.cvtColor(after, cv2.COLOR_BGR2BGRA)
    zz.vision.before_after(before, after, [20, 40, 30, 60], filename="BA2")
    assert "BA2.png" in os.listdir()


def test_grid_vertical():
    test = cv2.imread(f"{data}/test.jpg")
    test = cv2.resize(test, (200, 300))
    imgs = [(test + np.random.rand(*test.shape)).astype(np.uint8) for _ in range(8)]
    imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
    imgs[3] = cv2.cvtColor(imgs[3], cv2.COLOR_BGR2BGRA)
    zz.vision.grid(*imgs, filename="GRID_VERTICAL")
    assert "GRID_VERTICAL.png" in os.listdir()


def test_grid_horizontal():
    test = cv2.imread(f"{data}/test.jpg")
    test = cv2.resize(test, (300, 200))
    imgs = [(test + np.random.rand(*test.shape)).astype(np.uint8) for _ in range(8)]
    imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
    imgs[3] = cv2.cvtColor(imgs[3], cv2.COLOR_BGR2BGRA)
    zz.vision.grid(*imgs, filename="GRID_HORIZONTAL")
    assert "GRID_HORIZONTAL.png" in os.listdir()


def test_vert():
    test = cv2.imread(f"{data}/test.jpg")
    imgs = [
        cv2.resize(test, (random.randrange(300, 600), random.randrange(300, 600)))
        for _ in range(5)
    ]
    imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
    imgs[3] = cv2.cvtColor(imgs[3], cv2.COLOR_BGR2BGRA)
    zz.vision.vert(*imgs, filename="VERT")
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


def test_bbox_bgr_xyxwh():
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
    BGRA = zz.vision.masks(img, mks, color=[random.randint(0, 255) for _ in range(3)])
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


def test_masks_bgra_str():
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
    cls = ["a", "b", "c"]
    class_list = [cls[random.randint(0, 2)] for _ in range(cnt)]
    class_color = {}
    for c in cls:
        class_color[c] = [random.randint(0, 255) for _ in range(3)]
    BGRA = zz.vision.masks(img, mks, class_list=class_list, class_color=class_color)
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
    BGRA = zz.vision.text(img, boxes, ["오래오래", "오래오래", "오래오래"], (0, 255, 0), vis=True)
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
    BGRA = zz.vision.text(img, boxes, ["오래오래", "오래오래", "오래오래"], (0, 255, 0), vis=True)
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
