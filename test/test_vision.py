import os
import random

import cv2
import numpy as np

import zerohertzLib as zz

tmp = __file__.replace("test_vision.py", "")


def test_img2gif():
    for i in range(5):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(f"{tmp}{i}.png", img)
    zz.vision.img2gif(tmp, "img")
    assert "img.gif" in os.listdir()


def test_vid2gif():
    zz.vision.vid2gif(f"{tmp}test.mov", "vid", quality=20)
    assert "vid.gif" in os.listdir()


def test_before_after_org():
    before = cv2.imread(f"{tmp}test.jpg")
    after = cv2.GaussianBlur(before, (0, 0), 25)
    after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    zz.vision.before_after(before, after, quality=10, output_filename="ba1")
    assert "ba1.png" in os.listdir()


def test_before_after_crop():
    before = cv2.imread(f"{tmp}test.jpg")
    after = cv2.resize(before, (100, 100))
    zz.vision.before_after(before, after, [20, 40, 30, 60], output_filename="ba2")
    assert "ba2.png" in os.listdir()


def test_grid_vertical():
    test = cv2.imread(f"{tmp}test.jpg")
    imgs = [(test + np.random.rand(*test.shape)).astype(np.uint8) for _ in range(8)]
    imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
    zz.vision.grid(*imgs, output_filename="grid_vertical")
    assert "grid_vertical.png" in os.listdir()


def test_grid_horizontal():
    test = cv2.imread(f"{tmp}test.jpg")
    shape = test.shape
    test = cv2.resize(test, (shape[0], shape[1]))
    imgs = [(test + np.random.rand(*test.shape)).astype(np.uint8) for _ in range(8)]
    imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
    zz.vision.grid(*imgs, output_filename="grid_horizontal")
    assert "grid_horizontal.png" in os.listdir()


def test_bbox_bgr():
    img = cv2.imread(f"{tmp}test.jpg")
    box = np.array(
        [
            [100, 200],
            [100, 1500],
            [1400, 1500],
            [1400, 200],
        ]
    )
    BGR = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_BGR.png", BGR)
    assert "BBOX_BGR.png" in os.listdir()


def test_bbox_bgra():
    img = cv2.imread(f"{tmp}test.jpg")
    box = np.array(
        [
            [100, 200],
            [100, 1500],
            [1400, 1500],
            [1400, 200],
        ]
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    BGRA = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_BGRA.png", BGRA)
    assert "BBOX_BGRA.png" in os.listdir()


def test_bbox_gray():
    img = cv2.imread(f"{tmp}test.jpg")
    box = np.array(
        [
            [100, 200],
            [100, 1500],
            [1400, 1500],
            [1400, 200],
        ]
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    GRAY = zz.vision.bbox(img, box, thickness=10)
    cv2.imwrite("BBOX_GRAY.png", GRAY)
    assert "BBOX_GRAY.png" in os.listdir()


def test_masks_bgr():
    img = cv2.imread(f"{tmp}test.jpg")
    H, W, _ = img.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mask in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(100, 400)
        cv2.circle(mask, (center_x, center_y), radius, (True), -1)
    mks = mks.astype(bool)
    BGR = zz.vision.masks(img, mks)
    cv2.imwrite("MASK_BGR.png", BGR)
    assert "MASK_BGR.png" in os.listdir()


def test_masks_bgra():
    img = cv2.imread(f"{tmp}test.jpg")
    H, W, _ = img.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mask in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(100, 400)
        cv2.circle(mask, (center_x, center_y), radius, (True), -1)
    mks = mks.astype(bool)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    BGRA = zz.vision.masks(img, mks, [random.randint(0, 255) for _ in range(3)])
    cv2.imwrite("MASK_BGRA.png", BGRA)
    assert "MASK_BGRA.png" in os.listdir()


def test_masks_gray_int():
    img = cv2.imread(f"{tmp}test.jpg")
    H, W, _ = img.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mask in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(100, 400)
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
    img = cv2.imread(f"{tmp}test.jpg")
    H, W, _ = img.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mask in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(100, 400)
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


def test_text_bgr():
    img = cv2.imread(f"{tmp}test.jpg")
    box = np.array(
        [
            [100, 200],
            [100, 1500],
            [1400, 1500],
            [1400, 200],
        ]
    )
    BGR = zz.vision.text(img, box, "먼지야")
    cv2.imwrite("TEXT_BGR.png", BGR)
    assert "TEXT_BGR.png" in os.listdir()


def test_text_bgra():
    img = cv2.imread(f"{tmp}test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    box = np.array(
        [
            [100, 200],
            [100, 1500],
            [1400, 1500],
            [1400, 200],
        ]
    )
    box = np.array([img.shape[1] / 2, img.shape[0] / 3, 500, 1000])
    box = zz.vision.xywh2xyxy(box)
    BGRA = zz.vision.text(img, box, "오래오래", (0, 255, 0))
    BGRA = zz.vision.bbox(BGRA, box, thickness=10)
    box = np.array([img.shape[1] / 2, img.shape[0] / 3 * 2, 1000, 500])
    box = zz.vision.xywh2xyxy(box)
    BGRA = zz.vision.text(BGRA, box, "오래오래", (255, 0, 0))
    BGRA = zz.vision.bbox(BGRA, box, thickness=10)
    cv2.imwrite("TEXT_BGRA.png", BGRA)
    assert "TEXT_BGRA.png" in os.listdir()


def test_text_gray():
    img = cv2.imread(f"{tmp}test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    box = np.array(
        [
            [100, 200],
            [100, 1500],
            [1400, 1500],
            [1400, 200],
        ]
    )
    GRAY = zz.vision.text(img, box, "행복해라")
    cv2.imwrite("TEXT_GRAY.png", GRAY)
    assert "TEXT_GRAY.png" in os.listdir()


def test_xyxy2xywh():
    box = np.array(
        [
            [200, 100],
            [1500, 100],
            [1500, 1500],
            [200, 1500],
        ]
    )
    assert (zz.vision.xyxy2xywh(box) == np.array([850.0, 800.0, 1300.0, 1400.0])).all()


def test_xywh2xyxy():
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
