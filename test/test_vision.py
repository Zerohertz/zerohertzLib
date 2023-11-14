import os

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


def test_before_after():
    before = cv2.imread(f"{tmp}test.jpg")
    after = cv2.GaussianBlur(before, (0, 0), 25)
    after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    zz.vision.before_after(before, after, quality=10, output_filename="ba1")
    assert "ba1.png" in os.listdir()
    before = cv2.imread(f"{tmp}test.jpg")
    after = cv2.resize(before, (100, 100))
    zz.vision.before_after(before, after, [20, 40, 30, 60], output_filename="ba2")
    assert "ba2.png" in os.listdir()


def test_grid():
    test = cv2.imread(f"{tmp}test.jpg")
    imgs = [(test + np.random.rand(*test.shape)).astype(np.uint8) for _ in range(8)]
    imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
    zz.vision.grid(*imgs, output_filename="grid")
    assert "grid.png" in os.listdir()
