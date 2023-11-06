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
