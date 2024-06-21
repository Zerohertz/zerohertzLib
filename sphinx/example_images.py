import os
import random
import shutil
from datetime import datetime, timedelta
from glob import glob

import cv2
import FinanceDataReader as fdr
import numpy as np

import zerohertzLib as zz

TMP = os.path.dirname(__file__)
EXAMPLE_PATH = os.path.join(TMP, "source/_static/examples/dynamic")

NOW = datetime.now()
START_DAY = NOW - timedelta(days=30 * 18)
START_DAY = START_DAY.strftime("%Y%m%d")
TITLE = "Tesla"
DATA = fdr.DataReader("TSLA", START_DAY)

IMAGE = cv2.imread(os.path.join(TMP, "..", "test/data/test.jpg"))


# monitoring.storage.png
def example_storage():
    path = zz.monitoring.storage(".")
    shutil.move(path, f"{EXAMPLE_PATH}/monitoring.storage.png")


# plot.barv.png
def example_barv():
    data = {"Terran": 27, "Zerg": 40, "Protoss": -30}
    path = zz.plot.barv(
        data, xlab="Races", ylab="Population", title="Star Craft", dim=""
    )
    left = cv2.imread(path)
    data = {
        "xticks": ["Terran", "Zerg", "Protoss"],
        "Type A": [4, 5, 6],
        "Type B": [4, 3, 2],
        "Type C": [8, 5, 12],
        "Type D": [6, 3, 2],
    }
    path = zz.plot.barv(
        data, xlab="Races", ylab="Time [sec]", title="Star Craft", dim="%", sign=2
    )
    right = cv2.imread(path)
    zz.vision.before_after(left, right, file_name=f"{EXAMPLE_PATH}/plot.barv")


# plot.barh.png
def example_barh():
    data = {"Terran": 27, "Zerg": 40, "Protoss": -30}
    path = zz.plot.barh(
        data, xlab="Population", ylab="Races", title="Star Craft", dim=""
    )
    left = cv2.imread(path)
    data = {
        "yticks": ["Terran", "Zerg", "Protoss"],
        "Type A": [4, 5, 6],
        "Type B": [4, 3, 2],
        "Type C": [8, 5, 12],
        "Type D": [6, 3, 2],
    }
    path = zz.plot.barh(
        data, xlab="Time [Sec]", ylab="Races", title="Star Craft", dim="%", sign=2
    )
    right = cv2.imread(path)
    zz.vision.before_after(left, right, file_name=f"{EXAMPLE_PATH}/plot.barh")


# plot.hist.png
def example_hist():
    data = {
        "Terran": list(np.random.rand(1000) * 10),
        "Zerg": list(np.random.rand(1000) * 10 + 1),
        "Protoss": list(np.random.rand(1000) * 10 + 2),
    }
    path = zz.plot.hist(data, xlab="Scores", ylab="Population", title="Star Craft")
    shutil.move(path, f"{EXAMPLE_PATH}/plot.hist.png")


# plot.pie.png
def example_pie():
    data = {"Terran": 27, "Zerg": 40, "Protoss": 30}
    path = zz.plot.pie(data, dim="$", title="Star Craft")
    shutil.move(path, f"{EXAMPLE_PATH}/plot.pie.png")


# plot.plot.1.png
# plot.plot.2.png
def example_plot():
    xdata = [i for i in range(20)]
    ydata = {
        "Terran": list(np.random.rand(20) * 10),
        "Zerg": list(np.random.rand(20) * 10 + 1),
        "Protoss": list(np.random.rand(20) * 10 + 2),
    }
    path = zz.plot.plot(
        xdata, ydata, xlab="Time [Sec]", ylab="Scores", title="Star Craft"
    )
    shutil.move(path, f"{EXAMPLE_PATH}/plot.plot.1.png")
    ydata["Total"] = [
        sum(data) + 10 for data in zip(ydata["Terran"], ydata["Protoss"], ydata["Zerg"])
    ]
    path = zz.plot.plot(
        xdata, ydata, xlab="Time [Sec]", ylab="Scores", stacked=True, title="Star Craft"
    )
    shutil.move(path, f"{EXAMPLE_PATH}/plot.plot.2.png")


# plot.candle.png
def example_candle():
    path = zz.plot.candle(DATA, TITLE)
    left = cv2.imread(path)
    signals = zz.quant.macd(DATA)
    path = zz.plot.candle(DATA, "MACD", signals=signals)
    right = cv2.imread(path)
    zz.vision.before_after(left, right, file_name=f"{EXAMPLE_PATH}/plot.candle")


# plot.scatter.png
def example_scatter():
    data = {
        "Terran": [list(np.random.rand(200) * 10), list(np.random.rand(200) * 10)],
        "Zerg": [list(np.random.rand(200) * 5 - 1), list(np.random.rand(200) * 5 + 1)],
        "Protoss": [
            list(np.random.rand(200) * 10 + 3),
            list(np.random.rand(200) * 10 - 2),
        ],
    }
    path = zz.plot.scatter(
        data, xlab="Cost [Mineral]", ylab="Scores", title="Star Craft", markersize=400
    )
    shutil.move(path, f"{EXAMPLE_PATH}/plot.scatter.png")


# plot.table.png
def example_table():
    data = [
        ["123", 123, 123.4],
        [123.4, "123", 123],
        [123, 123.4, "123"],
        ["123", 123, 123.4],
    ]
    col = ["Terran", "Zerg", "Protoss"]
    row = ["test1", "test2", "test3", "test4"]
    left = zz.plot.table(data, col, row, title="Star Craft")
    right = zz.plot.table(data, col, row, title="Star Craft2", fontsize=50)
    zz.vision.before_after(
        cv2.imread(left), cv2.imread(right), file_name=f"{EXAMPLE_PATH}/plot.table"
    )


# quant.moving_average.png
def example_moving_average():
    signals = zz.quant.moving_average(DATA)
    path = zz.plot.candle(DATA, "Moving Average", signals=signals)
    shutil.move(path, f"{EXAMPLE_PATH}/quant.moving_average.png")


# quant.rsi.png
def example_rsi():
    signals = zz.quant.rsi(DATA)
    path = zz.plot.candle(DATA, "RSI", signals=signals)
    shutil.move(path, f"{EXAMPLE_PATH}/quant.rsi.png")


# quant.bollinger_bands.png
def example_bollinger_bands():
    signals = zz.quant.bollinger_bands(DATA)
    path = zz.plot.candle(DATA, "Bollinger Bands", signals=signals)
    shutil.move(path, f"{EXAMPLE_PATH}/quant.bollinger_bands.png")


# quant.momentum.png
def example_momentum():
    signals = zz.quant.momentum(DATA)
    path = zz.plot.candle(DATA, "Momentum", signals=signals)
    shutil.move(path, f"{EXAMPLE_PATH}/quant.momentum.png")


# quant.macd.png
def example_macd():
    signals = zz.quant.macd(DATA)
    path = zz.plot.candle(DATA, "MACD", signals=signals)
    shutil.move(path, f"{EXAMPLE_PATH}/quant.macd.png")


# vision.before_after.1.png
# vision.before_after.2.png
def example_before_after():
    before = IMAGE.copy()
    after = cv2.GaussianBlur(before, (0, 0), 25)
    after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    zz.vision.before_after(
        before, after, quality=10, file_name=f"{EXAMPLE_PATH}/vision.before_after.1"
    )
    after = cv2.resize(before, (100, 100))
    zz.vision.before_after(
        before,
        after,
        [20, 40, 30, 60],
        file_name=f"{EXAMPLE_PATH}/vision.before_after.2",
    )


# vision.grid.png
def example_grid():
    imgs = [
        cv2.resize(IMAGE, (random.randrange(300, 1000), random.randrange(300, 1000)))
        for _ in range(8)
    ]
    imgs[2] = cv2.cvtColor(imgs[2], cv2.COLOR_BGR2GRAY)
    imgs[3] = cv2.cvtColor(imgs[3], cv2.COLOR_BGR2BGRA)
    zz.vision.grid(imgs, file_name="grid_1")
    zz.vision.grid(imgs, color=(0, 255, 0), file_name="grid_2")
    zz.vision.grid(imgs, color=(0, 0, 0, 0), file_name="grid_3")
    zz.vision.vert(
        [cv2.imread(f"grid_{i+1}.png", cv2.IMREAD_UNCHANGED) for i in range(3)],
        file_name=f"{EXAMPLE_PATH}/vision.grid",
    )


# vision.vert.png
def example_vert():
    imgs = [
        cv2.resize(IMAGE, (random.randrange(300, 600), random.randrange(300, 600)))
        for _ in range(5)
    ]
    zz.vision.vert(imgs, file_name=f"{EXAMPLE_PATH}/vision.vert")


# vision.poly2mask.png
def example_poly2mask():
    poly = [[10, 10], [20, 10], [30, 40], [20, 60], [10, 20]]
    mask1 = zz.vision.poly2mask(poly, (70, 100))
    poly = np.array(poly)
    mask2 = zz.vision.poly2mask([poly, poly - 10, poly + 20], (70, 100))
    zz.vision.vert(
        [
            mask1.astype(np.uint8) * 255,
            np.transpose(mask2, (1, 2, 0)).astype(np.uint8) * 255,
        ],
        file_name=f"{EXAMPLE_PATH}/vision.vert",
    )


# vision.pad.png
def example_pad():
    img = cv2.cvtColor(IMAGE, cv2.COLOR_BGRA2GRAY)
    res1 = cv2.resize(img, (500, 1000))
    res1, _ = zz.vision.pad(res1, (1000, 1000), color=(0, 255, 0))
    res2 = cv2.resize(IMAGE, (1000, 500))
    res2, _ = zz.vision.pad(res2, (1000, 1000))
    img = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2BGRA)
    res3 = cv2.resize(img, (500, 1000))
    res3, _ = zz.vision.pad(res3, (1000, 1000), color=(0, 0, 255, 128))
    poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
    res4 = cv2.resize(IMAGE, (2000, 1000))
    res4 = zz.vision.bbox(res4, poly, color=(255, 0, 0), thickness=20)
    res4, poly = zz.vision.pad(res4, (1000, 1000), poly=poly)
    res4 = zz.vision.bbox(res4, poly, color=(0, 0, 255))
    poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
    res5 = cv2.resize(IMAGE, (2000, 1000))
    res5 = zz.vision.bbox(res5, poly, color=(255, 0, 0), thickness=20)
    res5, info = zz.vision.pad(res5, (1000, 1000), color=(128, 128, 128))
    poly = poly * info[0] + info[1:]
    res5 = zz.vision.bbox(res5, poly, color=(0, 0, 255))
    zz.vision.vert(
        [res1, res2, res3, res4, res5], file_name=f"{EXAMPLE_PATH}/vision.pad"
    )


# vision.cutout.png
def example_cutout():
    poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
    res1 = zz.vision.cutout(IMAGE, poly)
    res2 = zz.vision.cutout(IMAGE, poly, 128, False)
    res3 = zz.vision.cutout(IMAGE, poly, background=128)
    zz.vision.vert([res1, res2, res3], file_name=f"{EXAMPLE_PATH}/vision.cutout")


# vision.transparent.png
def example_transparent():
    res1 = zz.vision.transparent(IMAGE)
    res2 = zz.vision.transparent(IMAGE, reverse=True)
    zz.vision.vert([res1, res2], file_name=f"{EXAMPLE_PATH}/vision.transparent")


# vision.bbox.png
def example_bbox():
    box = np.array([[100, 200], [100, 1000], [1200, 1000], [1200, 200]])
    res1 = zz.vision.bbox(IMAGE, box, thickness=10)
    boxes = np.array([[250, 200, 100, 100], [600, 600, 800, 200], [900, 300, 300, 400]])
    res2 = zz.vision.bbox(IMAGE, boxes, (0, 255, 0), thickness=10)
    zz.vision.vert([res1, res2], file_name=f"{EXAMPLE_PATH}/vision.bbox")


# vision.mask.png
def example_mask():
    H, W, _ = IMAGE.shape
    cnt = 30
    mks = np.zeros((cnt, H, W), np.uint8)
    for mks_ in mks:
        center_x = random.randint(0, W)
        center_y = random.randint(0, H)
        radius = random.randint(30, 200)
        cv2.circle(mks_, (center_x, center_y), radius, (True), -1)
    mks = mks.astype(bool)
    res1 = zz.vision.mask(IMAGE, mks)
    cls = [i for i in range(cnt)]
    class_list = [cls[random.randint(0, 5)] for _ in range(cnt)]
    class_color = {}
    for c in cls:
        class_color[c] = [random.randint(0, 255) for _ in range(3)]
    res2 = zz.vision.mask(IMAGE, mks, class_list=class_list, class_color=class_color)
    poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
    res3 = zz.vision.mask(IMAGE, poly=poly)
    poly = zz.vision.xyxy2poly(
        zz.vision.poly2xyxy((np.random.rand(cnt, 4, 2) * (W, H)))
    )
    res4 = zz.vision.mask(
        IMAGE, poly=poly, class_list=class_list, class_color=class_color
    )
    zz.vision.vert([res1, res2, res3, res4], file_name=f"{EXAMPLE_PATH}/vision.mask")


# vision.text.png
def example_text():
    box = np.array([[100, 200], [100, 1000], [1200, 1000], [1200, 200]])
    res1 = zz.vision.text(IMAGE, box, "먼지야")
    boxes = np.array([[250, 200, 100, 100], [600, 600, 800, 200], [900, 300, 300, 400]])
    res2 = zz.vision.text(IMAGE, boxes, ["먼지야", "먼지야", "먼지야"], vis=True)
    zz.vision.vert([res1, res2], file_name=f"{EXAMPLE_PATH}/vision.text")


# vision.paste.png
def example_paste():
    poly = np.array([[100, 400], [400, 400], [800, 900], [400, 1100], [100, 800]])
    target = zz.vision.cutout(IMAGE, poly, 200)
    res1 = zz.vision.paste(IMAGE, target, [200, 200, 1000, 800], resize=False, vis=True)
    res2 = zz.vision.paste(
        IMAGE, target, [200, 200, 1000, 800], resize=True, vis=True, alpha=255
    )
    poly -= zz.vision.poly2xyxy(poly)[:2]
    target = zz.vision.bbox(target, poly, color=(255, 0, 0), thickness=20)
    res3, poly3 = zz.vision.paste(
        IMAGE, target, [200, 200, 1000, 800], resize=False, poly=poly
    )
    res3 = zz.vision.bbox(res3, poly3)
    res4, poly4 = zz.vision.paste(
        IMAGE, target, [200, 200, 1000, 800], resize=True, poly=poly
    )
    res4 = zz.vision.bbox(res4, poly4)
    res5, poly5 = zz.vision.paste(
        IMAGE, target, [200, 200, 1000, 800], resize=True, poly=poly, gaussian=501
    )
    res5 = zz.vision.bbox(res5, poly5)

    zz.vision.vert(
        [res1, res2, res3, res4, res5], file_name=f"{EXAMPLE_PATH}/vision.paste"
    )


def remove():
    for path in glob("./*.png"):
        os.remove(path)


if __name__ == "__main__":
    zz.util.rmtree(EXAMPLE_PATH)
    # zerohertzLib.monitoring
    example_storage()
    # zerohertzLib.plot
    example_barv()
    example_barh()
    example_hist()
    example_pie()
    example_plot()
    example_candle()
    example_scatter()
    example_table()
    # zerohertzLib.quant
    example_moving_average()
    example_rsi()
    example_bollinger_bands()
    example_momentum()
    example_macd()
    # zerohertzLib.vision
    example_before_after()
    example_grid()
    example_vert()
    example_poly2mask()
    example_pad()
    example_cutout()
    example_transparent()
    example_bbox()
    example_mask()
    example_text()
    example_paste()
    # clean
    remove()
