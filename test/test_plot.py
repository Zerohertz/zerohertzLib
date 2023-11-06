import os

import numpy as np

import zerohertzLib as zz


def test_bar():
    zz.plot.bar(
        {"테란": 27, "저그": 40, "프로토스": 30},
        xlab="종족",
        ylab="인구 [명]",
        title="Star Craft (bar)",
    )
    assert "star_craft_(bar).png" in os.listdir()


def test_barh():
    zz.plot.barh(
        {"테란": 27, "저그": 40, "프로토스": 30},
        xlab="인구 [명]",
        ylab="종족",
        title="Star Craft (barh)",
    )
    assert "star_craft_(barh).png" in os.listdir()


def test_hist():
    zz.plot.hist(
        {
            "테란": list(np.random.rand(1000) * 10),
            "저그": list(np.random.rand(1000) * 10 + 1),
            "프로토스": list(np.random.rand(1000) * 10 + 2),
        },
        xlab="성적 [점]",
        ylab="인원 [명]",
        title="Star Craft (hist)",
    )
    assert "star_craft_(hist).png" in os.listdir()


def test_plot():
    zz.plot.plot(
        [i for i in range(20)],
        {
            "테란": list(np.random.rand(20) * 10),
            "저그": list(np.random.rand(20) * 10 + 1),
            "프로토스": list(np.random.rand(20) * 10 + 2),
        },
        xlab="시간 [초]",
        ylab="성적 [점]",
        title="Star Craft (plot)",
    )
    assert "star_craft_(plot).png" in os.listdir()
