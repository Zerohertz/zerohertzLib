import os

import numpy as np

import zerohertzLib as zz


def test_barv():
    path = zz.plot.barv(
        {"테란": 27, "저그": 40, "프로토스": 30},
        xlab="종족",
        ylab="인구 [명]",
        title="Star Craft (barv)",
    )
    assert path.split("/")[-1] in os.listdir()


def test_barv_palette():
    path = zz.plot.barv(
        {"테란": 27},
        xlab="종족",
        ylab="인구 [명]",
        title="Star Craft (barv, palette)",
        colors="Set2",
    )
    assert path.split("/")[-1] in os.listdir()


def test_barv_colors():
    path = zz.plot.barv(
        {"테란": 27, "저그": 40, "프로토스": 30},
        xlab="종족",
        ylab="인구 [명]",
        title="Star Craft (barv, colors)",
        colors=["#800a0a", "#0a800a", "#0a0a80"],
    )
    assert path.split("/")[-1] in os.listdir()


def test_barv_stacked():
    data = {
        "xticks": ["테란", "저그", "프로토스"],
        "Type A": [4, 5, 6],
        "Type B": [4, 3, 2],
        "Type C": [8, 5, 12],
        "Type D": [6, 3, 2],
    }
    path = zz.plot.barv(
        data,
        xlab="종족",
        ylab="시간 [초]",
        title="Star Craft (barv, stacked)",
    )
    assert path.split("/")[-1] in os.listdir()


def test_barv_stacked_palette():
    data = {
        "xticks": ["테란", "저그", "프로토스"],
        "Type A": [4, 5, 6],
    }
    path = zz.plot.barv(
        data,
        xlab="종족",
        ylab="시간 [초]",
        title="Star Craft (barv, stacked, palette)",
        colors="Set2",
    )
    assert path.split("/")[-1] in os.listdir()


def test_barv_stacked_colors():
    data = {
        "xticks": ["테란", "저그", "프로토스"],
        "Type A": [4, 5, 6],
        "Type B": [4, 3, 2],
        "Type C": [8, 5, 12],
        "Type D": [6, 3, 2],
    }
    path = zz.plot.barv(
        data,
        xlab="종족",
        ylab="시간 [초]",
        title="Star Craft (barv, stacked, colors)",
        colors=["#800a0a", "#0a800a", "#0a0a80", "#000000"],
    )
    assert path.split("/")[-1] in os.listdir()


def test_barh():
    path = zz.plot.barh(
        {"테란": 27, "저그": 40, "프로토스": 30},
        xlab="인구 [명]",
        ylab="종족",
        title="Star Craft (barh)",
    )
    assert path.split("/")[-1] in os.listdir()


def test_barh_stacked():
    data = {
        "yticks": ["테란", "저그", "프로토스"],
        "Type A": [4, 5, 6],
        "Type B": [4, 3, 2],
        "Type C": [8, 5, 12],
        "Type D": [6, 3, 2],
    }
    path = zz.plot.barh(
        data,
        xlab="시간 [초]",
        ylab="종족",
        title="Star Craft (barh, stacked)",
    )
    assert path.split("/")[-1] in os.listdir()


def test_hist():
    path = zz.plot.hist(
        {
            "테란": list(np.random.rand(1000) * 10),
            "저그": list(np.random.rand(1000) * 10 + 1),
            "프로토스": list(np.random.rand(1000) * 10 + 2),
        },
        xlab="성적 [점]",
        ylab="인원 [명]",
        title="Star Craft (hist)",
    )
    assert path.split("/")[-1] in os.listdir()


def test_hist_palette():
    path = zz.plot.hist(
        {
            "테란": list(np.random.rand(1000) * 10),
        },
        xlab="성적 [점]",
        ylab="인원 [명]",
        title="Star Craft (hist, palette)",
        colors="Set2",
    )
    assert path.split("/")[-1] in os.listdir()


def test_hist_colors():
    path = zz.plot.hist(
        {
            "테란": list(np.random.rand(1000) * 10),
            "저그": list(np.random.rand(1000) * 10 + 1),
            "프로토스": list(np.random.rand(1000) * 10 + 2),
        },
        xlab="성적 [점]",
        ylab="인원 [명]",
        title="Star Craft (hist, colors)",
        colors=["#800a0a", "#0a800a", "#0a0a80"],
    )
    assert path.split("/")[-1] in os.listdir()


def test_plot():
    path = zz.plot.plot(
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
    assert path.split("/")[-1] in os.listdir()


def test_plot_stacked():
    xdata = [i for i in range(20)]
    ydata = {
        "테란": list(np.random.rand(20) * 10),
        "저그": list(np.random.rand(20) * 10 + 1),
        "프로토스": list(np.random.rand(20) * 10 + 2),
    }
    ydata["Total"] = [
        sum(data) + 10 for data in zip(ydata["테란"], ydata["프로토스"], ydata["저그"])
    ]
    path = zz.plot.plot(
        xdata,
        ydata,
        xlab="시간 [초]",
        ylab="성적 [점]",
        stacked=True,
        title="Star Craft (plot, stacked)",
    )
    assert path.split("/")[-1] in os.listdir()


def test_plot_stacked_palette():
    xdata = [i for i in range(20)]
    ydata = {
        "테란": list(np.random.rand(20) * 10),
        "저그": list(np.random.rand(20) * 10 + 1),
        "프로토스": list(np.random.rand(20) * 10 + 2),
    }
    ydata["Total"] = [
        sum(data) + 10 for data in zip(ydata["테란"], ydata["프로토스"], ydata["저그"])
    ]
    path = zz.plot.plot(
        xdata,
        ydata,
        xlab="시간 [초]",
        ylab="성적 [점]",
        stacked=True,
        title="Star Craft (plot, stacked, palette)",
        colors="Set2",
    )
    assert path.split("/")[-1] in os.listdir()


def test_plot_stacked_colors():
    xdata = [i for i in range(20)]
    ydata = {
        "테란": list(np.random.rand(20) * 10),
        "저그": list(np.random.rand(20) * 10 + 1),
        "프로토스": list(np.random.rand(20) * 10 + 2),
    }
    ydata["Total"] = [
        sum(data) + 10 for data in zip(ydata["테란"], ydata["프로토스"], ydata["저그"])
    ]
    path = zz.plot.plot(
        xdata,
        ydata,
        xlab="시간 [초]",
        ylab="성적 [점]",
        stacked=True,
        title="Star Craft (plot, stacked, colors)",
        colors=["#800a0a", "#0a800a", "#0a0a80"],
    )
    assert path.split("/")[-1] in os.listdir()


def test_pie():
    path = zz.plot.pie(
        {"테란": 27, "저그": 40, "프로토스": 30}, dim="명", title="Star Craft (pie)"
    )
    assert path.split("/")[-1] in os.listdir()


def test_pie_palette():
    path = zz.plot.pie(
        {"테란": 27, "저그": 40, "프로토스": 30},
        dim="명",
        title="Star Craft (pie, palette)",
        colors="Set2",
    )
    assert path.split("/")[-1] in os.listdir()


def test_pie_colors():
    path = zz.plot.pie(
        {"테란": 27, "저그": 40, "프로토스": 30},
        dim="명",
        title="Star Craft (pie, colors)",
        colors=["#800a0a", "#0a800a", "#0a0a80"],
    )
    assert path.split("/")[-1] in os.listdir()


def test_scatter():
    path = zz.plot.scatter(
        {
            "테란": [list(np.random.rand(200) * 10), list(np.random.rand(200) * 10)],
            "저그": [
                list(np.random.rand(200) * 5 - 1),
                list(np.random.rand(200) * 5 + 1),
            ],
            "프로토스": [
                list(np.random.rand(200) * 10 + 3),
                list(np.random.rand(200) * 10 - 2),
            ],
        },
        xlab="비용 [미네랄]",
        ylab="전투력 [점]",
        title="Star Craft (scatter)",
        markersize=400,
    )
    assert path.split("/")[-1] in os.listdir()


def test_scatter_palette():
    path = zz.plot.scatter(
        {
            "테란": [list(np.random.rand(200) * 10), list(np.random.rand(200) * 10)],
        },
        xlab="비용 [미네랄]",
        ylab="전투력 [점]",
        title="Star Craft (scatter, palette)",
        colors="Set2",
        markersize=400,
    )
    assert path.split("/")[-1] in os.listdir()


def test_scatter_colors():
    path = zz.plot.scatter(
        {
            "테란": [list(np.random.rand(200) * 10), list(np.random.rand(200) * 10)],
            "저그": [
                list(np.random.rand(200) * 5 - 1),
                list(np.random.rand(200) * 5 + 1),
            ],
            "프로토스": [
                list(np.random.rand(200) * 10 + 3),
                list(np.random.rand(200) * 10 - 2),
            ],
        },
        xlab="비용 [미네랄]",
        ylab="전투력 [점]",
        title="Star Craft (scatter, colors)",
        colors=["#800a0a", "#0a800a", "#0a0a80"],
        markersize=400,
    )
    assert path.split("/")[-1] in os.listdir()


def test_table():
    data = [
        ["123", 123, 123.4],
        [123.4, "123", 123],
        [123, 123.4, "123"],
        ["123", 123, 123.4],
    ]
    col = ["테란", "저그", "프로토스"]
    row = ["test1", "test2", "test3", "test4"]
    path = zz.plot.table(data, col, row, title="Star Craft (table)", fontsize=50)
    assert path.split("/")[-1] in os.listdir()
