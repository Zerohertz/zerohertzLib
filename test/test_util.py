# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import os

import zerohertzLib as zz

tmp = os.path.dirname(__file__)
data = os.path.join(tmp, "data")


def test_Json():
    js = zz.util.Json(os.path.join(data, "json"))
    assert isinstance(js["title"], str)
    key = js._get_key("language")
    assert key == "head/repo/language"
    assert js._get_value(key) == "Python"
    assert js.name.endswith(".json")
    assert isinstance(js.keys, list)
    assert js.get("language") == js._get_value(key)
    js.tree()


def test_JsonDir():
    jsd = zz.util.JsonDir(os.path.join(data, "json"))
    assert len(jsd) == 5
    assert isinstance(jsd[0], zz.util.Json)
    assert isinstance(jsd[0]["title"], str)
    assert jsd._get_key("language") == "head/repo/language"
    assert isinstance(jsd.unique("sha"), set)
    jsd.tree()


def test_write_json():
    zz.util.write_json(
        [{"id": "4169", "전투력": 4209, "정보": ["아무", "거나"]}] * 100, "star_craft"
    )
    assert "star_craft.json" in os.listdir()


def test_csv():
    zz.util.write_csv(
        [
            ["id", "Races", "Scores"],
            ["5hi9", "Protoss", 1248],
            ["gor2", "Terran", 2309],
            ["gk03", "Zerg", 291],
        ],
        "star_craft",
    )
    assert "star_craft.csv" in os.listdir()
    data = zz.util.read_csv("star_craft.csv")
    assert list(data.keys()) == ["id", "Races", "Scores"]
    data = zz.util.read_csv("star_craft.csv", False)
    assert data[0][1] == "5hi9"
    assert data[2][3] == "291"


def test_tsv():
    zz.util.write_csv(
        [
            ["id", "Races", "Scores"],
            ["5hi9", "Protoss", 1248],
            ["gor2", "Terran", 2309],
            ["gk03", "Zerg", 291],
        ],
        "star_craft",
        True,
    )
    assert "star_craft.tsv" in os.listdir()
    data = zz.util.read_csv("star_craft.tsv")
    assert list(data.keys()) == ["id", "Races", "Scores"]
    data = zz.util.read_csv("star_craft.tsv", False)
    assert data[0][1] == "5hi9"
    assert data[2][3] == "291"


def test_MakeData():
    class MockData(zz.util.MakeData):
        def condition(self, json_instance):
            return True

    target_path = "mockdata"
    print(len(zz.util.JsonDir(f"{data}/annotation/mock/labels")))

    mockdata = MockData(
        f"{data}/annotation/mock/images",
        f"{data}/annotation/mock/labels",
        "name",
        target_path,
    )
    mockdata.make()
    assert "test.jpg" in os.listdir(os.path.join(target_path, "data"))
    assert "test.json" in os.listdir(os.path.join(target_path, "json"))


def test_find_ext():
    assert isinstance(zz.util.find_ext(), dict)


def test_sort_dict():
    target = {3: 6, 4: 2, 2: 7}
    solution = zz.util.sort_dict(target)
    answer = {2: 7, 3: 6, 4: 2}
    assert solution == answer
    assert list(solution.keys()) == list(answer.keys())

    target = {3: 6, 4: 2, 2: 7}
    order = [4, 2, 3]
    solution = zz.util.sort_dict(target, order)
    answer = {4: 2, 2: 7, 3: 6}
    assert solution == answer
    assert list(solution.keys()) == order

    target = {"C": 6, "D": 2, "A": 7}
    solution = zz.util.sort_dict(target)
    answer = {"A": 7, "C": 6, "D": 2}
    assert solution == answer
    assert list(solution.keys()) == list(answer.keys())

    target = {"C": 6, "D": 2, "A": 7}
    order = ["D", "C"]
    solution = zz.util.sort_dict(target, order)
    answer = {"D": 2, "C": 6}
    assert solution == answer
    assert list(solution.keys()) == order
