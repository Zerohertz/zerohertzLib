import os

import zerohertzLib as zz

tmp = os.path.dirname(__file__)
data = os.path.join(tmp, "data", "json")


def test_Json():
    js = zz.util.Json(data)
    assert type(js["title"]) == str
    key = js._getKey("language")
    assert key == "head/repo/language"
    assert js._getValue(key) == "Python"
    assert js.name.endswith(".json")
    assert type(js.keys) == list
    assert js.get("language") == js._getValue(key)
    js.tree()


def test_JsonDir():
    jsd = zz.util.JsonDir(data)
    assert len(jsd) == 5
    assert type(jsd[0]) == zz.util.Json
    assert type(jsd[0]["title"]) == str
    assert jsd._getKey("language") == "head/repo/language"
    assert type(jsd.unique("sha")) == set
    jsd.tree()


def test_write_csv():
    zz.util.write_csv(
        [
            ["id", "종족", "점수"],
            ["5hi9", "프로토스", 1248],
            ["gor2", "테란", 2309],
            ["gk03", "저그", 291],
        ],
        "star_craft",
    )
    assert "star_craft.csv" in os.listdir()


def test_write_tsv():
    zz.util.write_csv(
        [
            ["id", "종족", "점수"],
            ["5hi9", "프로토스", 1248],
            ["gor2", "테란", 2309],
            ["gk03", "저그", 291],
        ],
        "star_craft",
        True,
    )
    assert "star_craft.tsv" in os.listdir()


def test_write_json():
    zz.util.write_json(
        [{"id": "4169", "전투력": 4209, "정보": ["아무", "거나"]}] * 100,
        "star_craft",
    )
    assert "star_craft.json" in os.listdir()
