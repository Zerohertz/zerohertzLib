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
