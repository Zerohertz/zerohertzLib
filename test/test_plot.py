import zerohertzLib as zz


def test_bar():
    zz.plot.bar(
        {"테란": 27, "저그": 40, "프로토스": 30}, xlab="종족", ylab="인구 [명]", title="Star Craft"
    )
