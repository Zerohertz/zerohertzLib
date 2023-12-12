import os
from datetime import datetime, timedelta

import zerohertzLib as zz

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
NOW = datetime.now()
START_DAY = NOW - timedelta(days=30 * 6)
START_DAY = START_DAY.strftime("%Y%m%d")


def test_quant_slack_bot_fdr_kor():
    qnt = zz.quant.QuantSlackBotFDR(
        1,
        start_day=START_DAY,
        ohlc="Close",
        top=4,
        token=SLACK_BOT_TOKEN,
        channel="test",
        name="Stock Test",
        icon_emoji="rocket",
        analysis=True,
    )
    qnt.index()


def test_quant_slack_bot_fdr_ovs():
    qnt = zz.quant.QuantSlackBotFDR(
        1,
        start_day=START_DAY,
        top=2,
        token=SLACK_BOT_TOKEN,
        channel="test",
        name="Stock Test",
        icon_emoji="rocket",
        kor=False,
    )
    qnt.buy()
