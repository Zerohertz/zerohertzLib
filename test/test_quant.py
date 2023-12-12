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
        token=SLACK_BOT_TOKEN,
        channel="test",
        start_day=START_DAY,
        ohlc="Close",
        top=4,
        name="Stock Test",
        icon_emoji="rocket",
        analysis=False,
    )
    qnt.index()


def test_quant_slack_bot_fdr_ovs():
    qnt = zz.quant.QuantSlackBotFDR(
        1,
        token=SLACK_BOT_TOKEN,
        channel="test",
        start_day=START_DAY,
        top=2,
        name="Stock Test",
        icon_emoji="rocket",
        analysis=False,
        kor=False,
    )
    qnt.buy()
