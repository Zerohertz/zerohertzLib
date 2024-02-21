import os
from datetime import datetime, timedelta

import FinanceDataReader as fdr

import zerohertzLib as zz

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
NOW = datetime.now()
START_DAY = NOW - timedelta(days=30 * 18)
START_DAY = START_DAY.strftime("%Y%m%d")
DATA = fdr.DataReader("066570", START_DAY)

zz.plot.font(kor=True)


def test_moving_average_backtest():
    signals = zz.quant.moving_average(DATA)
    zz.quant.backtest(DATA, signals)


def test_rsi_backtest():
    signals = zz.quant.rsi(DATA)
    zz.quant.backtest(DATA, signals)


def test_bollinger_bands_backtest():
    signals = zz.quant.bollinger_bands(DATA)
    zz.quant.backtest(DATA, signals)


def test_momentum_backtest():
    signals = zz.quant.momentum(DATA)
    zz.quant.backtest(DATA, signals)


def test_macd_backtest():
    signals = zz.quant.macd(DATA)
    zz.quant.backtest(DATA, signals)


def test_experiments():
    experiments = zz.quant.Experiments("Test", DATA)
    experiments.moving_average()
    experiments.rsi()
    experiments.bollinger_bands()
    experiments.momentum()
    experiments.macd()


def test_quant():
    qnt = zz.quant.Quant("Test", DATA, top=3, report=True)
    qnt_signals = qnt.signals.copy()
    results = zz.quant.backtest(
        DATA, qnt_signals, threshold=(qnt.threshold_sell, qnt.threshold_buy)
    )
    assert qnt.buy == results["buy"]
    assert qnt.sell == results["sell"]
    assert qnt.profit == results["profit"]
    assert qnt.transaction == results["transaction"]
    assert (qnt.signals == qnt_signals).all().all()
    qnt()


def test_quant_slack_bot_fdr_kor():
    qsb = zz.quant.QuantSlackBotFDR(
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
    qsb.index()


def test_quant_slack_bot_fdr_ovs():
    qsb = zz.quant.QuantSlackBotFDR(
        1,
        start_day=START_DAY,
        top=2,
        token=SLACK_BOT_TOKEN,
        channel="test",
        name="Stock Test",
        icon_emoji="rocket",
        kor=False,
    )
    qsb.buy()
