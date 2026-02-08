# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import os
from datetime import datetime, timedelta

import FinanceDataReader as fdr

import zerohertzLib as zz

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
DISCORD_BOT_CHANNEL = os.environ.get("DISCORD_BOT_CHANNEL")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_BOT_CHANNEL = "test"

NOW = datetime.now()
QUANT_START_DAY = (NOW - timedelta(days=30 * 3)).strftime("%Y%m%d")
QUANT_SYMBOL_STR = [
    "360750",  # TIGER 미국S&P500
    "RANDOM_SYMBOL_FOR_TEST",
    "379810",  # KODEX 미국나스닥100
    "381170",  # TIGER 미국테크TOP10 INDXX
    "0047A0",  # TIGER 차이나테크TOP10
    "488770",  # KODEX 머니마켓액티브
    "411060",  # ACE KRX금현물
]
QUANT_SYMBOL_INT = 8
QUANT_SYMBOL_MIN = 2
QUANT_TEST_DATA = fdr.DataReader(QUANT_SYMBOL_STR[0], QUANT_START_DAY)

zz.plot.font(kor=True)


def test_moving_average_backtest():
    signals = zz.quant.moving_average(QUANT_TEST_DATA)
    zz.quant.backtest(QUANT_TEST_DATA, signals)


def test_rsi_backtest():
    signals = zz.quant.rsi(QUANT_TEST_DATA)
    zz.quant.backtest(QUANT_TEST_DATA, signals)


def test_bollinger_bands_backtest():
    signals = zz.quant.bollinger_bands(QUANT_TEST_DATA)
    zz.quant.backtest(QUANT_TEST_DATA, signals)


def test_momentum_backtest():
    signals = zz.quant.momentum(QUANT_TEST_DATA)
    zz.quant.backtest(QUANT_TEST_DATA, signals)


def test_macd_backtest():
    signals = zz.quant.macd(QUANT_TEST_DATA)
    zz.quant.backtest(QUANT_TEST_DATA, signals)


def test_experiments():
    experiments = zz.quant.Experiments("Test", QUANT_TEST_DATA)
    experiments.moving_average()
    experiments.rsi()
    experiments.bollinger_bands()
    experiments.momentum()
    experiments.macd()


def test_quant():
    qnt = zz.quant.Quant("Test", QUANT_TEST_DATA, top=3, report=True)
    qnt_signals = qnt.signals.copy()
    results = zz.quant.backtest(
        QUANT_TEST_DATA, qnt_signals, threshold=(qnt.threshold_sell, qnt.threshold_buy)
    )
    assert qnt.buy == results["buy"]
    assert qnt.sell == results["sell"]
    assert qnt.profit == results["profit"]
    assert qnt.transaction == results["transaction"]
    assert (qnt.signals == qnt_signals).all().all()
    qnt()


def test_quant_mocked_bot_fdr_kor():
    qsb = zz.quant.QuantBotFDR(
        QUANT_SYMBOL_STR,
        start_day=QUANT_START_DAY,
        ohlc="Close",
        top=1,
        mp_num=4,
        analysis=True,
    )
    qsb.index()


def test_quant_mocked_bot_fdr_ovs():
    qsb = zz.quant.QuantBotFDR(
        QUANT_SYMBOL_INT,
        start_day=QUANT_START_DAY,
        top=1,
        mp_num=4,
        kor=False,
    )
    qsb.buy()


def test_quant_discord_bot_fdr_kor():
    qsb = zz.quant.QuantBotFDR(
        QUANT_SYMBOL_STR[:QUANT_SYMBOL_MIN],
        start_day=QUANT_START_DAY,
        ohlc="Close",
        top=4,
        token=DISCORD_BOT_TOKEN,
        channel=DISCORD_BOT_CHANNEL,
        mp_num=2,
        analysis=True,
    )
    qsb.index()


def test_quant_discord_bot_fdr_ovs():
    qsb = zz.quant.QuantBotFDR(
        QUANT_SYMBOL_MIN,
        start_day=QUANT_START_DAY,
        top=4,
        token=DISCORD_BOT_TOKEN,
        channel=DISCORD_BOT_CHANNEL,
        kor=False,
    )
    qsb.buy()


def test_quant_slack_bot_fdr_kor():
    qsb = zz.quant.QuantBotFDR(
        QUANT_SYMBOL_STR[:QUANT_SYMBOL_MIN],
        start_day=QUANT_START_DAY,
        ohlc="Close",
        top=2,
        token=SLACK_BOT_TOKEN,
        channel=SLACK_BOT_CHANNEL,
        name="Stock Test",
        icon_emoji="rocket",
        analysis=True,
    )
    qsb.index()


def test_quant_slack_bot_fdr_ovs():
    qsb = zz.quant.QuantBotFDR(
        QUANT_SYMBOL_MIN,
        start_day=QUANT_START_DAY,
        top=2,
        token=SLACK_BOT_TOKEN,
        channel=SLACK_BOT_CHANNEL,
        name="Stock Test",
        icon_emoji="rocket",
        kor=False,
    )
    qsb.buy()


def test_cash2str():
    from zerohertzLib.quant.util import _cash2str

    assert _cash2str(1000, kor=True) == "₩1,000"
    assert _cash2str(1234567, kor=True) == "₩1,234,567"
    assert _cash2str(0, kor=True) == "₩0"
    assert _cash2str(-500, kor=True) == "-₩500"
    assert _cash2str(-1234567, kor=True) == "-₩1,234,567"
    assert _cash2str(1000.50, kor=True) == "₩1,000"
    assert _cash2str(1000, kor=False) == "$1,000.00"
    assert _cash2str(1234567.89, kor=False) == "$1,234,567.89"
    assert _cash2str(0, kor=False) == "$0.00"
    assert _cash2str(-500, kor=False) == "-$500.00"
    assert _cash2str(-1234567.89, kor=False) == "-$1,234,567.89"
    assert _cash2str(0.5, kor=False) == "$0.50"


def test_seconds_to_hms():
    from zerohertzLib.quant.util import _seconds_to_hms

    assert _seconds_to_hms(0) == "0.00s"
    assert _seconds_to_hms(30) == "30.00s"
    assert _seconds_to_hms(59.5) == "59.50s"
    assert _seconds_to_hms(60) == "1m 0.00s"
    assert _seconds_to_hms(90) == "1m 30.00s"
    assert _seconds_to_hms(3599) == "59m 59.00s"
    assert _seconds_to_hms(3600) == "1h 0m 0.00s"
    assert _seconds_to_hms(3661) == "1h 1m 1.00s"
    assert _seconds_to_hms(7384) == "2h 3m 4.00s"
    assert _seconds_to_hms(30, sign=0) == "30s"
    assert _seconds_to_hms(90, sign=4) == "1m 30.0000s"
    assert _seconds_to_hms(3661, sign=1) == "1h 1m 1.0s"
