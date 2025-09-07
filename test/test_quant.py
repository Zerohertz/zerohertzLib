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
QUANT_START_DAY = (NOW - timedelta(days=30 * 2)).strftime("%Y%m%d")
QUANT_SYMBOL_STR = "379800"
QUANT_SYMBOL_INT = 2
QUANT_TEST_DATA = fdr.DataReader(QUANT_SYMBOL_STR, QUANT_START_DAY)

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
        [QUANT_SYMBOL_STR],
        start_day=QUANT_START_DAY,
        ohlc="Close",
        top=1,
        analysis=True,
    )
    qsb.index()


def test_quant_mocked_bot_fdr_ovs():
    qsb = zz.quant.QuantBotFDR(
        QUANT_SYMBOL_INT,
        start_day=QUANT_START_DAY,
        top=1,
        kor=False,
    )
    qsb.buy()


def test_quant_discord_bot_fdr_kor():
    qsb = zz.quant.QuantBotFDR(
        QUANT_SYMBOL_INT,
        start_day=QUANT_START_DAY,
        ohlc="Close",
        top=4,
        token=DISCORD_BOT_TOKEN,
        channel=DISCORD_BOT_CHANNEL,
        analysis=True,
    )
    qsb.index()


def test_quant_discord_bot_fdr_ovs():
    qsb = zz.quant.QuantBotFDR(
        QUANT_SYMBOL_INT,
        start_day=QUANT_START_DAY,
        top=4,
        token=DISCORD_BOT_TOKEN,
        channel=DISCORD_BOT_CHANNEL,
        kor=False,
    )
    qsb.buy()


def test_quant_slack_bot_fdr_kor():
    qsb = zz.quant.QuantBotFDR(
        [QUANT_SYMBOL_STR],
        start_day=QUANT_START_DAY,
        ohlc="Close",
        top=4,
        token=SLACK_BOT_TOKEN,
        channel=SLACK_BOT_CHANNEL,
        name="Stock Test",
        icon_emoji="rocket",
        analysis=True,
    )
    qsb.index()


def test_quant_slack_bot_fdr_ovs():
    qsb = zz.quant.QuantBotFDR(
        QUANT_SYMBOL_INT,
        start_day=QUANT_START_DAY,
        top=4,
        token=SLACK_BOT_TOKEN,
        channel=SLACK_BOT_CHANNEL,
        name="Stock Test",
        icon_emoji="rocket",
        kor=False,
    )
    qsb.buy()
