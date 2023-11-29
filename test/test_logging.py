import os

import zerohertzLib as zz

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")


def test_logging():
    log = zz.logging.Logger("TEST_1")
    log.debug("debug")
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")


def test_logging_discord():
    log = zz.logging.Logger("TEST_2", logger_level=20, discord=DISCORD_WEBHOOK_URL)
    log.debug("debug")
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")


def test_logging_slack_webhook():
    log = zz.logging.Logger(
        "TEST_3", logger_level=20, slack=SLACK_WEBHOOK_URL, channel="test"
    )
    log.debug("debug")
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")


def test_logging_slack_bot():
    log = zz.logging.Logger(
        "TEST_4", logger_level=20, slack=SLACK_BOT_TOKEN, channel="test"
    )
    log.debug("debug")
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")
