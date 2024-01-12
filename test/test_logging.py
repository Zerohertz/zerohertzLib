import os

import zerohertzLib as zz

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")


def test_logging():
    logger = zz.logging.Logger("TEST_1")
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")


def test_logging_discord():
    logger = zz.logging.Logger("TEST_2", logger_level=20, discord=DISCORD_WEBHOOK_URL)
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")


def test_logging_slack_webhook():
    logger = zz.logging.Logger(
        "TEST_3", logger_level=20, slack=SLACK_WEBHOOK_URL, channel="test"
    )
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")


def test_logging_slack_bot():
    logger = zz.logging.Logger(
        "TEST_4", logger_level=20, slack=SLACK_BOT_TOKEN, channel="test"
    )
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")
