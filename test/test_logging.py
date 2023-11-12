import zerohertzLib as zz


def test_logging():
    log = zz.logging.Logger("TEST_1")
    log.debug("debug")
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")


def test_logging_discord():
    log = zz.logging.Logger(
        "TEST_3",
        loggerLevel=20,
        discord="https://discord.com/api/webhooks/1170962638583373904/xVJKW1KkNo7Pc1HykJ85cHs_4SvRkKCbOvbf1qe1j8QXOnJyTGyJy8T7sI7kvfA8SGb-",
    )
    log.debug("debug")
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")
