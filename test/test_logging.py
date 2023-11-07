import zerohertzLib as zz


def test_logging():
    log = zz.logging.Logger("TEST_1")
    log.debug("debug")
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")
