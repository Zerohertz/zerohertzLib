import zerohertzLib as zz

WEBHOOK = "https://discord.com/api/webhooks/1174193014923591791/vBPMpb0otKQH0lflp169u0a-8gJPZyDg17SPEsxKDDlmv3PMFl4eNrt3KWQgUmnWpYJ9"


def test_logging():
    log = zz.logging.Logger("TEST_1")
    log.debug("debug")
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")


def test_logging_discord():
    log = zz.logging.Logger(
        "TEST_2",
        logger_level=20,
        discord=WEBHOOK,
    )
    log.debug("debug")
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")
