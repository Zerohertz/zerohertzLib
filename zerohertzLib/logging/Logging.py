import logging
from typing import Optional


class Logger:
    """이쁘게 Log를 찍어보는 Class ~!

    Note:
        더 예뻐지고 싶습니다...

    Args:
        logger_name (``Optional[str]``): Logger의 이름
        file_name(``Optional[str]``): ``.log`` 파일의 이름 (미입력 시 미출력)
        loggerLevel (``Optional[int]``): ``logging.getLogger`` 의 level
        consoleLevel (``Optional[int]``): ``logging.StreamHandler`` 의 level
        fileLevel (``Optional[int]``): ``logging.FileHandler`` 의 level

    Examples:
        >>> logger = zz.logging.Logger("TEST_1")
        >>> logger.debug("debug")
        2023-11-07 21:41:36,505 | TEST_1 | DEBUG    | debug
        >>> logger.info("info")
        2023-11-07 21:41:36,505 | TEST_1 | INFO     | info
        >>> logger.warning("warning")
        2023-11-07 21:41:36,505 | TEST_1 | WARNING  | warning
        >>> logger.error("error")
        2023-11-07 21:41:36,505 | TEST_1 | ERROR    | error
        >>> logger.critical("critical")
        2023-11-07 21:41:36,505 | TEST_1 | CRITICAL | critical
    """

    def __init__(
        self,
        logger_name: Optional[str] = None,
        file_name: Optional[str] = None,
        loggerLevel: Optional[int] = logging.DEBUG,
        consoleLevel: Optional[int] = logging.DEBUG,
        fileLevel: Optional[int] = logging.DEBUG,
    ):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(loggerLevel)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(consoleLevel)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        if not file_name is None:
            file_handler = logging.FileHandler(f"{file_name}.log")
            file_handler.setLevel(fileLevel)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, log: str) -> None:
        self.logger.debug(log)

    def info(self, log: str) -> None:
        self.logger.info(log)

    def warning(self, log: str) -> None:
        self.logger.warning(log)

    def error(self, log: str) -> None:
        self.logger.error(log)

    def critical(self, log: str) -> None:
        self.logger.critical(log)
