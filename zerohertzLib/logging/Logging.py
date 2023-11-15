import io
import logging
from typing import List, Optional

import requests

from zerohertzLib.api import Discord


class Logger:
    """이쁘게 Log를 찍어보는 Class ~!

    Note:
        더 예뻐지고 싶습니다...

    Args:
        logger_name (``Optional[str]``): Logger의 이름
        file_name(``Optional[str]``): ``.log`` 파일의 이름 (미입력 시 미출력)
        discord (``Optional[str]``): Discord webhook의 URL (``loggerLevel`` 적용)
        loggerLevel (``Optional[int]``): ``logging.getLogger`` 의 level
        consoleLevel (``Optional[int]``): ``logging.StreamHandler`` 의 level
        fileLevel (``Optional[int]``): ``logging.FileHandler`` 의 level

    Examples:
        >>> logger = zz.logging.Logger("TEST_1")
        >>> logger.debug("debug")
        2023-11-07 21:41:36,505 | DEBUG    | TEST_1 | debug
        >>> logger.info("info")
        2023-11-07 21:41:36,505 | INFO     | TEST_1 | info
        >>> logger.warning("warning")
        2023-11-07 21:41:36,505 | WARNING  | TEST_1 | warning
        >>> logger.error("error")
        2023-11-07 21:41:36,505 | ERROR    | TEST_1 | error
        >>> logger.critical("critical")
        2023-11-07 21:41:36,505 | CRITICAL | TEST_1 | critical
    """

    def __init__(
        self,
        logger_name: Optional[str] = None,
        file_name: Optional[str] = None,
        discord: Optional[str] = None,
        loggerLevel: Optional[int] = logging.DEBUG,
        consoleLevel: Optional[int] = logging.DEBUG,
        fileLevel: Optional[int] = logging.DEBUG,
    ):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(loggerLevel)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
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
        self.discord = discord
        if not self.discord is None:
            self.Discord = Discord(discord)
            self.log_stream = io.StringIO()
            stream_handler = logging.StreamHandler(self.log_stream)
            stream_handler.setLevel(loggerLevel)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def debug(self, log: str) -> None:
        self.logger.debug(log)
        if not self.discord is None:
            self._send_discord_message()

    def info(self, log: str) -> None:
        self.logger.info(log)
        if not self.discord is None:
            self._send_discord_message()

    def warning(self, log: str) -> None:
        self.logger.warning(log)
        if not self.discord is None:
            self._send_discord_message()

    def error(self, log: str) -> None:
        self.logger.error(log)
        if not self.discord is None:
            self._send_discord_message()

    def critical(self, log: str) -> None:
        self.logger.critical(log)
        if not self.discord is None:
            self._send_discord_message()

    def _send_discord_message(self) -> List[requests.models.Response]:
        # TODO: response에 대한 처리
        response = self.Discord.message(self.log_stream.getvalue(), codeblock=True)
        self.log_stream.seek(0)
        self.log_stream.truncate()
        return response
