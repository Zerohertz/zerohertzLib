"""
MIT License

Copyright (c) 2023 Hyogeun Oh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import io
import logging
from typing import List, Optional

import requests

from zerohertzLib.api import Discord, SlackBot, SlackWebhook


class Logger:
    """이쁘게 log를 찍어보는 class

    Note:
        더 예뻐지고 싶습니다...

    Args:
        logger_name (``Optional[str]``): Logger의 이름
        file_name(``Optional[str]``): ``.log`` file의 이름 (미입력 시 미출력)
        discord (``Optional[str]``): Discord Webhook의 URL (``logger_level`` 적용)
        slack (``Optional[str]``): Slack Webhook의 URL 혹은 Bot의 token (``logger_level`` 적용)
        channel (``Optional[str]``): Slack Webhook 또는 Bot이 전송할 channel
        logger_level (``Optional[int]``): ``logging.getLogger`` 의 level
        console_level (``Optional[int]``): ``logging.StreamHandler`` 의 level
        file_level (``Optional[int]``): ``logging.FileHandler`` 의 level

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
        slack: Optional[str] = None,
        channel: Optional[str] = None,
        logger_level: Optional[int] = logging.DEBUG,
        console_level: Optional[int] = logging.DEBUG,
        file_level: Optional[int] = logging.DEBUG,
    ) -> None:
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        if file_name is not None:
            file_handler = logging.FileHandler(f"{file_name}.log")
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        if discord is not None and slack is not None:
            raise ValueError("Slack and Discord cannot be used simultaneously")
        self.sender = None
        if slack is not None:
            if channel is None:
                raise ValueError("A 'channel' value is required to use Slack")
            if "hooks.slack.com" in slack:
                self.sender = SlackWebhook(
                    slack, channel, name="Logger", icon_emoji="memo"
                )
            else:
                self.sender = SlackBot(slack, channel, name="Logger", icon_emoji="memo")
        if discord is not None:
            self.sender = Discord(discord)
        if slack is not None or discord is not None:
            self.log_stream = io.StringIO()
            stream_handler = logging.StreamHandler(self.log_stream)
            stream_handler.setLevel(logger_level)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def debug(self, log: str) -> None:
        self.logger.debug(log)
        if self.sender is not None:
            self._send()

    def info(self, log: str) -> None:
        self.logger.info(log)
        if self.sender is not None:
            self._send()

    def warning(self, log: str) -> None:
        self.logger.warning(log)
        if self.sender is not None:
            self._send()

    def error(self, log: str) -> None:
        self.logger.error(log)
        if self.sender is not None:
            self._send()

    def critical(self, log: str) -> None:
        self.logger.critical(log)
        if self.sender is not None:
            self._send()

    def _send(self) -> List[requests.models.Response]:
        response = self.sender.message(self.log_stream.getvalue(), codeblock=True)
        self.log_stream.seek(0)
        self.log_stream.truncate()
        return response
