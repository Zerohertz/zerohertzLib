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

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from .handler import DiscordHandler, SlackBotHandler, SlackWebhookHandler


class Logger(logging.Logger):
    """이쁘게 log를 찍어보는 class

    Note:
        더 예뻐지고 싶습니다...

    Args:
        logger_name (``Optional[str]``): Logger의 이름
        width (``Optional[int]``): Logger의 너비
        show_path (``Optional[bool]``): Logger의 호출 경로 표시 여부
        file_name(``Optional[str]``): ``.log`` file의 이름 (미입력 시 미출력)
        discord (``Optional[str]``): Discord Webhook의 URL (``logger_level`` 적용)
        slack (``Optional[str]``): Slack Webhook의 URL 혹은 Bot의 token (``logger_level`` 적용)
        channel (``Optional[str]``): Slack Webhook 또는 Bot이 전송할 channel
        logger_level (``Optional[int]``): ``logging.Logger`` 의 level
        console_level (``Optional[int]``): ``rich.logging.RichHandler`` 의 level
        file_level (``Optional[int]``): ``logging.FileHandler`` 의 level

    Examples:
        >>> logger = zz.logging.Logger("TEST_1")
        >>> logger.debug("debug")
        [03/13/24 00:00:00] DEBUG    [TEST_1] debug                                  <stdin>:1
        >>> logger.info("info")
        [03/13/24 00:00:00] INFO     [TEST_1] info                                   <stdin>:1
        >>> logger.warning("warning")
        [03/13/24 00:00:00] WARNING  [TEST_1] warning                                <stdin>:1
        >>> logger.error("error")
        [03/13/24 00:00:00] ERROR    [TEST_1] error                                  <stdin>:1
        >>> logger.critical("critical")
        [03/13/24 00:00:00] CRITICAL [TEST_1] critical                               <stdin>:1
    """

    def __init__(
        self,
        logger_name: Optional[str] = None,
        width: Optional[int] = None,
        show_path: Optional[bool] = True,
        file_name: Optional[str] = None,
        discord: Optional[str] = None,
        slack: Optional[str] = None,
        channel: Optional[str] = None,
        logger_level: Optional[int] = logging.NOTSET,
        console_level: Optional[int] = logging.NOTSET,
        file_level: Optional[int] = logging.NOTSET,
    ) -> None:
        super().__init__(logger_name, logger_level)
        default_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        rich_formatter = logging.Formatter("[%(name)s] %(message)s")
        console_handler = RichHandler(
            console=Console(width=width),
            show_path=show_path,
            rich_tracebacks=True,
            keywords=[f"[{logger_name}]"],
        )
        console_handler.setLevel(console_level)
        console_handler.setFormatter(rich_formatter)
        self.addHandler(console_handler)
        if file_name is not None:
            file_handler = logging.FileHandler(f"{file_name}.log")
            file_handler.setLevel(file_level)
            file_handler.setFormatter(default_formatter)
            self.addHandler(file_handler)
        if discord is not None and slack is not None:
            raise ValueError("Slack and Discord cannot be used simultaneously")
        self.sender = None
        if slack is not None:
            if channel is None:
                raise ValueError("A 'channel' value is required to use Slack")
            if "hooks.slack.com" in slack:
                slack_handler = SlackWebhookHandler(
                    slack, channel, name=logger_name, icon_emoji="memo"
                )
            else:
                slack_handler = SlackBotHandler(
                    slack, channel, name=logger_name, icon_emoji="memo"
                )
            slack_handler.setLevel(console_level)
            slack_handler.setFormatter(default_formatter)
            self.addHandler(slack_handler)
        if discord is not None:
            discord_handler = DiscordHandler(discord)
            discord_handler.setLevel(console_level)
            discord_handler.setFormatter(default_formatter)
            self.addHandler(discord_handler)
