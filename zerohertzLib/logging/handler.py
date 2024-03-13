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
from logging import Handler
from typing import Optional

from zerohertzLib.api import Discord, SlackBot, SlackWebhook


class DiscordHandler(Handler, Discord):
    """Discord Webhook 기반 handler

    Args:
        webhook_url (``str``): Discord Webhook의 URL
        level (``Optional[int]``): Logger의 level

    Examples:
        >>> logger = logging.Logger()
        >>> logger.addHandler(zz.logging.DiscordHandler(webhook_url))
    """

    def __init__(self, webhook_url: str, level: Optional[int] = logging.NOTSET) -> None:
        Handler.__init__(self, level)
        Discord.__init__(self, webhook_url)

    def emit(self, record: logging.LogRecord) -> None:
        """`logging.Handler.emit <https://docs.python.org/ko/3/library/logging.html#logging.Handler.emit>`_ 구현"""
        self.message(self.format(record), codeblock=True)


class SlackBotHandler(Handler, SlackBot):
    """Slack Bot 기반 handler

    Args:
        token (``str``): Slack Bot의 token
        channel (``str``): Slack Bot이 전송할 channel
        name (``Optional[str]``): Slack Bot의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Bot의 표시될 사진 (emoji)
        icon_url (``Optional[str]``): Slack Bot의 표시될 사진 (photo)
        timeout (``Optional[int]``): ``message``, method 사용 시 사용될 timeout
        level (``Optional[int]``): Logger의 level

    Examples:
        >>> logger = logging.Logger()
        >>> logger.addHandler(zz.logging.SlackBotHandler(token, channel, name, icon_emoji))
    """

    def __init__(
        self,
        token: str,
        channel: str,
        name: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        icon_url: Optional[str] = None,
        timeout: Optional[int] = 10,
        level: Optional[int] = logging.NOTSET,
    ) -> None:
        Handler.__init__(self, level)
        SlackBot.__init__(self, token, channel, name, icon_emoji, icon_url, timeout)

    def emit(self, record: logging.LogRecord) -> None:
        """`logging.Handler.emit <https://docs.python.org/ko/3/library/logging.html#logging.Handler.emit>`_ 구현"""
        self.message(self.format(record), codeblock=True)


class SlackWebhookHandler(Handler, SlackWebhook):
    """Slack Webhook 기반 handler

    Args:
        webhook_url (``str``): Slack Webhook의 URL
        channel (``Optional[str]``): Slack Webhook이 전송할 channel
        name (``Optional[str]``): Slack Webhook의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Webhook의 표시될 사진 (emoji)
        icon_url (``Optional[str]``): Slack Webhook의 표시될 사진 (photo)
        timeout (``Optional[int]``): ``message`` method 사용 시 사용될 timeout
        level (``Optional[int]``): Logger의 level

    Examples:
        >>> logger = logging.Logger()
        >>> logger.addHandler(zz.logging.SlackWebhookHandler(webhook_url, channel, name, icon_emoji))
    """

    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        name: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        icon_url: Optional[str] = None,
        timeout: Optional[int] = 10,
        level: Optional[int] = logging.NOTSET,
    ) -> None:
        Handler.__init__(self, level)
        SlackWebhook.__init__(
            self, webhook_url, channel, name, icon_emoji, icon_url, timeout
        )

    def emit(self, record: logging.LogRecord) -> None:
        """`logging.Handler.emit <https://docs.python.org/ko/3/library/logging.html#logging.Handler.emit>`_ 구현"""
        self.message(self.format(record), codeblock=True)
