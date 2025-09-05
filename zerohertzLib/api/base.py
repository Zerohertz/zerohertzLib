# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import requests

ResponseType = TypeVar("ResponseType")


def _get_codeblock(message: str, codeblock: str | bool = False) -> str:
    """Code block formatting을 위한 helper function

    Args:
        message (``str``): Formatting할 message
        codeblock (``str | bool``): Code block 언어 지정 또는 적용 여부

    Returns:
        ``str``: Formatting된 message
    """
    if not codeblock:
        return message
    if isinstance(codeblock, str):
        return f"```{codeblock}\n{message}\n```"
    return f"```{message}```"


class AbstractWebhook(ABC):
    """Webhook 기반 messaging을 위한 abstract base class"""

    @abstractmethod
    def message(self, message: str, codeblock: str | bool = False) -> requests.Response:
        """Webhook을 통해 message 전송

        Args:
            message (``str``): 전송할 message
            codeblock (``str | bool``): 전송되는 message의 style

        Returns:
            ``requests.Response``: Webhook의 응답
        """

    @abstractmethod
    def file(self, path: str) -> requests.Response:
        """Webhook을 통해 file 전송

        Args:
            path (``str``): 전송할 file 경로

        Returns:
            ``requests.Response``: Webhook의 응답
        """

    def _get_codeblock(self, message: str, codeblock: str | bool = False) -> str:
        """Code block formatting을 위한 helper method

        Args:
            message (``str``): Formatting할 message
            codeblock (``str | bool``): Code block 언어 지정 또는 적용 여부

        Returns:
            ``str``: Formatting된 message
        """
        return _get_codeblock(message=message, codeblock=codeblock)


class AbstractBot(ABC, Generic[ResponseType]):
    """Bot 기반 messaging을 위한 abstract base class

    Args:
        ResponseType: Bot API 응답의 type
    """

    @abstractmethod
    def message(
        self,
        message: str,
        codeblock: str | bool = False,
        thread_id: str | None = None,
    ) -> ResponseType:
        """Bot을 통해 message 전송

        Args:
            message (``str``): 전송할 message
            codeblock (``str | bool``): 전송되는 message의 style
            thread_id (``str | None``): Thread ID (댓글 전송용)

        Returns:
            ``ResponseType``: Bot의 응답
        """

    @abstractmethod
    def get_thread_id(self, response: ResponseType, **kwargs) -> str:
        """Bot 응답에서 thread ID 추출

        Args:
            response (``ResponseType``): Bot의 응답
            **kwargs: 추가 매개변수

        Returns:
            ``str``: Thread ID
        """

    @abstractmethod
    def file(self, path: str, thread_id: str | None = None) -> ResponseType:
        """Bot을 통해 file 전송

        Args:
            path (``str``): 전송할 file 경로
            thread_id (``str | None``): Thread ID (댓글 전송용)

        Returns:
            ``ResponseType``: Bot의 응답
        """

    def _get_codeblock(self, message: str, codeblock: str | bool = False) -> str:
        """Code block formatting을 위한 helper method

        Args:
            message (``str``): Formatting할 message
            codeblock (``str | bool``): Code block 언어 지정 또는 적용 여부

        Returns:
            ``str``: Formatting된 message
        """
        return _get_codeblock(message=message, codeblock=codeblock)


class MockedBot(AbstractBot[None]):
    """Testing을 위한 mock bot implementation

    실제 API 호출 없이 bot interface를 테스트할 때 사용
    """

    def __init__(self) -> None:
        """MockedBot 초기화"""

    def message(
        self,
        message: str,
        codeblock: str | bool = False,
        thread_id: str | None = None,
    ) -> None:
        """Mock message 전송 (실제로는 아무것도 하지 않음)

        Args:
            message (``str``): 전송할 message
            codeblock (``str | bool``): 전송되는 message의 style
            thread_id (``str | None``): Thread ID (댓글 전송용)

        Returns:
            ``None``: 항상 None 반환
        """
        return None

    def get_thread_id(self, response: None, **kwargs) -> str:
        """Mock thread ID 반환 (빈 문자열)

        Args:
            response (``None``): Mock 응답 (None)
            **kwargs: 추가 매개변수

        Returns:
            ``str``: 빈 문자열
        """
        return ""

    def file(self, path: str, thread_id: str | None = None) -> None:
        """Mock file 전송 (실제로는 아무것도 하지 않음)

        Args:
            path (``str``): 전송할 file 경로
            thread_id (``str | None``): Thread ID (댓글 전송용)

        Returns:
            ``None``: 항상 None 반환
        """
        return None
