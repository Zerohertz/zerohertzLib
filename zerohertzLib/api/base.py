# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import requests

ResponseType = TypeVar("ResponseType")


def _get_codeblock(message: str, codeblock: str | bool = False) -> str:
    """Code block formatting을 위한 helper function

    Args:
        message: Formatting할 message
        codeblock: Code block 언어 지정 또는 적용 여부

    Returns:
        Formatting된 message
    """
    if not codeblock:
        return message
    if isinstance(codeblock, str):
        return f"```{codeblock}\n{message}\n```"
    return f"```{message}```"


class AbstractWebhook:
    """Webhook 기반 messaging을 위한 abstract base class"""

    @abstractmethod
    def message(self, message: str, codeblock: str | bool = False) -> requests.Response:
        """Webhook을 통해 message 전송

        Args:
            message: 전송할 message
            codeblock: 전송되는 message의 style

        Returns:
            Webhook의 응답
        """

    @abstractmethod
    def file(self, path: str) -> requests.Response:
        """Webhook을 통해 file 전송

        Args:
            path: 전송할 file 경로

        Returns:
            Webhook의 응답
        """

    def _get_codeblock(self, message: str, codeblock: str | bool = False) -> str:
        """Code block formatting을 위한 helper method

        Args:
            message: Formatting할 message
            codeblock: Code block 언어 지정 또는 적용 여부

        Returns:
            Formatting된 message
        """
        return _get_codeblock(message=message, codeblock=codeblock)


class AbstractBot(Generic[ResponseType]):
    """Bot 기반 messaging을 위한 abstract base class"""

    @abstractmethod
    def message(
        self,
        message: str,
        codeblock: str | bool = False,
        thread_id: str | None = None,
    ) -> ResponseType:
        """Bot을 통해 message 전송

        Args:
            message: 전송할 message
            codeblock: 전송되는 message의 style
            thread_id: Thread ID (댓글 전송용)

        Returns:
            Bot의 응답
        """

    @abstractmethod
    def get_thread_id(self, response: ResponseType, **kwargs: Any) -> str:
        """Bot 응답에서 thread ID 추출

        Args:
            response: Bot의 응답
            **kwargs: 추가 매개변수

        Returns:
            Thread ID
        """

    @abstractmethod
    def file(self, path: str, thread_id: str | None = None) -> ResponseType:
        """Bot을 통해 file 전송

        Args:
            path: 전송할 file 경로
            thread_id: Thread ID (댓글 전송용)

        Returns:
            Bot의 응답
        """

    def _get_codeblock(self, message: str, codeblock: str | bool = False) -> str:
        """Code block formatting을 위한 helper method

        Args:
            message: Formatting할 message
            codeblock: Code block 언어 지정 또는 적용 여부

        Returns:
            Formatting된 message
        """
        return _get_codeblock(message=message, codeblock=codeblock)


class MockedBot:
    """Testing을 위한 mocked bot implementation

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
        return None

    def get_thread_id(self, response: None, **kwargs: Any) -> str:
        return ""

    def file(self, path: str, thread_id: str | None = None) -> None:
        return None
