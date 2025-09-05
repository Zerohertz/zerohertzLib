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

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import requests

ResponseType = TypeVar("ResponseType")


def _get_codeblock(message: str, codeblock: str | bool = False) -> str:
    if not codeblock:
        return message
    if isinstance(codeblock, str):
        return f"```{codeblock}\n{message}\n```"
    return f"```{message}```"


class AbstractWebhook(ABC):
    @abstractmethod
    def message(
        self, message: str, codeblock: str | bool = False
    ) -> requests.Response: ...

    @abstractmethod
    def file(self, path: str) -> requests.Response: ...

    def _get_codeblock(self, message: str, codeblock: str | bool = False) -> str:
        return _get_codeblock(message=message, codeblock=codeblock)


class AbstractBot(ABC, Generic[ResponseType]):
    @abstractmethod
    def message(
        self,
        message: str,
        codeblock: str | bool = False,
        thread_id: str | None = None,
    ) -> ResponseType: ...

    @abstractmethod
    def get_thread_id(self, response: ResponseType, **kwargs) -> str: ...

    @abstractmethod
    def file(self, path: str, thread_id: str | None = None) -> ResponseType: ...

    def _get_codeblock(self, message: str, codeblock: str | bool = False) -> str:
        return _get_codeblock(message=message, codeblock=codeblock)


class MockedBot(AbstractBot[None]):
    def __init__(self) -> None: ...

    def message(
        self,
        message: str,
        codeblock: str | bool = False,
        thread_id: str | None = None,
    ) -> None:
        return None

    def get_thread_id(self, response: None, **kwargs) -> str:
        return ""

    def file(self, path: str, thread_id: str | None = None) -> None:
        return None
