# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import json
from typing import Any

import requests

from zerohertzLib.api.base import AbstractBot, AbstractWebhook


class DiscordWebhook(AbstractWebhook):
    """Discord Webhook의 data 전송을 위한 class

    Args:
        webhook_url: Discord Webhook의 URL

    Examples:
        >>> discord = zz.api.DiscordWebhook("https://discord.com/api/webhooks/...")
    """

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def message(self, message: str, codeblock: str | bool = False) -> requests.Response:
        """Discord Webhook을 통해 message 전송

        Args:
            message: 전송할 message
            codeblock: 전송되는 message의 스타일

        Returns:
            Discord Webhook의 응답

        Examples:
            >>> discord = zz.api.DiscordWebhook("https://discord.com/api/webhooks/...")
            >>> discord.message("Testing...")
            <Response [204]>
        """
        headers = {"Content-Type": "application/json"}
        message = self._get_codeblock(message, codeblock)
        data = {"content": message}
        return requests.post(
            self.webhook_url, data=json.dumps(data), headers=headers, timeout=10
        )

    def file(self, path: str) -> requests.Response:
        """Discord Webhook을 통해 file 전송

        Args:
            path: 전송할 file 경로

        Returns:
            Discord Webhook의 응답

        Examples:
            >>> discord = zz.api.DiscordWebhook("https://discord.com/api/webhooks/...")
            >>> discord.file("test.jpg")
            <Response [200]>
        """
        with open(path, "rb") as file:
            files = {
                "file": (path, file),
            }
            response = requests.post(self.webhook_url, files=files, timeout=10)
        return response


class DiscordBot(AbstractBot[requests.Response]):
    """Discord Bot의 data 전송을 위한 class

    Args:
        token: Discord Bot 토큰
        channel: Discord Bot이 전송할 channel
        timeout: API 요청 시 사용될 timeout

    Examples:
        >>> discord = zz.api.DiscordBot("YOUR_BOT_TOKEN", "1234567890")
        >>> discord = zz.api.DiscordBot("YOUR_BOT_TOKEN", "1234567890", timeout=30)
    """

    def __init__(self, token: str, channel: str, timeout: int = 30) -> None:
        self.api_base = "https://discord.com/api/v10"
        self.headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
        }
        self.channel = channel
        self.timeout = timeout

    def message(
        self,
        message: str,
        codeblock: str | bool = False,
        thread_id: str | None = None,
    ) -> requests.Response:
        """Discord Bot을 통해 message 전송

        Args:
            message: 전송할 message
            codeblock: 전송되는 message의 스타일
            thread_id: 댓글을 전송할 thread의 ID

        Returns:
            Discord Bot의 응답

        Examples:
            >>> response = discord.message("test")
            >>> response
            <Response [200]>
            >>> response = discord.message('print("hi")', codeblock="python")
            >>> response
            <Response [200]>
        """
        channel_id = thread_id if thread_id else self.channel
        return requests.post(
            f"{self.api_base}/channels/{channel_id}/messages",
            data=json.dumps(
                {"content": self._get_codeblock(message=message, codeblock=codeblock)}
            ),
            headers=self.headers,
            timeout=self.timeout,
        )

    def get_thread_id(self, response: requests.Response, **kwargs: Any) -> str:
        """Discord Bot 응답에서 thread를 생성하고 thread ID 반환

        Args:
            response: Thread를 생성할 메시지 response
            **kwargs: 추가 매개변수 (name 포함 가능)

        Returns:
            생성된 thread ID

        Examples:
            >>> response = discord.message("test")
            >>> thread_id = discord.get_thread_id(response, name="Discussion")
            >>> discord.message("reply", thread_id=thread_id)
            <Response [200]>
        """
        payload = {
            "name": kwargs.get("name", "New thread")[:100],
            "type": 11,
        }
        message_id = response.json()["id"]
        return requests.post(
            f"{self.api_base}/channels/{self.channel}/messages/{message_id}/threads",
            data=json.dumps(payload),
            headers=self.headers,
            timeout=self.timeout,
        ).json()["id"]

    def file(self, path: str, thread_id: str | None = None) -> requests.Response:
        """Discord Bot을 통해 file 전송

        Args:
            path: 전송할 file 경로
            thread_id: 댓글을 전송할 thread의 ID

        Returns:
            Discord Bot의 응답

        Examples:
            >>> response = discord.file("test.jpg")
            >>> response
            <Response [200]>
        """
        channel_id = thread_id if thread_id else self.channel
        with open(path, "rb") as file:
            files = {"file": (path, file)}
            headers = {"Authorization": self.headers["Authorization"]}
            response = requests.post(
                f"{self.api_base}/channels/{channel_id}/messages",
                files=files,
                headers=headers,
                timeout=self.timeout,
            )
        return response
