# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import json
from typing import Any

import requests
from slack_sdk import WebClient
from slack_sdk.web import SlackResponse

from zerohertzLib.api.base import AbstractBot, AbstractWebhook


class SlackWebhook(AbstractWebhook):
    """Slack Webhook의 data 전송을 위한 class

    Args:
        webhook_url: Slack Webhook의 URL
        channel: Slack Webhook이 전송할 channel
        name: Slack Webhook의 표시될 이름
        icon_emoji: Slack Webhook의 표시될 사진 (emoji)
        icon_url: Slack Webhook의 표시될 사진 (photo)
        timeout: `message`, `file` method 사용 시 사용될 timeout

    Examples:
        >>> slack = zz.api.SlackWebhook("https://hooks.slack.com/services/...")
        >>> slack = zz.api.SlackWebhook("https://hooks.slack.com/services/...", name="TEST", icon_emoji="ghost")

        ![Slack Webhook example](../../../assets/api/SlackWebhook.png){ width="300" }
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str | None = None,
        name: str | None = None,
        icon_emoji: str | None = None,
        icon_url: str | None = None,
        timeout: int = 10,
    ) -> None:
        self.webhook_url = webhook_url
        self.headers = {"Content-Type": "application/json"}
        self.data = {
            "channel": channel,
        }
        if name is not None:
            self.data["username"] = name
        if icon_emoji is not None:
            self.data["icon_emoji"] = f":{icon_emoji}:"
        if icon_url is not None:
            self.data["icon_url"] = icon_url
        self.timeout = timeout

    def message(self, message: str, codeblock: str | bool = False) -> requests.Response:
        """Slack Webhook을 통해 message 전송

        Args:
            message: 전송할 message
            codeblock: 전송되는 message의 스타일

        Returns:
            Slack Webhook의 응답

        Examples:
            >>> slack.message("test")
            <Response [200]>
        """
        if codeblock:
            message = f"``{message}``"
        self.data["text"] = message
        return requests.post(
            self.webhook_url,
            data=json.dumps(self.data),
            headers=self.headers,
            timeout=self.timeout,
        )

    def file(self, path: str) -> requests.Response:
        """Slack Webhook을 통해 file 전송

        Note:
            Slack Webhook은 File upload를 직접 지원하지 않음
            이 method는 NotImplementedError를 발생시킴
            File upload가 필요한 경우 SlackBot을 사용

        Args:
            path: 전송할 file 경로

        Raises:
            NotImplementedError: Slack Webhook은 File upload를 지원하지 않음
        """
        raise NotImplementedError(
            "Slack Webhook does not support file uploads. Use SlackBot instead."
        )


class SlackBot(AbstractBot[SlackResponse]):
    """Slack Bot의 data 전송을 위한 class

    ![Slack Bot scope setup](../../../assets/api/SlackBot.scope.png){ width="300" }

    Args:
        token: Slack Bot의 token
        channel: Slack Bot이 전송할 channel
        timeout: `message`, `file` method 사용 시 사용될 timeout
        name: Slack Bot에 표시될 이름
        icon_emoji: Slack Bot에 표시될 사진 (emoji)
        icon_url: Slack Bot에 표시될 사진 (photo)

    Examples:
        >>> slack = zz.api.SlackBot("xoxb-...", "test")
        >>> slack = zz.api.SlackBot("xoxb-...", "test", name="TEST")
        >>> slack = zz.api.SlackBot("xoxb-...", "test", icon_emoji="sparkles")
        >>> slack = zz.api.SlackBot("xoxb-...", "test", name="zerohertzLib", icon_url="https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284166558-0ba4b755-39cc-48ee-ba3b-5c02f54c4ca7.png")

        ![Slack Bot example](../../../assets/api/SlackBot.png){ width="300" }
    """

    def __init__(
        self,
        token: str,
        channel: str,
        timeout: int = 30,
        name: str | None = None,
        icon_emoji: str | None = None,
        icon_url: str | None = None,
    ) -> None:
        self.webclient = WebClient(token, timeout=timeout)
        self.channel2id = {}
        for channel_info in self.webclient.conversations_list(
            types="public_channel,private_channel"
        ).data["channels"]:
            channel_id, channel_name, is_channel = (
                channel_info.get("id"),
                channel_info.get("name"),
                channel_info.get("is_channel"),
            )
            if is_channel:
                self.channel2id[channel_name] = channel_id
        self.channel_name = channel
        self.channel_id = self.channel2id[channel]
        self.username = name
        self.icon_emoji = icon_emoji
        self.icon_url = icon_url

    def message(
        self,
        message: str,
        codeblock: str | bool = False,
        thread_id: str | None = None,
    ) -> SlackResponse:
        """Slack Bot을 통해 message 전송

        Args:
            message: 전송할 message
            codeblock: 전송되는 message의 스타일
            thread_id: 댓글을 전송할 thread의 timestamp

        Returns:
            Slack Bot의 응답

        Examples:
            >>> response = slack.message("test")
            >>> response
            <slack_sdk.web.slack_response.SlackResponse object at 0x7fb0c4346340>
            >>> slack.message("test", True, response.get("ts"))
            <slack_sdk.web.slack_response.SlackResponse object at 0x7fb0761b1100>
        """
        return self.webclient.chat_postMessage(
            channel=self.channel_id,
            text=self._get_codeblock(message=message, codeblock=codeblock),
            thread_ts=thread_id,
            icon_emoji=self.icon_emoji,
            icon_url=self.icon_url,
            username=self.username,
        )

    def get_thread_id(self, response: SlackResponse, **kwargs: Any) -> str:
        """Slack Bot 응답에서 thread ID 추출

        Args:
            response: Slack Bot의 응답
            **kwargs: 추가 매개변수 (사용되지 않음)

        Returns:
            Thread ID (timestamp)

        Examples:
            >>> response = slack.message("test")
            >>> thread_id = slack.get_thread_id(response)
            >>> slack.message("reply", thread_id=thread_id)
            <slack_sdk.web.slack_response.SlackResponse object at 0x...>
        """
        return response.get("ts", "")

    def file(self, path: str, thread_id: str | None = None) -> SlackResponse:
        """Slack Bot을 통해 file 전송

        Note:
            `name` 과 `icon_*` 의 적용 불가

        Args:
            path: 전송할 file 경로
            thread_id: 댓글을 전송할 thread의 timestamp

        Returns:
            Slack Bot의 응답

        Examples:
            >>> response = slack.file("test.jpg")
            >>> response
            <slack_sdk.web.slack_response.SlackResponse object at 0x7fb0675e0c10>
        """
        return self.webclient.files_upload_v2(
            file=path, channel=self.channel_id, thread_ts=thread_id
        )
