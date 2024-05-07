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

import json
from typing import Optional

import requests
from slack_sdk import WebClient
from slack_sdk.web import SlackResponse


class SlackWebhook:
    """Slack Webhook의 data 전송을 위한 class

    Args:
        webhook_url (``str``): Slack Webhook의 URL
        channel (``Optional[str]``): Slack Webhook이 전송할 channel
        name (``Optional[str]``): Slack Webhook의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Webhook의 표시될 사진 (emoji)
        icon_url (``Optional[str]``): Slack Webhook의 표시될 사진 (photo)
        timeout (``Optional[int]``): ``message``, ``file`` method 사용 시 사용될 timeout

    Examples:
        >>> slack = zz.api.SlackWebhook("https://hooks.slack.com/services/...")
        >>> slack = zz.api.SlackWebhook("https://hooks.slack.com/services/...", name="TEST", icon_emoji="ghost")

        .. image:: _static/examples/static/api.SlackWebhook.png
            :align: center
            :width: 300px
    """

    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        name: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        icon_url: Optional[str] = None,
        timeout: Optional[int] = 10,
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

    def message(
        self,
        message: str,
        codeblock: Optional[bool] = False,
    ) -> requests.models.Response:
        """Slack Webhook을 통해 message 전송

        Args:
            message (``str``): 전송할 message
            codeblock (``Optional[bool]``): 전송되는 message의 스타일

        Returns:
            ``requests.models.Response``: Slack Webhook의 응답

        Examples:
            >>> slack.message("test")
            <Response [200]>
        """
        if message == "":
            return None
        if codeblock:
            message = f"```{message}```"
        self.data["text"] = message
        return requests.post(
            self.webhook_url,
            data=json.dumps(self.data),
            headers=self.headers,
            timeout=self.timeout,
        )


class SlackBot:
    """Slack Bot의 data 전송을 위한 class

    .. image:: _static/examples/static/api.SlackBot.scope.png
        :align: center
        :width: 300px

    Args:
        token (``str``): Slack Bot의 token
        channel (``str``): Slack Bot이 전송할 channel
        name (``Optional[str]``): Slack Bot의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Bot의 표시될 사진 (emoji)
        icon_url (``Optional[str]``): Slack Bot의 표시될 사진 (photo)
        timeout (``Optional[int]``): ``message``, ``file`` method 사용 시 사용될 timeout

    Examples:
        >>> slack = zz.api.SlackBot("xoxb-...", "test")
        >>> slack = zz.api.SlackBot("xoxb-...", "test", name="TEST")
        >>> slack = zz.api.SlackBot("xoxb-...", "test", icon_emoji="sparkles")
        >>> slack = zz.api.SlackBot("xoxb-...", "test", name="zerohertzLib", icon_url="https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284166558-0ba4b755-39cc-48ee-ba3b-5c02f54c4ca7.png")

        .. image:: _static/examples/static/api.SlackBot.png
            :align: center
            :width: 300px
    """

    def __init__(
        self,
        token: str,
        channel: str,
        name: Optional[str] = None,
        icon_emoji: Optional[str] = None,
        icon_url: Optional[str] = None,
        timeout: Optional[int] = 30,
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
        codeblock: Optional[bool] = False,
        thread_ts: Optional[str] = None,
    ) -> SlackResponse:
        """Slack Bot을 통해 message 전송

        Args:
            message (``str``): 전송할 message
            codeblock (``Optional[bool]``): 전송되는 message의 스타일
            thread_ts (``Optional[str]``): 댓글을 전송할 thread의 timestamp

        Returns:
            ``slack_sdk.web.slack_response.SlackResponse``: Slack Bot의 응답

        Examples:
            >>> response = slack.message("test")
            >>> response
            <slack_sdk.web.slack_response.SlackResponse object at 0x7fb0c4346340>
            >>> slack.message("test", True, response.get("ts"))
            <slack_sdk.web.slack_response.SlackResponse object at 0x7fb0761b1100>
        """
        if message == "":
            return None
        if codeblock:
            message = f"```{message}```"
        return self.webclient.chat_postMessage(
            channel=self.channel_id,
            text=message,
            thread_ts=thread_ts,
            icon_emoji=self.icon_emoji,
            icon_url=self.icon_url,
            username=self.username,
        )

    def file(self, path: str, thread_ts: Optional[str] = None) -> SlackResponse:
        """Slack Bot을 통해 file 전송

        Note:
            ``name`` 과 ``icon_*`` 의 적용 불가

        Args:
            path (``str``): 전송할 file 경로
            thread_ts (``Optional[str]``): 댓글을 전송할 thread의 timestamp

        Returns:
            ``slack_sdk.web.slack_response.SlackResponse``: Slack Bot의 응답

        Examples:
            >>> response = slack.file("test.jpg")
            >>> response
            <slack_sdk.web.slack_response.SlackResponse object at 0x7fb0675e0c10>
        """
        return self.webclient.files_upload_v2(
            file=path, channel=self.channel_id, thread_ts=thread_ts
        )
