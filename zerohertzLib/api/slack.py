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


class SlackWebhook:
    """Slack Webhook의 데이터 전송을 위한 class

    Args:
        webhook_url (``str``): Slack Webhook의 URL

    Examples:
        >>> slack = zz.api.SlackWebhook("https://hooks.slack.com/services/...")
        >>> slack = zz.api.SlackWebhook("https://hooks.slack.com/services/...", name="TEST", icon_emoji="ghost")

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/286251394-358e5662-06af-454b-8b1b-8b05e320bf5a.png
            :alt: Slack Webhook
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
            timeout=10,
        )


class SlackBot:
    """Slack Bot의 데이터 전송을 위한 class

    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/286245553-83bc9a7f-a840-4983-bc54-8e27fc9d01b1.png
        :alt: Slack Bot Scopes
        :align: center
        :width: 300px

    Args:
        token (``str``): Slack Bot의 token
        channel (``channel``): Slack Bot이 전송할 channel
        name (``Optional[str]``): Slack Bot의 표시될 이름
        icon_emoji (``Optional[str]``): Slack Bot의 표시될 사진 (emoji)
        icon_url (``Optional[str]``): Slack Bot의 표시될 사진 (photo)

    Examples:
        >>> slack = zz.api.SlackBot("xoxb-...", "test")
        >>> slack = zz.api.SlackBot("xoxb-...", "test", name="TEST")
        >>> slack = zz.api.SlackBot("xoxb-...", "test", icon_emoji="sparkles")
        >>> slack = zz.api.SlackBot("xoxb-...", "test", name="zerohertzLib", icon_url="https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/284166558-0ba4b755-39cc-48ee-ba3b-5c02f54c4ca7.png")

        .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/42334717/286248478-10f5e8f4-a48c-4bf3-b742-da5818f30fee.png
            :alt: Slack Bot Scopes
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
    ) -> None:
        self.token = token
        self.headers = {
            "Authorization": "Bearer " + token,
        }
        self.channel = channel
        self.data = {
            "channel": channel,
        }
        if name is not None:
            self.data["username"] = name
        if icon_emoji is not None:
            self.data["icon_emoji"] = f":{icon_emoji}:"
        if icon_url is not None:
            self.data["icon_url"] = icon_url

    def message(
        self,
        message: str,
        codeblock: Optional[bool] = False,
    ) -> requests.models.Response:
        """Slack Bot을 통해 message 전송

        Args:
            message (``str``): 전송할 message
            codeblock (``Optional[bool]``): 전송되는 message의 스타일

        Returns:
            ``requests.models.Response``: Slack Bot의 응답

        Examples:
            >>> slack.message("test")
            <Response [200]>
        """
        if message == "":
            return None
        if codeblock:
            message = f"```{message}```"
        data = self.data.copy()
        data["text"] = message
        return requests.post(
            "https://slack.com/api/chat.postMessage",
            headers=self.headers,
            json=data,
            timeout=10,
        )

    def file(self, path: str) -> requests.models.Response:
        """Slack Bot을 통해 file 전송

        Note:
            ``name`` 과 ``icon_*`` 의 적용 불가

        Args:
            path (``str``): 전송할 file 경로

        Returns:
            ``requests.models.Response``: Slack Bot의 응답

        Examples:
            >>> slack.file("test.jpg")
            <Response [200]>
        """
        with open(path, "rb") as file:
            response = requests.post(
                "https://slack.com/api/files.upload",
                headers=self.headers,
                files={"file": file},
                data={"channels": self.channel},
                timeout=10,
            )
        return response
