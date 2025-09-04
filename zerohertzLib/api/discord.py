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
import time

import requests


class DiscordWebhook:
    """Discord Webhook의 data 전송을 위한 class

    Args:
        webhook_url (``str``): Discord Webhook의 URL

    Examples:
        >>> discord = zz.api.Discord("https://discord.com/api/webhooks/...")
    """

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def _split_string_in_chunks(self, text: str, chunk_size: int) -> list[str]:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _message(
        self, message: str, codeblock: bool = False
    ) -> requests.models.Response:
        headers = {"Content-Type": "application/json"}
        if codeblock:
            message = "```\n" + message + "\n```"
        data = {"content": message}
        return requests.post(
            self.webhook_url, data=json.dumps(data), headers=headers, timeout=10
        )

    def message(
        self,
        message: str,
        gap: int = 1,
        codeblock: bool = False,
    ) -> list[requests.models.Response]:
        """Discord Webhook을 통해 message 전송

        Args:
            message (``str``): Discord Webhook의 입력
            gap (``int``): ``message`` 의 전송 간 간격 (``message`` 가 1500자 이내라면 0)
            codeblock (``bool``): 전송되는 message의 스타일

        Returns:
            ``List[requests.models.Response]``: Discord Webhook의 응답

        Examples:
            >>> discord = zz.api.Discord("https://discord.com/api/webhooks/...")
            >>> discord.message("Testing...")
            [<Response [204]>]
        """
        cts = self._split_string_in_chunks(message, 1500)
        responses = []
        if len(cts) == 1:
            responses.append(self._message(cts[0], codeblock))
        else:
            for content in cts:
                responses.append(self._message(content, codeblock))
                if gap > 0:
                    time.sleep(gap)
        return responses

    def image(self, image_path: str) -> requests.models.Response:
        """Discord Webhook을 통해 image 전송

        Args:
            image_path (``str``): 전송할 image 경로

        Returns:
            ``requests.models.Response``: Discord Webhook의 응답

        Examples:
            >>> discord = zz.api.Discord("https://discord.com/api/webhooks/...")
            >>> zz.api.image("test.jpg")
            <Response [200]>
        """
        with open(image_path, "rb") as file:
            files = {
                "file": (image_path, file),
            }
            response = requests.post(self.webhook_url, files=files, timeout=10)
        return response


class DiscordBot:
    """Discord Bot의 data 전송을 위한 class

    Args:
        token (``str``): Discord Bot 토큰
        channel (``str``): Discord Bot이 전송할 channel
        timeout (``int``): API 요청 시 사용될 timeout

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
        codeblock: str = "",
        thread_id: str | None = None,
    ) -> requests.models.Response:
        """Discord Bot을 통해 message 전송

        Args:
            message (``str``): 전송할 message
            codeblock (``str``): 전송되는 message의 스타일
            thread_id (``str | None``): 댓글을 전송할 thread의 ID

        Returns:
            ``requests.models.Response``: Discord Bot의 응답

        Examples:
            >>> response = discord.message("test")
            >>> response
            <Response [200]>
            >>> response = discord.message('print("hi")', codeblock="python")
            >>> response
            <Response [200]>
        """
        if codeblock:
            message = f"```{codeblock}\n" + message + "\n```"
        channel_id = thread_id if thread_id else self.channel
        payload = {"content": message}
        return requests.post(
            f"{self.api_base}/channels/{channel_id}/messages",
            data=json.dumps(payload),
            headers=self.headers,
            timeout=self.timeout,
        )

    def create_thread(
        self,
        name: str,
        message_id: str,
    ) -> requests.models.Response:
        """메시지에서 스레드 생성

        Args:
            name (``str``): 스레드 이름
            message_id (``str``): 스레드를 생성할 메시지 ID

        Returns:
            ``requests.models.Response``: Discord Bot의 응답

        Examples:
            >>> response = discord.message("test")
            >>> response
            <Response [200]>
            >>> response = discord.create_thread("test", response.json()["id"])
            >>> response
            <Response [201]>
            >>> response = discord.message("test", thread_id=response.json()["id"])
            <Response [200]>
        """
        payload = {
            "name": name,
            "type": 11,
        }
        return requests.post(
            f"{self.api_base}/channels/{self.channel}/messages/{message_id}/threads",
            data=json.dumps(payload),
            headers=self.headers,
            timeout=self.timeout,
        )

    def file(self, path: str, thread_id: str | None = None) -> requests.models.Response:
        """Discord Bot을 통해 file 전송

        Args:
            path (``str``): 전송할 file 경로
            thread_id (``str | None``): 댓글을 전송할 thread의 ID

        Returns:
            ``requests.models.Response``: Discord Bot의 응답

        Examples:
            >>> response = bot.file("test.jpg")
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
