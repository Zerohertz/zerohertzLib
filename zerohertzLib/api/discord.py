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
from typing import List, Optional

import requests


class Discord:
    """Discord Webhook의 데이터 전송을 위한 class

    Args:
        webhook_url (``str``): Discord Webhook의 URL

    Examples:
        >>> discord = zz.api.Discord("https://discord.com/api/webhooks/...")
    """

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def _split_string_in_chunks(self, text: str, chunk_size: int) -> List[str]:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _message(
        self, message: str, codeblock: Optional[bool] = False
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
        gap: Optional[int] = 1,
        codeblock: Optional[bool] = False,
    ) -> List[requests.models.Response]:
        """Discord Webhook에 message 전송

        Args:
            message (``str``): Discord Webhook의 입력
            gap (``Optional[int]``): ``message`` 의 전송 간 간격 (``message`` 가 1500자 이내라면 0)
            codeblock (``Optional[bool]``): 전송되는 message의 스타일

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
        """Discord Webhook에 image 전송

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
