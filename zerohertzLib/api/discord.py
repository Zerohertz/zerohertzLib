import json
import time
from typing import List, Optional

import requests


class Discord:
    """Discord Webhook의 데이터 전송을 위한 클래스

    Args:
        webhook_url (``str``): Discord Webhook의 URL

    Examples:
        >>> discord = zz.api.Discord("https://discord.com/api/webhooks/...")
    """

    def __init__(self, webhook_url: str):
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
        return requests.post(self.webhook_url, data=json.dumps(data), headers=headers)

    def message(
        self,
        message: str,
        t: Optional[int] = 1,
        codeblock: Optional[bool] = False,
    ) -> List[requests.models.Response]:
        """Discord Webhook에 메세지 전송

        Args:
            message (``str``): Discord Webhook의 입력
            t (``Optional[int]``): ``message`` 의 전송 간 간격 (``message`` 가 1500자 이내라면 0)
            codeblock (``Optional[bool]``): 전송되는 메세지의 스타일

        Returns:
            ``List[requests.models.Response]``: Discord Webhook의 응답

        Examples:
            >>> discord = zz.api.Discord("https://discord.com/api/webhooks/...")
            >>> discord.message("Testing...")
            [<Response [204]>]
        """
        contents = self._split_string_in_chunks(message, 1500)
        responses = []
        if len(contents) == 1:
            responses.append(self._message(contents[0], codeblock))
        else:
            for content in contents:
                responses.append(self._message(content, codeblock))
                if t > 0:
                    time.sleep(t)
        return responses

    def image(self, image_path: str) -> requests.models.Response:
        """Discord Webhook에 이미지 전송

        Args:
            image_path (``str``): 전송할 이미지 경로

        Returns:
            ``requests.models.Response``: Discord Webhook의 응답

        Examples:
            >>> discord = zz.api.Discord("https://discord.com/api/webhooks/...")
            >>> zz.api.image("test.jpg")
            <Response [200]>
        """
        with open(image_path, "rb") as f:
            files = {
                "file": (image_path, f),
            }
            response = requests.post(self.webhook_url, files=files)
        return response
