import json
import time
from typing import List, Optional

import requests


def _split_string_in_chunks(text, chunk_size):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def send_discord_message(
    webhook_url: str,
    message: str,
    t: Optional[int] = 1,
    codeblock: Optional[bool] = False,
) -> List[requests.models.Response]:
    """Discord Webhook에 메세지 전송

    Args:
        webhook_url (``str``): Discord Webhook의 URL
        message (``str``): Discord Webhook의 입력
        t (``int``): ``message`` 의 전송 간 간격
        codeblock (``bool``): 전송되는 메세지의 스타일

    Returns:
        ``List[requests.models.Response]``: Discord Webhook의 응답

    Examples:
        >>> zz.api.send_discord_message("https://discord.com/api/webhooks/...", "Testing...")
        [<Response [204]>]
    """
    headers = {"Content-Type": "application/json"}
    contents = _split_string_in_chunks(message, 1500)
    responses = []
    for content in contents:
        if codeblock:
            content = "```\n" + content + "\n```"
        data = {"content": content}
        responses.append(
            requests.post(webhook_url, data=json.dumps(data), headers=headers)
        )
        if t > 0:
            time.sleep(t)
    return responses
