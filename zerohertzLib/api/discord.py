import json
import time
from typing import List

import requests


def _split_string_in_chunks(text, chunk_size):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def send_discord_message(
    webhook_url: str, message: str
) -> List[requests.models.Response]:
    """Discord Webhook에 메세지 전송

    Args:
        message (``str``): Discord Webhook의 입력

    Returns:
        ``List[requests.models.Response]``: Discord Webhook의 응답

    Examples:
        >>> import zerohertzLib as zz
        >>> zz.api.send_discord_message("https://discord.com/api/webhooks/...", "Testing...")
        [<Response [204]>]
    """
    headers = {"Content-Type": "application/json"}
    contents = _split_string_in_chunks(message, 1500)
    responses = []
    for content in contents:
        data = {"content": content}
        responses.append(
            requests.post(webhook_url, data=json.dumps(data), headers=headers)
        )
        time.sleep(1)
    return responses
