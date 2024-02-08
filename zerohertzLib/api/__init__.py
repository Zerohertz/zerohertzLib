"""
.. admonition:: API
    :class: hint

    다양한 API를 쉽게 사용할 수 있는 class들
"""

from zerohertzLib.api.discord import Discord
from zerohertzLib.api.github import GitHub
from zerohertzLib.api.koreainvestment import KoreaInvestment
from zerohertzLib.api.open_ai import OpenAI
from zerohertzLib.api.slack import SlackBot, SlackWebhook

__all__ = ["Discord", "GitHub", "OpenAI", "SlackWebhook", "SlackBot", "KoreaInvestment"]
