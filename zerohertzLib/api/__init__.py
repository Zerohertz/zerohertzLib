# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

"""
!!! hint "API"
    다양한 API를 쉽게 사용할 수 있는 class들
"""

from zerohertzLib.api.discord import DiscordBot, DiscordWebhook
from zerohertzLib.api.github import GitHub
from zerohertzLib.api.koreainvestment import KoreaInvestment
from zerohertzLib.api.slack import SlackBot, SlackWebhook

__all__ = [
    "DiscordBot",
    "DiscordWebhook",
    "GitHub",
    "KoreaInvestment",
    "SlackBot",
    "SlackWebhook",
]
