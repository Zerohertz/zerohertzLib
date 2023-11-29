import os

import zerohertzLib as zz

OPENAI_TOKEN = os.environ.get("OPENAI_TOKEN")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")

tmp = os.path.dirname(__file__)
data = os.path.join(tmp, "data")


def test_discord_message():
    discord = zz.api.Discord(DISCORD_WEBHOOK_URL)
    for response in discord.message("Testing..."):
        assert response.status_code == 204


def test_discord_messages():
    discord = zz.api.Discord(DISCORD_WEBHOOK_URL)
    for response in discord.message("Testing..." * 200):
        assert response.status_code == 204


def test_discord_image():
    discord = zz.api.Discord(DISCORD_WEBHOOK_URL)
    response = discord.image(f"{data}/test.jpg")
    assert response.status_code == 200


def test_slack_webhook():
    slack = zz.api.SlackWebhook(
        SLACK_WEBHOOK_URL, "test", name="Test Webhook", icon_emoji="wrench"
    )
    response = slack.message("Testing...")
    assert response.status_code == 200


def test_slack_bot_message():
    slack = zz.api.SlackBot(
        SLACK_BOT_TOKEN, "test", name="Test Bot", icon_emoji="hammer"
    )
    response = slack.message("Testing...")
    assert response.status_code == 200


def test_slack_bot_file():
    slack = zz.api.SlackBot(
        SLACK_BOT_TOKEN, "test", name="Test Bot", icon_emoji="hammer"
    )
    response = slack.file(f"{data}/test.jpg")
    assert response.status_code == 200


def test_openai():
    client = zz.api.OpenAI(OPENAI_TOKEN)
    response = client("오늘 기분이 어때? 1줄로 대답해줘.", model="gpt3")
    assert isinstance(response, str)
    slack = zz.api.SlackBot(
        SLACK_BOT_TOKEN, "test", name="Test Bot", icon_emoji="hammer"
    )
    slack.message(response)
