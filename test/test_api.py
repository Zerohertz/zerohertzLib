import os
import random
import time

import zerohertzLib as zz

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
GH_TOKEN = os.environ.get("GH_TOKEN")
TIME_SLEEP = 20

tmp = os.path.dirname(__file__)
data = os.path.join(tmp, "data")


def test_discord_messages():
    discord = zz.api.Discord(DISCORD_WEBHOOK_URL)
    time.sleep(random.randrange(TIME_SLEEP))
    for response in discord.message("Testing..." * 200):
        assert response.status_code == 204


def test_discord_image():
    discord = zz.api.Discord(DISCORD_WEBHOOK_URL)
    time.sleep(random.randrange(TIME_SLEEP))
    response = discord.image(f"{data}/test.jpg")
    assert response.status_code == 200


def test_slack_webhook():
    slack = zz.api.SlackWebhook(
        SLACK_WEBHOOK_URL, "test", name="Test Webhook", icon_emoji="wrench"
    )
    time.sleep(random.randrange(TIME_SLEEP))
    response = slack.message("Testing...")
    assert response.status_code == 200


def test_slack_bot_message():
    slack = zz.api.SlackBot(
        SLACK_BOT_TOKEN, "test", name="Test Bot", icon_emoji="hammer"
    )
    time.sleep(random.randrange(TIME_SLEEP))
    response = slack.message("Testing...")
    assert response.status_code == 200


def test_slack_bot_file():
    slack = zz.api.SlackBot(
        SLACK_BOT_TOKEN, "test", name="Test Bot", icon_emoji="hammer"
    )
    time.sleep(random.randrange(TIME_SLEEP))
    response = slack.file(f"{data}/test.jpg")
    assert response.status_code == 200


def test_discord_webhook():
    webhook = zz.api.DiscordWebhook(DISCORD_WEBHOOK_URL)
    time.sleep(random.randrange(TIME_SLEEP))
    response = webhook.send_message("Testing DiscordWebhook...")
    assert response.status_code == 204


def test_discord_bot_message():
    bot = zz.api.DiscordBot(DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID)
    time.sleep(random.randrange(TIME_SLEEP))
    response = bot.message("Testing DiscordBot message...")
    assert response.status_code == 200


def test_discord_bot_file():
    bot = zz.api.DiscordBot(DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID)
    time.sleep(random.randrange(TIME_SLEEP))
    response = bot.file(f"{data}/test.jpg")
    assert response.status_code == 200


def test_discord_bot_create_thread():
    bot = zz.api.DiscordBot(DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID)
    time.sleep(random.randrange(TIME_SLEEP))

    # 먼저 메시지 전송
    message_response = bot.message("Testing thread creation...")
    assert message_response.status_code == 200
    message_data = message_response.json()
    message_id = message_data.get("id")

    # 스레드 생성
    time.sleep(2)
    thread_response = bot.create_thread(message_id, "Test Thread")
    assert thread_response.status_code in [200, 201]

    # 스레드에 댓글 작성
    thread_data = thread_response.json()
    thread_id = thread_data.get("id")
    time.sleep(2)
    reply_response = bot.message("Thread reply test", thread_id)
    assert reply_response.status_code == 200


def test_discord_bot_create_thread_with_replies():
    bot = zz.api.DiscordBot(DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID)
    time.sleep(random.randrange(TIME_SLEEP))

    responses = bot.create_thread_with_replies(
        "Testing thread with replies...",
        "Automated Test Thread",
        ["First reply", "Second reply", "Third reply"],
        reply_delay=1,
    )

    assert "message_id" in responses
    assert "thread_id" in responses
    assert len(responses["reply_responses"]) == 3
    for reply_response in responses["reply_responses"]:
        assert reply_response.status_code == 200


def test_github_release_note():
    gh = zz.api.GitHub(token=GH_TOKEN)
    gh.release_note()


# def test_openai():
#     client = zz.api.OpenAI(OPENAI_API_KEY)
#     response = client("오늘 기분이 어때? 1줄로 대답해줘.", model="gpt3.5")
#     assert isinstance(response, str)
#     slack = zz.api.SlackBot(
#         SLACK_BOT_TOKEN, "test", name="Test Bot", icon_emoji="hammer"
#     )
#     slack.message(response)
