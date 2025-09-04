import os
import random
import time

import zerohertzLib as zz

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
DISCORD_BOT_CHANNEL = os.environ.get("DISCORD_BOT_CHANNEL")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
GH_TOKEN = os.environ.get("GH_TOKEN")
TIME_SLEEP = 20

tmp = os.path.dirname(__file__)
data = os.path.join(tmp, "data")


def test_slack_webhook() -> None:
    slack = zz.api.SlackWebhook(
        SLACK_WEBHOOK_URL, "test", name="Test Webhook", icon_emoji="wrench"
    )
    time.sleep(random.randrange(TIME_SLEEP))
    response = slack.message("Testing...")
    assert response.status_code == 200


def test_slack_bot_message() -> None:
    slack = zz.api.SlackBot(
        SLACK_BOT_TOKEN, "test", name="Test Bot", icon_emoji="hammer"
    )
    time.sleep(random.randrange(TIME_SLEEP))
    response = slack.message("Testing...")
    assert response.status_code == 200


def test_slack_bot_file() -> None:
    slack = zz.api.SlackBot(
        SLACK_BOT_TOKEN, "test", name="Test Bot", icon_emoji="hammer"
    )
    time.sleep(random.randrange(TIME_SLEEP))
    response = slack.file(f"{data}/test.jpg")
    assert response.status_code == 200


def test_discord_webhook_message() -> None:
    discord = zz.api.DiscordWebhook(DISCORD_WEBHOOK_URL)
    time.sleep(random.randrange(TIME_SLEEP))
    for response in discord.message("Testing..." * 200):
        assert response.status_code == 204


def test_discord_webhook_image() -> None:
    discord = zz.api.DiscordWebhook(DISCORD_WEBHOOK_URL)
    time.sleep(random.randrange(TIME_SLEEP))
    response = discord.image(f"{data}/test.jpg")
    assert response.status_code == 200


def test_discord_bot_message() -> None:
    discord = zz.api.DiscordBot(DISCORD_BOT_TOKEN, DISCORD_BOT_CHANNEL)
    time.sleep(random.randrange(TIME_SLEEP))
    response = discord.message("Testing DiscordBot message...")
    assert response.status_code == 200


def test_discord_bot_codeblock() -> None:
    discord = zz.api.DiscordBot(DISCORD_BOT_TOKEN, DISCORD_BOT_CHANNEL)
    time.sleep(random.randrange(TIME_SLEEP))

    cpp_code = """#include <iostream>
int main() {
    std::cout << "Hello World!" << std::endl;
    return 0;
}"""
    response = discord.message(cpp_code, codeblock="cpp")
    assert response.status_code == 200

    rust_code = """fn main() {
    let name = "World";
    println!("Hello, {}!", name);
    let numbers = vec![1, 2, 3, 4, 5];
    println!("Numbers: {:?}", numbers);
}"""
    response = discord.message(rust_code, codeblock="rust")
    assert response.status_code == 200

    go_code = """package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}"""
    response = discord.message(go_code, codeblock="go")
    assert response.status_code == 200

    python_code = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print([fibonacci(i) for i in range(10)])"""
    response = discord.message(python_code, codeblock="python")
    assert response.status_code == 200


def test_discord_bot_file() -> None:
    discord = zz.api.DiscordBot(DISCORD_BOT_TOKEN, DISCORD_BOT_CHANNEL)
    time.sleep(random.randrange(TIME_SLEEP))
    response = discord.file(f"{data}/test.jpg")
    assert response.status_code == 200


def test_discord_bot_create_thread() -> None:
    discord = zz.api.DiscordBot(DISCORD_BOT_TOKEN, DISCORD_BOT_CHANNEL)
    time.sleep(random.randrange(TIME_SLEEP))

    # 먼저 메시지 전송
    message_response = discord.message("Testing thread creation...")
    assert message_response.status_code == 200
    message_data = message_response.json()
    message_id = message_data.get("id")

    # 스레드 생성
    time.sleep(2)
    thread_response = discord.create_thread(message_id, "Test Thread")
    assert thread_response.status_code in [200, 201]

    # 스레드에 댓글 작성
    thread_data = thread_response.json()
    thread_id = thread_data.get("id")
    time.sleep(2)
    reply_response = discord.message("Thread reply test", thread_id)
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
