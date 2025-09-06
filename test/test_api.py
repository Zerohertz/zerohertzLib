# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import os
import random
import time

import pytest

import zerohertzLib as zz

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
DISCORD_BOT_CHANNEL = os.environ.get("DISCORD_BOT_CHANNEL")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
SLACK_CHANNEL = "test"
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
GH_TOKEN = os.environ.get("GH_TOKEN")
TIME_SLEEP = 20

tmp = os.path.dirname(__file__)
data = os.path.join(tmp, "data")


def test_slack_webhook() -> None:
    slack = zz.api.SlackWebhook(
        SLACK_WEBHOOK_URL,
        SLACK_CHANNEL,
        name="Test Webhook",
        icon_emoji="wrench",
    )
    time.sleep(random.randrange(TIME_SLEEP))
    response = slack.message("Testing...")
    assert response.status_code == 200

    with pytest.raises(NotImplementedError):
        response = slack.file(os.path.join(data, "test.jpg"))


def test_slack_bot_message() -> None:
    slack = zz.api.SlackBot(
        SLACK_BOT_TOKEN, SLACK_CHANNEL, name="Test Bot", icon_emoji="hammer"
    )
    time.sleep(random.randrange(TIME_SLEEP))
    response = slack.message("Testing...")
    assert response.status_code == 200
    thread_id = slack.get_thread_id(response)

    response = slack.message("Thread reply test", codeblock=True, thread_id=thread_id)
    assert response.status_code == 200
    response = slack.file(os.path.join(data, "test.jpg"), thread_id=thread_id)
    assert response.status_code == 200


def test_slack_bot_file() -> None:
    slack = zz.api.SlackBot(
        SLACK_BOT_TOKEN, SLACK_CHANNEL, name="Test Bot", icon_emoji="hammer"
    )
    time.sleep(random.randrange(TIME_SLEEP))
    response = slack.file(os.path.join(data, "test.jpg"))
    assert response.status_code == 200


def test_discord_webhook_message() -> None:
    discord = zz.api.DiscordWebhook(DISCORD_WEBHOOK_URL)
    time.sleep(random.randrange(TIME_SLEEP))
    response = discord.message("Testing...")
    assert response.status_code == 204


def test_discord_webhook_file() -> None:
    discord = zz.api.DiscordWebhook(DISCORD_WEBHOOK_URL)
    time.sleep(random.randrange(TIME_SLEEP))
    response = discord.file(os.path.join(data, "test.jpg"))
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
    time.sleep(random.randrange(TIME_SLEEP))

    rust_code = """fn main() {
    let name = "World";
    println!("Hello, {}!", name);
    let numbers = vec![1, 2, 3, 4, 5];
    println!("Numbers: {:?}", numbers);
}"""
    response = discord.message(rust_code, codeblock="rust")
    assert response.status_code == 200
    time.sleep(random.randrange(TIME_SLEEP))

    go_code = """package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}"""
    response = discord.message(go_code, codeblock="go")
    assert response.status_code == 200
    time.sleep(random.randrange(TIME_SLEEP))

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
    response = discord.file(os.path.join(data, "test.jpg"))
    assert response.status_code == 200


def test_discord_bot_create_thread() -> None:
    discord = zz.api.DiscordBot(DISCORD_BOT_TOKEN, DISCORD_BOT_CHANNEL)

    response = discord.message("Testing thread creation...")
    assert response.status_code == 200
    time.sleep(random.randrange(TIME_SLEEP))

    thread_id = discord.get_thread_id(response=response)
    time.sleep(random.randrange(TIME_SLEEP))

    response = discord.message("Thread reply test", thread_id=thread_id)
    assert response.status_code == 200

    response = discord.file(os.path.join(data, "test.jpg"), thread_id=thread_id)
    assert response.status_code == 200


def test_github_release_note() -> None:
    gh = zz.api.GitHub(token=GH_TOKEN)
    gh.release_note()
