import zerohertzLib as zz


def test_discord():
    for response in zz.api.send_discord_message(
        "https://discord.com/api/webhooks/1147058609130311690/geDqeubwN5vKomEExJnFckS9SSYFJgvdJh2IX-02kHWmWjyGgzCebEvLzyj6Gn1f92fd",
        "Testing..." * 1,
    ):
        assert response.status_code == 204

    for response in zz.api.send_discord_message(
        "https://discord.com/api/webhooks/1147058609130311690/geDqeubwN5vKomEExJnFckS9SSYFJgvdJh2IX-02kHWmWjyGgzCebEvLzyj6Gn1f92fd",
        "Testing..." * 200,
    ):
        assert response.status_code == 204
