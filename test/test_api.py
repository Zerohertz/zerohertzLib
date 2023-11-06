import zerohertzLib as zz


def test_discord():
    for response in zz.api.send_discord_message(
        "https://discord.com/api/webhooks/1170962638583373904/xVJKW1KkNo7Pc1HykJ85cHs_4SvRkKCbOvbf1qe1j8QXOnJyTGyJy8T7sI7kvfA8SGb-",
        "Testing..." * 1,
    ):
        assert response.status_code == 204

    for response in zz.api.send_discord_message(
        "https://discord.com/api/webhooks/1170962638583373904/xVJKW1KkNo7Pc1HykJ85cHs_4SvRkKCbOvbf1qe1j8QXOnJyTGyJy8T7sI7kvfA8SGb-",
        "Testing..." * 200,
    ):
        assert response.status_code == 204
