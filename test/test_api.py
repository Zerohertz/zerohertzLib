import os

import zerohertzLib as zz

WEBHOOK = "https://discord.com/api/webhooks/1174193014923591791/vBPMpb0otKQH0lflp169u0a-8gJPZyDg17SPEsxKDDlmv3PMFl4eNrt3KWQgUmnWpYJ9"
tmp = os.path.dirname(__file__)
data = os.path.join(tmp, "data")


def test_discord_message():
    discord = zz.api.Discord(WEBHOOK)
    for response in discord.message("Testing..."):
        assert response.status_code == 204


def test_discord_messages():
    discord = zz.api.Discord(WEBHOOK)
    for response in discord.message("Testing..." * 200):
        assert response.status_code == 204


def test_discord_image():
    discord = zz.api.Discord(WEBHOOK)
    response = discord.image(f"{data}/test.jpg")
    assert response.status_code == 200
