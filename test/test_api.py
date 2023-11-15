import zerohertzLib as zz

WEBHOOK = "https://discord.com/api/webhooks/1174193014923591791/vBPMpb0otKQH0lflp169u0a-8gJPZyDg17SPEsxKDDlmv3PMFl4eNrt3KWQgUmnWpYJ9"


def test_discord():
    for response in zz.api.send_discord_message(
        WEBHOOK,
        "Testing..." * 1,
    ):
        assert response.status_code == 204

    for response in zz.api.send_discord_message(
        WEBHOOK,
        "Testing..." * 200,
    ):
        assert response.status_code == 204
