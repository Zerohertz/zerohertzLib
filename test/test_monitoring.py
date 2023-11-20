import os

import zerohertzLib as zz


def test_storage():
    zz.monitoring.storage(".")
    assert "zerohertzlib.png" in os.listdir()


def test_cpu():
    zz.monitoring.cpu()
    assert "cpu.png" in os.listdir()
