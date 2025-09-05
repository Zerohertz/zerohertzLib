# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import os

import zerohertzLib as zz


def test_storage():
    path = zz.monitoring.storage()
    assert path.split("/")[-1] in os.listdir()


def test_cpu():
    zz.monitoring.cpu()
    assert "cpu.png" in os.listdir()
