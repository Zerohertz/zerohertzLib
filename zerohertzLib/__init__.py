# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

from zerohertzLib import algorithm, monitoring, plot, util

try:
    from zerohertzLib import api
except ImportError:
    pass

try:
    from zerohertzLib import mlops
except ImportError:
    pass

try:
    from zerohertzLib import quant
except ImportError:
    pass

try:
    from zerohertzLib import vision
except ImportError as error:
    print("=" * 100)
    print(f"[Warning] {error}")
    print("Please Install OpenCV Dependency")
    print("--->\t$ sudo apt install python3-opencv -y\t<---")
    print("(but you can use other submodules except zerohertzLib.vision)")
    print("=" * 100)

__all__ = ["algorithm", "monitoring", "plot", "util", "api", "mlops", "quant", "vision"]
__version__ = "v1.2.2"
