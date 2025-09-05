# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

from zerohertzLib import algorithm, monitoring, plot, util

__all__ = ["algorithm", "monitoring", "plot", "util"]

try:
    from zerohertzLib import api  # noqa: F401

    __all__.append("api")
except ImportError:
    pass

try:
    from zerohertzLib import mlops  # noqa: F401

    __all__.append("mlops")
except ImportError:
    pass

try:
    from zerohertzLib import quant  # noqa: F401

    __all__.append("quant")
except ImportError:
    pass

try:
    from zerohertzLib import vision  # noqa: F401

    __all__.append("vision")
except ImportError as error:
    print("=" * 100)
    print(f"[Warning] {error}")
    print("Please Install OpenCV Dependency")
    print("--->\t$ sudo apt install python3-opencv -y\t<---")
    print("(but you can use other submodules except zerohertzLib.vision)")
    print("=" * 100)

__version__ = "v1.2.0"
