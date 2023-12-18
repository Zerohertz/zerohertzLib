# pylint: disable = C0103
"""
MIT License

Copyright (c) 2023 Hyogeun Oh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from zerohertzLib import algorithm, monitoring, plot, util

try:
    from zerohertzLib import api
except ImportError:
    pass

try:
    from zerohertzLib import logging
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

__version__ = "v0.6.9"
