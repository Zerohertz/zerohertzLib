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

import argparse

from .compare import grid, vert
from .loader import ImageLoader


def _vert() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="./")
    parser.add_argument("-s", "--shape", type=int, default=2000)
    args = parser.parse_args()
    ils = ImageLoader(args.path)
    vert([img for _, img in ils], args.shape)


def _grid() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="./")
    parser.add_argument("-c", "--cnt", type=int, default=0)
    parser.add_argument("-s", "--shape", type=int, default=2000)
    args = parser.parse_args()
    ils = ImageLoader(args.path)
    ils.image_paths.sort()
    if args.cnt == 0:
        imgs = [img for _, img in ils]
    else:
        imgs = [ils[i][1] for i in range(args.cnt**2)]
    grid(imgs, args.shape)
