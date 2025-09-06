# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import argparse

from .compare import grid, vert
from .gif import img2gif, vid2gif
from .loader import ImageLoader


def _vert() -> None:
    parser = argparse.ArgumentParser(
        description="zerohertzLib.vision.vert()\n"
        "https://zerohertz.github.io/zerohertzLib/zerohertzLib.vision.html#zerohertzLib.vision.vert",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("path", nargs="?", default="./")
    parser.add_argument("-f", "--file-name", type=str, default="tmp")
    parser.add_argument("-s", "--shape", type=int, default=2000)
    args = parser.parse_args()
    ils = ImageLoader(args.path)
    ils.image_paths.sort()
    vert(imgs=[img for _, img in ils], height=args.shape, file_name=args.file_name)


def _grid() -> None:
    parser = argparse.ArgumentParser(
        description="zerohertzLib.vision.grid()\n"
        "https://zerohertz.github.io/zerohertzLib/zerohertzLib.vision.html#zerohertzLib.vision.grid",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("path", nargs="?", default="./")
    parser.add_argument("-f", "--file-name", type=str, default="tmp")
    parser.add_argument("-c", "--cnt", type=int, default=0)
    parser.add_argument("-s", "--shape", type=int, default=2000)
    args = parser.parse_args()
    ils = ImageLoader(args.path)
    ils.image_paths.sort()
    if args.cnt == 0:
        imgs = [img for _, img in ils]
    else:
        imgs = [ils[i][1] for i in range(args.cnt**2)]
    grid(imgs=imgs, size=args.shape, file_name=args.file_name)


def _img2gif() -> None:
    parser = argparse.ArgumentParser(
        description="zerohertzLib.vision.img2gif()\n"
        "https://zerohertz.github.io/zerohertzLib/zerohertzLib.vision.html#zerohertzLib.vision.img2gif",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("path", nargs="?", default="./")
    parser.add_argument("-f", "--file-name", type=str, default="tmp")
    parser.add_argument("-d", "--duration", type=int, default=500)
    args = parser.parse_args()
    img2gif(path=args.path, file_name=args.file_name, duration=args.duration)


def _vid2gif() -> None:
    parser = argparse.ArgumentParser(
        description="zerohertzLib.vision.vid2gif()\n"
        "https://zerohertz.github.io/zerohertzLib/zerohertzLib.vision.html#zerohertzLib.vision.vid2gif",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("path")
    parser.add_argument("-f", "--file-name", type=str, default="tmp")
    parser.add_argument("-q", "--quality", type=int, default=100)
    parser.add_argument("-p", "--fps", type=int, default=15)
    parser.add_argument("-s", "--speed", type=float, default=1.0)
    args = parser.parse_args()
    vid2gif(
        path=args.path,
        file_name=args.file_name,
        quality=args.quality,
        fps=args.fps,
        speed=args.speed,
    )
