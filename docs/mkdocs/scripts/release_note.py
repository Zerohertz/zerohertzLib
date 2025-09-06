# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import argparse
import os

from loguru import logger

import zerohertzLib as zz


def main():
    logger.info(f"zerohertzLib version: {zz.__version__}")
    if args.token is None:
        gh = zz.api.GitHub()
    else:
        gh = zz.api.GitHub(token=args.token)
    gh.release_note(path=os.path.join(os.path.dirname(__file__), "..", ".."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token")
    args = parser.parse_args()
    main()
