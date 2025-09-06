# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Zerohertz (Hyogeun Oh)

import argparse

from .triton import TritonClientURL


def _trictl() -> None:
    """
    Triton Inference CLI
    """
    parser = argparse.ArgumentParser(
        description="Triton Inference Server CLI\n\n"
        "Exmaples:\n\ttrictl load 0\n"
        "\ttrictl unload model\n"
        "\ttrictl status",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "action",
        type=str,
        choices=["load", "unload", "status"],
        help="load, unload: Control models if the triton inference server is running with '--model-control-mode=explicit'\n"
        "status: Display the current status of the triton inference server",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="The name or ID of the model to load or unload",
    )
    parser.add_argument(
        "-g",
        "--grpc",
        type=str,
        default="localhost:8001",
        help="The endpoint of the triton inference server, including the port",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()
    url, port = args.grpc.split(":")
    client = TritonClientURL(url, port, verbose=args.verbose)
    if args.action in ["load", "unload"]:
        if args.model.isdigit():
            args.model = int(args.model)
        getattr(client, f"{args.action}_model")(args.model)
    elif args.action == "status":
        client.status()
