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
