#!/usr/bin/env python3
"""CLI entry point: argparse, load_dotenv, OPENAI_API_KEY check, run_chat."""

from __future__ import annotations

import argparse
import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

from chat_cli import run_chat


def main() -> None:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set. Set it in the environment or in a .env file.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Interactive CLI chat with OpenAI (Responses API).")
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="Model ID (default: gpt-5.2)",
    )
    parser.add_argument(
        "--system",
        action="append",
        default=None,
        metavar="TEXT",
        help="System prompt line (can be repeated for multiple lines).",
    )
    parser.add_argument(
        "--assistant",
        action="append",
        default=None,
        metavar="TEXT",
        help="Assistant prompt line (can be repeated for multiple lines).",
    )
    args = parser.parse_args()

    system = "\n".join(args.system) if args.system else None
    assistant = "\n".join(args.assistant) if args.assistant else None
    run_chat(system=system, assistant=assistant, model=args.model)


if __name__ == "__main__":
    main()
