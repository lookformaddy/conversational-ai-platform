"""End-to-end demo: run the scripted multi-turn conversation from the CLI."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from convai.cli import run_demo  # noqa: E402

if __name__ == "__main__":
    run_demo()
