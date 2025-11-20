"""Unified command-line interface for the myarm manipulator toolkit.

The parser definitions are delegated to the individual CLI modules under
``myarm.cli`` so the entry point stays lightweight and imports stay fast.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from myarm.cli import fk, ik, jacobian, verify, velocity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="myarm", description="myarm robotics CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    fk.register_subparsers(sub)
    ik.register_subparsers(sub)
    jacobian.register_subparsers(sub)
    verify.register_subparsers(sub)
    velocity.register_subparsers(sub)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
