"""CLI wiring for CoppeliaSim verification hooks."""

from __future__ import annotations

import argparse

from myarm.adapters.verify_fk_coppelia import configure_verify_fk_parser, run_verify_fk
from myarm.adapters.verify_ik_coppelia import configure_verify_ik_parser, run_verify_ik


def cmd_verify_fk(args: argparse.Namespace) -> int:
    return run_verify_fk(args)


def cmd_verify_ik(args: argparse.Namespace) -> int:
    return run_verify_ik(args)


def register_subparsers(subparsers: argparse._SubParsersAction) -> None:
    verify = subparsers.add_parser("verify", help="compare against CoppeliaSim")
    verify_sub = verify.add_subparsers(dest="verify_command", required=True)

    verify_fk = verify_sub.add_parser("fk", help="verify FK using CoppeliaSim")
    configure_verify_fk_parser(verify_fk)
    verify_fk.set_defaults(func=cmd_verify_fk)

    verify_ik = verify_sub.add_parser("ik", help="verify IK using CoppeliaSim")
    configure_verify_ik_parser(verify_ik)
    verify_ik.set_defaults(func=cmd_verify_ik)
