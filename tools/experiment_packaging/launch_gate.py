"""Precondition check before any future package push. This file never pushes.

There is no Kaggle submission command in this repository. A process that
passes --push is refused before receipt verification and before any network
call. Receipt verification still does not grant a launch.
"""
import argparse
import json
import sys

from verify_receipt import verify


def main(argv=None):
    parser = argparse.ArgumentParser(prog="launch-gate")
    parser.add_argument("root")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args(argv)
    if args.push:
        print("REFUSING_PUSH", file=sys.stderr)
        return 2
    print(json.dumps(verify(args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
