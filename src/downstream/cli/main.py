import argparse

from downstream.config.constants import (
    PROJECT_NAME,
    PROJECT_VERSION,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="downstream",
        description="DownStream Model Projection Toolkit",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        print(f"{PROJECT_NAME} {PROJECT_VERSION}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
