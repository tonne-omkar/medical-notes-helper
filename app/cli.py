"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="medical-notes-helper", description="Clinic staff use this CLI tool to draft administrative sections of medical notes by generating next-word suggestions, ensuring consistency and reducing manual typing.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Medical Notes Helper is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
