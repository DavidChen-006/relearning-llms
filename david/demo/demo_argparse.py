"""Playground for argparse — the same pattern train.py uses for CLI flags.

Try these:
    python david/demo/demo_argparse.py
    python david/demo/demo_argparse.py --epochs 3
    python david/demo/demo_argparse.py --epochs 5 --batch-size 64 --name david
    python david/demo/demo_argparse.py --dry-run
    python david/demo/demo_argparse.py --help
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Tiny argparse demo")
    parser.add_argument("--epochs", type=int, default=1, help="how many passes over the data")
    parser.add_argument("--batch-size", type=int, default=32, help="samples per gradient step")
    parser.add_argument("--name", type=str, default="world", help="who to greet")
    parser.add_argument("--dry-run", action="store_true", help="print config without running")
    args = parser.parse_args()

    print("parsed args:")
    print(f"  epochs     = {args.epochs}")
    print(f"  batch_size = {args.batch_size}")
    print(f"  name       = {args.name!r}")
    print(f"  dry_run    = {args.dry_run}")

    if args.dry_run:
        print("\n(dry run — skipping fake training loop)")
        return

    print(f"\nHello, {args.name}!")
    for epoch in range(args.epochs):
        print(f"  epoch {epoch + 1}/{args.epochs} — would train {args.batch_size} samples at a time")


if __name__ == "__main__":
    main()
