import argparse
from .planning import run_today


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history-days", type=int, default=365)
    args = p.parse_args()
    run_today(history_days=args.history_days)


if __name__ == "__main__":  # pragma: no cover
    main()
