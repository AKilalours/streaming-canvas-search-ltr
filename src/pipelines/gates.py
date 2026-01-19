# Commit 6: read metrics/latency from a run folder and fail if thresholds violated
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", required=True)
    ap.add_argument("--run_dir", required=True)
    _ = ap.parse_args()
    print("Gates stub. Implement in Commit 6.")


if __name__ == "__main__":
    main()

