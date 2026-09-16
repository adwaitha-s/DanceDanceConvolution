"""Minimal example of consuming the pose signal stream produced by pose_demo.py.

Reads JSON Lines records (one per frame) either from a file or from stdin, and
prints a rolling estimate of each tracked person's wrist height relative to
their shoulders -- a stand-in for whatever downstream model (e.g. a
convolution over pose-sequence windows) would consume this stream instead.

Usage:
    python pose_demo.py --track --stdout --no-display | python examples/example_consumer.py
    python examples/example_consumer.py --in pose_stream.jsonl
"""

import argparse
import json
import sys


def hands_up(person: dict) -> bool:
    kp = person["keypoints"]
    l_wrist_y, r_wrist_y = kp["left_wrist"][1], kp["right_wrist"][1]
    l_shoulder_y, r_shoulder_y = kp["left_shoulder"][1], kp["right_shoulder"][1]
    return l_wrist_y < l_shoulder_y and r_wrist_y < r_shoulder_y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", default=None,
                        help="JSONL file to read (default: stdin)")
    args = parser.parse_args()

    stream = open(args.infile) if args.infile else sys.stdin

    for line in stream:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        for person in record["people"]:
            gesture = "hands up" if hands_up(person) else "-"
            print(f"frame={record['frame']:>5} id={person['id']:>3} "
                  f"conf={person['conf']:.2f} {gesture}")


if __name__ == "__main__":
    main()
