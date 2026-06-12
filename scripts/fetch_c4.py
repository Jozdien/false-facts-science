"""Fetch C4 (en) webtext docs for SDF mixing (Slocum: 1:1 with synthetic docs)."""

import argparse

from datasets import load_dataset

from twohop.common import PROJECT_ROOT, save_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100_000)
    args = p.parse_args()

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    rows = []
    for ex in ds:
        if len(ex["text"]) > 20_000:  # skip extreme outliers
            continue
        rows.append({"content": ex["text"]})
        if len(rows) >= args.n:
            break
    out = PROJECT_ROOT / "data" / "c4" / f"c4_{args.n}.jsonl"
    save_jsonl(out, rows)
    print(f"wrote {len(rows)} docs to {out}")


if __name__ == "__main__":
    main()
