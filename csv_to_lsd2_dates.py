#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd

def main(in_csv: str, out_txt: str) -> None:
    """
    Converts a CSV file containing leaf names and their associated dates
    into the LSD2 date format required for phylogenetic tree calibration.

    The output format is:
        <number_of_leaves>
        leaf<name_1> <date_1>
        leaf<name_2> <date_2>
        ...

    Args:
        in_csv (str): Path to the input CSV file with two columns: leaf names and dates.
        out_txt (str): Path to the output TXT file formatted for LSD2.
    """
    # Try reading the CSV with header; fallback to no header
    try:
        df = pd.read_csv(in_csv, dtype=str)
        if df.shape[1] < 2:
            raise ValueError
        leaf_col = df.columns[0]
        date_col = df.columns[1]
    except Exception:
        df = pd.read_csv(in_csv, header=None, dtype=str)
        if df.shape[1] < 2:
            sys.stderr.write(f"Error: {in_csv} must contain at least two columns.\n")
            sys.exit(1)
        leaf_col = 0
        date_col = 1

    # Drop rows with missing values in either column
    df = df.dropna(subset=[leaf_col, date_col]).copy()

    # Prefix each leaf name with "leaf" (remove extra spaces)
    df["__leaf_prefixed__"] = df[leaf_col].astype(str).apply(lambda s: f"leaf{s.strip()}")

    # Convert date to 6-decimal float string (avoid rounding to years)
    def to_float_str(x: str) -> str:
        try:
            f = float(x)
        except:
            sys.stderr.write(f"Error: Could not convert {x} to float.\n")
            sys.exit(1)
        return f"{f:.6f}"

    df["__date_str__"] = df[date_col].apply(to_float_str)

    # Total number of leaves
    count = df.shape[0]

    # Write LSD2-compatible date file
    with open(out_txt, "w") as f:
        f.write(f"{count}\n")
        for _, row in df.iterrows():
            leaf_name = row["__leaf_prefixed__"]
            date_str = row["__date_str__"]
            f.write(f"{leaf_name} {date_str}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: python3 csv_to_lsd2_dates.py input_leaf_dates.csv output_lsd2_dates.txt\n")
        sys.exit(1)

    in_csv = sys.argv[1]
    out_txt = sys.argv[2]

    main(in_csv, out_txt)

