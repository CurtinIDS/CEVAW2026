from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


parser = ArgumentParser()
parser.add_argument("parquet")
args = parser.parse_args()

inpath = Path(args.parquet)
outpath = inpath.with_suffix(".csv")

df = pd.read_parquet(inpath)

print(f"Converting {inpath} to {outpath}...")
for column in df.columns:
    if column[-7:] == "_reason":
        df.drop(columns=[column], inplace=True)

df.to_csv(outpath)



