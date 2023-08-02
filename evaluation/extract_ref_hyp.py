import os
import argparse
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract reference and hypothesis from model generation')
    parser.add_argument("--file", default="/netscratch/anonymous/results/generations/unsup/motra-old/699517/generate-test.txt")
    parser.add_argument("--out_dir", default="/netscratch/jalota/datasets/motra-preprocessed/en_de/test/src_hyp/")
    parser.add_argument("--name", default="699517.tsv")
    args = parser.parse_args()
    contains_dup = False
    # if "bt_test" in args.file: 
    #     contains_dup = True

    # gen_modifiedComparable_translated_test.txt
    # gen_no_threshold.txt
    # gen_w_threshold_translated_test.txt

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    srcs = []
    hyps = []

    with open(args.file, encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("H-"):
                line = line.split()
                line = " ".join(line[2:])
                hyps.append(line)
            elif line.startswith("S-"):
                line = line.split()
                line = " ".join(line[1:])
                srcs.append(line)
            else:
                continue
        print(len(srcs), len(hyps))
        df = pd.DataFrame(
            {
                'source': srcs,
                'hypothesis': hyps
            }
        )
        df.to_csv(args.out_dir+args.name, sep='\t', index=False)
    