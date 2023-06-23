import argparse
import pandas as pd

def read_file(path):
    srcs = []
    tgts = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split("\t")
            for part in parts:
                if part.startswith('src'):
                    instances = part.split()
                    srcs.append(instances[1].strip())
                elif part.startswith('tgt'):
                    instances = part.split()
                    tgts.append(instances[1].strip())
                else:
                    continue
    return srcs, tgts

if __name__ == '__main__':
    # argparse the file name and outdir 
    src, tgt = read_file(path)
    sdf = pd.DataFrame(src)
    tdf = pd.DataFrame(tgt)
    sdf['label'] = 1
    tdf['label'] = 0
    df = pd.concat([sdf, tdf], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(args.out)
