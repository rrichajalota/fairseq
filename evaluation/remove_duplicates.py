import os
import argparse
import pandas as pd
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate test data for binary classification from fairseq-generate output')
    parser.add_argument("--file", default="/netscratch/jalota/datasets/motra-preprocessed/en_de/test/generated/gen_test_59030.tsv")
    parser.add_argument("--og_file", default="/netscratch/jalota/datasets/motra-preprocessed/en_de/test/test.tsv")
    parser.add_argument("--out_dir", default="/netscratch/jalota/datasets/motra-preprocessed/en_de/test/generated/")
    args = parser.parse_args()

    seen = set()
    dup = set()
    uniq = []
    df = pd.read_csv(args.file, sep='\t', skipinitialspace=True, header=None, names=['text', 'label'])
    # df.sort(columns=[0], inplace=True)
    df.sort_values(by='text', inplace=True)
    for _, row in df.iterrows():
        if row[0] in seen and row[1] == 0:
            print(row[0])
            seen.remove(row[0])
            dup.add(row[0])
        elif row[0] in seen and row[1] ==1:
            print(f"label 1: {row[0]}")
        else:
            seen.add(row[0])
    
    for _, row in df.iterrows():
        if row[0] not in dup:
            uniq.append(row)

    print(len(uniq))
    df2 = pd.DataFrame(uniq)
    print(df2.head())
    df2.to_csv(args.out_dir+"no_dup_gen_test_59030.tsv", header=None, index=False, sep='\t')

    print(f"num examples in each label: {df2.groupby('label').size()}")

    print(len(dup))

    df_og = pd.read_csv(args.og_file, sep='\t', skipinitialspace=True, header=None, names=['text', 'label'])
    fin_test = []
    count = 0
    for _, row in df_og.iterrows():
        if row[0] not in dup: # removes original sent. found in dup
            fin_test.append(row)
        else:
            count +=1 
    
    print(f"count: {count}")
    dff = pd.DataFrame(fin_test)
    to_remove = np.random.choice(dff[dff['label']==1].index,size=count,replace=False) # remove equal number of translated
    dff.drop(to_remove, inplace=True)
    print(len(dff))
    dff.to_csv(args.out_dir+"modified_test_59030.tsv", header=None, index=False, sep='\t')
    print(f"num examples in each label: {dff.groupby('label').size()}")




#################
"""""
dev.tsv -> original.tsv translated.tsv (comparable)
test.tsv -> original.tsv translated.tsv (comparable)
"""""


        
