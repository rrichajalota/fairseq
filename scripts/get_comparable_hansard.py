import pandas as pd
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract comparable corpus from Haifa Hansard')
    parser.add_argument("--dir", default="/ds/text/corpora_translationese_research_rabinovich/hansard.EN-FR/committees/")
    parser.add_argument("--out", default="/netscratch/jalota/datasets/haifa-hansard/")
    #parser.add_argument("--fname", default="snli_train", help="without extension")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    for path in Path(args.dir).glob('devtest.*'):
        print(path)
        path = str(path)

        og_ids = set()
        tr_ids = set()
        names = ['dev1', 'dev2', 'test1', 'test2']

        for name in names:
            og = open(args.out+name+"_original2.txt", 'w')
            tr = open(args.out+name+"_translated_fr2.txt", 'w')

            with open(path+"/"+name+".id") as idf:
                ids = idf.readlines()
                for i, line in enumerate(ids):
                    if 'LANGUAGE="EN"' in line:
                        og_ids.add(i)
                    else:
                        tr_ids.add(i)
            print(len(tr_ids))
            print(len(og_ids))
            
            
            with open(path+"/"+name+".en.tok") as f:
                for i, line in enumerate(f):
                    if i in og_ids:
                        og.write(line)
                    else:
                        tr.write(line)
            
            og.close()
            tr.close()



