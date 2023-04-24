import os
import argparse
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    """
    Run: sed -i '1,53d' gen.txt to remove logger outputs before passing the generated file. 
    """
    parser = argparse.ArgumentParser(description='generate test data for binary classification from fairseq-generate output')
    parser.add_argument("--file", default="/home/jalota/gen_w_threshold_translated_test.txt")
    parser.add_argument("--out_dir", default="/netscratch/jalota/datasets/motra-preprocessed/en_de/test/generated/")
    parser.add_argument("--name", default="pred_test.tsv")
    args = parser.parse_args()
    contains_dup = False
    # if "bt_test" in args.file: 
    #     contains_dup = True

    # gen_modifiedComparable_translated_test.txt
    # gen_no_threshold.txt
    # gen_w_threshold_translated_test.txt

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with open(args.file, encoding="utf-8") as f:
        lines = f.readlines()
        with open(args.out_dir+args.name, "w") as of:
            if not contains_dup:
                count = 0
                for i, line in enumerate(lines):
                    if line.startswith("H-"):
                        line = line.split()
                        line = " ".join(line[2:])
                        # tr = lines[i-2].split()
                        # tr = " ".join(tr[1:])
                        # if tr.strip() == "!" or tr.strip() == "co-rapporteur ." or tr.strip() == "Thank you very much for your attention .":
                        #     print(tr)
                        #     continue
                        of.write(f"{line}\t1")
                        of.write("\n")
                        count += 1
                    else:
                        continue
                print(count)
            else:
                i = 0
                bt2og_like = dict()
                while i < len(lines):
                    if lines[i].startswith("T-"):
                        tr = lines[i].split()
                        tr = " ".join(tr[1:])
                        i += 2
                        if i < len(lines) and lines[i].startswith("D-"):
                            og_like = lines[i].split()
                            og_like = " ".join(og_like[2:])

                            if tr not in bt2og_like:
                                bt2og_like[tr] = og_like
                    i += 1
                ogl_list = bt2og_like.values()
                print(f"len ogl_list: {len(ogl_list)}")
                for ogl in ogl_list:
                    of.write(f"{ogl}\t1")
                    of.write("\n")
            
            

        
        


