import os
import argparse
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    """
    Run: sed -i '1,53d' gen.txt to remove logger outputs before passing the generated file. 
    sed '1,56d' gen.txt > new_gen.txt
    """
    parser = argparse.ArgumentParser(description='generate test data for binary classification from fairseq-generate output')
    parser.add_argument("--file", default="/home/jalota/gen_w_threshold_translated_test.txt")
    parser.add_argument("--out_dir", default="/netscratch/jalota/test_perplexity/")
    parser.add_argument("--name", default="test")
    parser.add_argument("--exp", default="712684")
    args = parser.parse_args()
    contains_dup = False
    
    path = args.out_dir + args.exp 

    Path(path).mkdir(parents=True, exist_ok=True)

    with open(args.file, encoding="utf-8") as f:
        lines = f.readlines()
        with open(f"{path}/{args.name}", "w") as of:
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
                        of.write(f"{line}")
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
            
            

        
        


