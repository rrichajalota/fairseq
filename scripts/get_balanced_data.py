import argparse
import random
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe", default="/netscratch/jalota/datasets/haifa-hansard/test/bpe-12k/")
    parser.add_argument("--txt", default="/netscratch/jalota/datasets/haifa-hansard/test/")
    parser.add_argument("--nsamples", default=4068, help="should be equal to the num samples in translated.txt", type=int)
    args = parser.parse_args()
    for txt_file in Path(args.txt).glob('*.txt'):
        txt_file = str(txt_file)
        # if 'bt_original' in txt_file:
        #     with open(txt_file) as f:
        #         bt_txt = f.readlines()
        # elif 'original' in txt_file:
        #     with open(txt_file) as f:
        #         og_txt = f.readlines()
        if 'translated' in txt_file:
            with open(txt_file) as f:
                tr_txt = f.readlines()
        else:
            continue
    for txt_file in Path(args.bpe).glob('*.bpe'):
        txt_file = str(txt_file)
        # if 'bt_original' in txt_file:
        #     with open(txt_file) as f:
        #         bt_bpe = f.readlines()
        # elif 'original' in txt_file:
        #     with open(txt_file) as f:
        #         og_bpe = f.readlines()
        if 'translated' in txt_file:
            with open(txt_file) as f:
                tr_bpe = f.readlines()
        else:
            continue
    
    # print(f"len(og_bpe): {len(og_bpe)}")
    # print(f"len(og_txt): {len(og_txt)}")
    # print(f"len(bt_bpe): {len(bt_bpe)}")
    # print(f"len(bt_txt): {len(bt_txt)}")
    print(f"len(tr_txt): {len(tr_txt)}")
    print(f"len(tr_bpe): {len(tr_bpe)}")

    # r = [random.randint(0, len(og_bpe)-1) for _ in range(args.nsamples)]
    r = [random.randint(0, len(tr_bpe)-1) for _ in range(args.nsamples)]
    # og_txt = [og_txt[i] for i in r]
    # og_bpe = [og_bpe[i] for i in r]
    # bt_bpe = [bt_bpe[i] for i in r]
    # bt_txt = [bt_txt[i] for i in r]
    tr_bpe = [tr_bpe[i] for i in r]
    tr_txt = [tr_txt[i] for i in r]
    # print(len(og_bpe))
    # print(og_bpe[0])
    # print(bt_bpe[0])
    print(tr_bpe[0])
    print(tr_txt[0])

    # with open(f"{args.txt}bt_bal.txt", "w") as wf:
    #     for line in bt_txt:
    #         wf.write(line)
    
    # with open(f"{args.txt}og_bal.txt", "w") as wf:
    #     for line in og_txt:
    #         wf.write(line)

    # with open(f"{args.bpe}bt_bal.bpe", "w") as wf:
    #     for line in bt_bpe:
    #         wf.write(line)

    # with open(f"{args.bpe}og_bal.bpe", "w") as wf:
    #     for line in og_bpe:
    #         wf.write(line)

    with open(f"{args.txt}tr_bal.txt", "w") as wf:
        for line in tr_txt:
            wf.write(line)

    with open(f"{args.bpe}tr_bal.bpe", "w") as wf:
        for line in tr_bpe:
            wf.write(line)

    


    
