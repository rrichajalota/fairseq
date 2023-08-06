import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract comparable corpus from Haifa Hansard')
    parser.add_argument("--out", default="/netscratch/jalota/datasets/haifa-hansard/")
    parser.add_argument("--inp", default="/netscratch/jalota/datasets/haifa-hansard/train/uniq_og")
    parser.add_argument("--split", default='train')
    parser.add_argument("--name", default="pp_uniq_og", help="without extension")
    args = parser.parse_args()
    clean = True # for MOTRA ; True for Hansard
    count = 0
    
    with open(args.inp) as f:
        # with open(f"{args.out}{args.split}/{args.name}", "w") as fo:
        with open(args.out, "w") as fo:
            for line in f.readlines():
                wds = line.split()
                if clean:
                    if len(wds) > 4 and len(wds) < 505:
                        fo.write(f"{line}")
                else:
                    if len(wds) > 505:
                        count +=1
                        print(len(wds))
                    else:
                        fo.write(f"{line}")

            print(f"{count} lines have length > 505:")
