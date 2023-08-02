import argparse

def detokenize_bpe_string(bpe_string):
    return bpe_string.replace("@@ ", "")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract comparable corpus from Haifa Hansard')
    parser.add_argument("--out", default="/netscratch/jalota/datasets/haifa-hansard/test/")
    parser.add_argument("--inp", default="/netscratch/jalota/datasets/haifa-hansard/fairseq-pp/unsup_setup/filtered/test.tr-og.og")
    parser.add_argument("--name", default="new_filtered_og", help="without extension")
    args = parser.parse_args()

    with open(args.inp) as f:
        with open(f"{args.out}/{args.name}", "w") as fo:
        # with open(args.out, "w") as fo:
            for line in f.readlines():
                fo.write(f"{detokenize_bpe_string(line)}")