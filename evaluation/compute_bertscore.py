import evaluate
import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute bertscore given a tsv file with src and hypothesis')
    parser.add_argument("--file", default="/netscratch/anonymous/datasets/motra-preprocessed/en_de/test/src_hyp/699528.txt")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--model", default='roberta-base')
    args = parser.parse_args()
    print(args.model)
    print(args.file)
    count = 0
    c = 0
    df = pd.read_csv(args.file, sep="\t", names=['source', 'hypothesis'], header=0)
    for index, row in df.iterrows():
        if row['source'] == row['hypothesis']:
            count += 1
        else:
            if c < 10:
                print(f"row['source']: {row['source']}, row['hypothesis]: {row['hypothesis']}")
                c += 1
    print(f"count: {count}")
    print(f"test set size: {len(df)}")
        
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=df.hypothesis.tolist(), references=df.source.tolist(), model_type=args.model, lang='en')
    
    print(f"average precision: {sum(results['precision'])/len(results['precision'])}")
    print(f"average recall: {sum(results['recall'])/len(results['recall'])}")
    print(f"average f1: {sum(results['f1'])/len(results['f1'])}")
        