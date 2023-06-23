import evaluate
import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute perplexity of generated sentences')
    parser.add_argument("--file", default="/netscratch/anonymous/datasets/motra-preprocessed/en_de/test/unsup-generated/pred_no_th_699528.tsv")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--model", default='/netscratch/anonymous/checkpoints/gpt2-finetuned-motra/')
    args = parser.parse_args()
    print(args.model)
    print(args.file)
    
    df = pd.read_csv(args.file, sep="\t", names=['text', 'label'])

    perplexity = evaluate.load("perplexity", module_type="measurement")
    ppl_results = perplexity.compute(data=df['text'].tolist(), model_id=args.model, batch_size=args.batch_size, add_start_token=True)
    print(f"perplexity: {ppl_results['mean_perplexity']}")