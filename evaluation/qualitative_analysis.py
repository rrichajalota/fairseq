import argparse
import spacy
from typing import List
import pandas as pd

allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'] # or any other types
nlp = spacy.load('en_core_web_trf')
# nlp = spacy.load('de_dep_news_trf')

def token_filter(token):    
    return (token.pos_ in allowed_postags) & (not (token.is_punct | token.is_space | 
    token.is_stop)) # | len(token.text) <= 2

def type_token_ratio(text):
    if not text.strip():
        raise ValueError
    tokens = text.split()
    types = set(tokens)
    return len(types) / len(tokens)

def lexical_density(all_docs: List[str]):
    content_words = 0
    total_words = 0
    all_docs = [str(doc) for doc in all_docs]

    for doc in nlp.pipe(all_docs):
        content_toks = [token.lemma_ for token in doc if token_filter(token)]
        # print(content_toks)
        content_words += len(content_toks)
        total_words += len(doc)

    return (content_words/total_words)*100

def length_variety(src, hyp):
    return abs(len(str(src))-len(str(hyp))) / len(str(src))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run qualitative analysis')
    parser.add_argument("--file", default="/netscratch/anonymous/datasets/motra-preprocessed/en_de/test/src_hyp/699528.txt")
    parser.add_argument("--translated", action='store_true')
    args = parser.parse_args()
    print(args.file)

    if not args.translated:
        df = pd.read_csv(args.file, sep="\t", names=['source', 'hypothesis'], header=0)

        df['ttr'] = df.apply(lambda row : type_token_ratio(str(row['hypothesis'])), axis = 1)

        df['lv'] = df.apply(lambda row : length_variety(row['source'], row['hypothesis']), axis = 1)
        
        print(f"AVG TTR: {df.loc[:, 'ttr'].mean()}")
        print(f"AVG lexical density: {lexical_density(df['hypothesis'].tolist())}")
        print(f"AVG length variety: {df.loc[:, 'lv'].mean()}")
    
    else:
        df = pd.read_csv(args.file, sep="\t", names=['text', 'label'])

        df['ttr'] = df.apply(lambda row : type_token_ratio(row['text']), axis = 1)

        print(f"AVG TTR: {df.loc[:, 'ttr'].mean()}")
        print(f"AVG lexical density: {lexical_density(df['text'].tolist())}")




        

            

