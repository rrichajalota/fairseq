import pandas as pd

if __name__ == '__main__':
    train_tr="/netscratch/jalota/datasets/europarl-ppd/de/europarl_no_dup.tok"
    dev_tr="/netscratch/jalota/datasets/europarl-ppd/de/europarl_dev.txt"
    test_tr="/netscratch/jalota/datasets/europarl-ppd/de/europarl_test.txt"
    # train_tr = "/netscratch/jalota/datasets/haifa-hansard/train/translated.txt"
    # dev_tr = "/netscratch/jalota/datasets/haifa-hansard/dev/translated_4k_train.txt"
    # test_tr = "/netscratch/jalota/datasets/haifa-hansard/test/translated_4k_train.txt"

    # extract 8k sentences from train split and redistribute to dev and test splits
    train = open(train_tr)
    dev = open(dev_tr, 'w')
    test = open(test_tr, 'w')
    train_w = open("/netscratch/jalota/datasets/europarl-ppd/de/europarl_train.txt", "w")
    # train_w = open("/netscratch/jalota/datasets/haifa-hansard/train/tr_new", "w")

    lines = train.readlines()
    df = pd.DataFrame(lines, columns=['text'])
    print(df.head())
    dev_df = df.sample(n=5000, random_state=23)
    df = df.drop(dev_df.index)
    test_df = df.sample(n=5000, random_state=23)
    df = df.drop(test_df.index)

    print(len(df), len(test_df), len(dev_df))
    print(not set(df).isdisjoint(test_df))
    print(not set(df).isdisjoint(dev_df))
    print(not set(test_df).isdisjoint(dev_df))
    
    for row in test_df['text']:
        test.write(row)
    for row in dev_df['text']:
        dev.write(row)
    for row in df['text']:
        train_w.write(row)

    train.close()
    train_w.close()
    dev.close()
    test.close()
    

