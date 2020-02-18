#!/bin/bash

TEXT="data/toyset"
dest="data/data-bin-toyset/"
mkdir -p $dest

binarize_example_small() {
fairseq-preprocess --source-lang de --target-lang en \
 --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
 --destdir $dest --joined-dictionary --workers 20 --dataset-impl raw

}





binarize_big_lazy() {
    #p="/raid/data/daga01/fairseq_train/BPE_32k_red"
    #dest="/raid/data/daga01/fairseq_train/data-bin-32k-red-lazy"

    p="/raid/data/daga01/fairseq_train/BPE_32k_red_new"
    dest="/raid/data/daga01/fairseq_train/data-bin-32k-red-lazy-new-2"
    
    mkdir -p $dest

    CUDA_VISIBLE_DEVICES=1,2,3,4 fairseq-preprocess --source-lang src --target-lang tgt --trainpref $p/corpus --validpref $p/dev  --testpref $p/test --destdir $dest --joined-dictionary --dataset-impl lazy --workers 16


echo "Done binarizing"
}




binarize_test() {
    p="/raid/data/daga01/fairseq_train/mini_test"
    dest="/raid/data/daga01/fairseq_train/mini-test-bin/"

    mkdir -p $dest

    fairseq-preprocess --cpu --source-lang src --target-lang tgt --trainpref $p/mini_test  --destdir $dest --joined-dictionary --dataset-impl lazy --workers 16

echo "Done binarizing"
}




binarize_big_lazy
#binarize_example_small
#binarize_test
