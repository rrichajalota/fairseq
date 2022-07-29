#!/bin/bash

TEXT="data/toyset"
dest="data/data-bin-toyset/"
mkdir -p $dest

binarize_example_small() {
fairseq-preprocess --source-lang de --target-lang en \
 --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
 --destdir $dest --joined-dictionary --workers 20 --dataset-impl raw

}


binarize_big_mmap() {
    src="$1"
    tgt="$2"
    #p="/raid/data/daga01/fairseq_train/data/BPE_32k_red_new_${src}${tgt}"
    #dest="/raid/data/daga01/fairseq_train/data/data-bin-32k-red-lazy-new-${src}-${tgt}"

    p="/raid/data/daga01/data_ro-en/my_training_testing/BPE_32k"
    dest="/raid/data/daga01/data_ro-en/my_training_testing/data-bin-mmap-roen-test20"
    mkdir -p $dest
    
    #--cpu
    CUDA_VISIBLE_DEVICES=1,2,3,4 fairseq-preprocess \
    --source-lang $src --target-lang $tgt \
    --trainpref $p/corpus --validpref $p/dev  \
    --testpref $p/test20 \
    --destdir $dest \
    --joined-dictionary --dataset-impl mmap --workers 16


echo "Done binarizing"
}




binarize_test_with_dicts() {
    src="$1"
    tgt="$2"
    p="/raid/data/daga01/fairseq_train/data/BPE_32k_red_new/test14-${src}-${tgt}"
    dest="/raid/data/daga01/fairseq_train/data/data-bin-32k-red-lazy-new-${src}-${tgt}"

    mkdir -p $dest

    fairseq-preprocess \
    --source-lang $src --target-lang $tgt \
    --testpref $p/test_14  \
    --destdir $dest \
    --srcdict "${dest}/dict.${src}.txt"  --tgtdict "${dest}/dict.${tgt}.txt" \
    --dataset-impl lazy --workers 16

    #--joined-dictionary \
echo "Done binarizing"
}


binarize_de() {
    p="/raid/data/daga01/fairseq_train/lm_models/my_LM_de"
    dest="${p}/data-bin"
    
    
    CUDA_VISIBLE_DEVICES=1,2,3,4 fairseq-preprocess \
    --only-source \
    --trainpref $p/corpus_bpe.de --validpref $p/dev_bpe.de --testpref $p/test_14_bpe.de  \
    --destdir $dest \
    --joined-dictionary --dataset-impl lazy --workers 16


echo "Done binarizing"
}


#binarize_de

#binarize_big_lazy "en" "de"
binarize_big_mmap "src" "tgt"
