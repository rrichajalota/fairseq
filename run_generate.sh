#!/bin/bash



top="/raid/data/daga01/fairseq_train"
outdir="/raid/data/daga01/fairseq_out"


call_basic_model_toytest() {
model="${top}/checkpoints/basic-transf/checkpoint_best.pt"
#testdir="${top}/data/data-bin-32k-red-lazy-new"
testdir="${top}/data/data-bin-32k-red-lazy-new-shorter-minitest-5st"
#python generate.py $testdir --cpu --path $model --batch-size 8 --beam 10 --dataset-impl lazy --sacrebleu --remove-bpe sentencepiece
CUDA_VISIBLE_DEVICES=2 python generate.py $testdir --path $model --batch-size 8 --beam 10 --dataset-impl lazy --sacrebleu --remove-bpe sentencepiece

}




call_toy_model_toytest(){
model="${top}/checkpoints/toyset/checkpoint_best.pt"
testdir="${top}/data/data-bin-toyset"

python generate.py $testdir --cpu --path $model --batch-size 8 --beam 2 --dataset-impl raw
}




call_basic_model() {
model="${top}/checkpoints/basic-transf/checkpoint_best.pt"
#testdir="${top}/data/data-bin-32k-red-lazy-new-2"
testdir="${top}/data/data-bin-32k-red-lazy-new-short"
#fairseq-generate $testdir --cpu --path $model --beam 5 --dataset-impl lazy #--sacrebleu
python generate.py $testdir --cpu --path $model --batch-size 8 --beam 2 --dataset-impl lazy
}

#out="${top}/R2_result_basic_transf/wmt18_best_113/bleu.de-en"
#call_basic_model #> $out

#call_toy_model_toytest


call_basic_model_toytest > ${outdir}/OUT
