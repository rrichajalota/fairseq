#!/bin/bash

set -u 
set -e
set -o pipefail

top="/raid/data/daga01/fairseq_train"
outdir="/raid/data/daga01/fairseq_out"



call_basic_model() {
src="$1"
tgt="$2"

model="${top}/checkpoints/basic-transf/checkpoint_best.pt"

#testdir="${top}/data/data-bin-32k-red-lazy-new-renamed"
testdir="${top}/data/data-bin-32k-red-lazy-new-renamed-and-lm"

lm_model="${top}/lm_models/my_LM/checkpoint_best.pt"

#outdir="/raid/data/daga01/fairseq_out_nostopwords"
#outdir="/raid/data/daga01/fairseq_out_pplscore2"
outdir="/raid/data/daga01/fairseq_out_new/fairseq_out_vectorbertscore_euclidean_myLMde_beam50"

mkdir -p "$outdir"

#--cpu \
#--lm-path $lm_model \
#--lm-weight 0.0 \
#--num-workers 8 \

CUDA_VISIBLE_DEVICES=0,1,2,3 python fairseq_cli/generate.py $testdir --path $model \
 --batch-size 40 \
 --beam 50 --nbest 50 \
 --dataset-impl lazy \
 --print-alignment \
 --tokenizer moses -s $src -t $tgt \
 --remove-bpe sentencepiece \
 --sacrebleu \
 --results-path $outdir
}


call_basic_model_smalltest() {
src="$1"
tgt="$2"

model="${top}/checkpoints/basic-transf/checkpoint_best.pt"
testdir="${top}/data/data-bin-32k-red-lazy-new-shorter-minitest-5st-renamed"
outdir="/raid/data/daga01/fairseq_out_all/fairseq_out_test"
#lm_model="${top}/lm_models/adaptive_lm_wiki103.v2/model.pt"
lm_model="${top}/lm_models/my_LM/checkpoint_best.pt"
mkdir -p "$outdir"
#--cpu \
#--lm-path $lm_model \
# --lm-weight 0.0 \

CUDA_VISIBLE_DEVICES=7 python fairseq_cli/generate.py $testdir \
 --path $model \
 --batch-size 8 --beam 2 --nbest 2 \
 --dataset-impl lazy \
 --print-alignment \
 --tokenizer moses -s $src -t $tgt \
 --remove-bpe sentencepiece \
 --sacrebleu \
 --results-path $outdir
}


call_basic_model "en" "de"
#call_basic_model_smalltest "en" "de"
