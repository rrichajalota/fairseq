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
testdir="${top}/data/data-bin-32k-red-lazy-new-renamed"
outdir="/raid/data/daga01/fairseq_out_nostopwords"

#--cpu \
CUDA_VISIBLE_DEVICES=4,5 python fairseq_cli/generate.py $testdir --path $model \
 --batch-size 128 --beam 10 --nbest 10 \
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
outdir="/raid/data/daga01/fairseq_out_test"
mkdir -p "$outdir"
#--cpu \
CUDA_VISIBLE_DEVICES=7 python fairseq_cli/generate.py $testdir --path $model \
 --batch-size 8 --beam 2 --nbest 12 \
 --dataset-impl lazy \
 --print-alignment \
 --tokenizer moses -s $src -t $tgt \
 --remove-bpe sentencepiece \
 --sacrebleu \
 --results-path $outdir
}


#call_basic_model "en" "de"
call_basic_model_smalltest "en" "de"
