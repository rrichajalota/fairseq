#!/bin/bash

set -u 
set -e
set -o pipefail

#top="/raid/data/daga01/fairseq_train"
#outdir="/raid/data/daga01/fairseq_out"



call_basic_model() {
top="/raid/data/daga01/fairseq_train"
src="$1"
tgt="$2"

cnt="$3"
#model="${top}/checkpoints/basic-transf/checkpoint_best.pt"
model="${top}/checkpoints/basic-transf-sparse-200k/checkpoint7.pt"

#testdir="${top}/data/data-bin-32k-red-lazy-new-renamed"
#testdir="${top}/data/data-bin-32k-red-lazy-new-renamed-and-lm"

#testdir="/raid/data/daga01/fairseq_train/data/data_beam_8parts/data-bin"

testdir="/raid/data/daga01/data/mt_train_sparse_200k/binarized"

#outdir="/raid/data/daga01/fairseq_out_nostopwords"
#outdir="/raid/data/daga01/fairseq_out_pplscore2"
#outdir="/raid/data/daga01/fairseq_out_sparse_cp7_beam50/fairseq_out_scalarmean_cosine_LMorig_beam50/${cnt}"
outdir="/raid/data/daga01/fairseq_out_sparse_cp7_beam50/fairseq_out_scalarmean_cosine_LMorig_beam50"

mkdir -p "$outdir"

#--print-alignment \
#--gen-subset "test${cnt}" \

CUDA_VISIBLE_DEVICES="${cnt}" python fairseq_cli/generate.py $testdir --path $model \
 --batch-size 20 \
 --beam 50 --nbest 50 \
 --gen-subset "test" \
 --dataset-impl lazy \
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


#call_basic_model "en" "de" "0" &
#call_basic_model "en" "de" "1" &
#call_basic_model "en" "de" "2" &
#call_basic_model "en" "de" "3" &
#call_basic_model "en" "de" "4" &
#call_basic_model "en" "de" "5" &
#call_basic_model "en" "de" "6" &
#call_basic_model "en" "de" "7" &

call_basic_model "en" "de" 0

