#!/bin/bash

set -u 
set -e
set -o pipefail

top="/raid/data/daga01/data_ro-en/"


call_basic_model() {
src="$1"
tgt="$2"
gpu="$3"
beam="$4"
nbest="$5"

model="${top}/my_training_testing/checkpoints/checkpoint_best.pt"

#testdir="${top}/my_training/data-bin-mmap-roen"
testdir="${top}/my_training_testing/data-bin-mmap-roen-test20"

#outdir="${top}/eval_model_out/fo_beam${beam}"
#outdir="${top}/experiment_out/fo_vectorbertscore_cosine_beam${beam}"
outdir="${top}/fo_forcedecode/fo_vectorbertscore_cosine_beam${beam}"

mkdir -p "$outdir"

#--nbest $beam

CUDA_VISIBLE_DEVICES="${gpu}" python fairseq_cli/generate.py $testdir --path $model \
 --batch-size 40 \
 --beam $beam --nbest $nbest \
 --dataset-impl mmap \
 --tokenizer moses -s $src -t $tgt \
 --remove-bpe sentencepiece \
 --sacrebleu \
 --print-alignment \
 --results-path $outdir
}




gpu="1"
beam=10
nbest=1
call_basic_model "src" "tgt" "$gpu" "$beam" "$nbest"
#call_basic_model "ro" "en" "$gpu" "$beam" "$nbest"
