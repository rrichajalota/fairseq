#!/bin/bash

set -u 
set -e
set -o pipefail

top="/raid/data/daga01/fairseq_train"


train_lm(){
outdir="${top}/lm_models/my_LM_ende2"
traindata="${top}/data/data-bin-32k-red-lazy-new-renamed-and-lm"
mkdir -p $outdir
#fairseq-train --task cross_lingual_lm \
fairseq-train --task language_modeling \
  $traindata \
  --train-subset 'train.en-de.en' \
  --valid-subset 'valid.en-de.en' \
  --save-dir $outdir \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 \
  --update-freq 16 \
  --num-workers 4 \
  --max-update 200000 \
  --save-interval-updates 10 --keep-interval-updates 5 --keep-best-checkpoints 7
#  --reset-dataloader \
#  --dataset-impl lazy

}


train_lm_de(){
traindata="${top}/lm_models/my_LM_de/data-bin"
outdir="${top}/lm_models/my_LM_de_2/checkpoints"
mkdir -p $outdir
fairseq-train --task language_modeling \
  $traindata \
  --train-subset 'train' \
  --valid-subset 'valid' \
  --save-dir $outdir \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 \
  --update-freq 16 \
  --num-workers 8 \
  --max-update 200000 \
  --save-interval-updates 10 --keep-interval-updates 5 --keep-best-checkpoints 7 
#  --reset-dataloader \
#  --dataset-impl lazy

}



eval_lm() {
lang="$1"
#outdir="${top}/lm_models/my_LM_de/chackpoints"
outdir="${top}/lm_models/my_LM_de_2"
#outdir="${top}/lm_models/my_LM_de/chp_copy_ca52000updates"
echo "Language: $lang"
#test_data="${top}/data/data-bin-32k-red-lazy-new-renamed-and-lm/${lang}"
test_data="/raid/data/daga01/fairseq_train/lm_models/my_LM_de_2/data-bin"
#--cpu \

# I call calculate distances at every generate, for both LM and translation task. The path is actually in my code
# TODO: should actually receive it as an argument

fairseq-eval-lm ${test_data} \
    --path "${outdir}/checkpoint_best.pt" \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400


}

#train_lm_de
train_lm
#eval_lm "de"
#eval_lm "en"
