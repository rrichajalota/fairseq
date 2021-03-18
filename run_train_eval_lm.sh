#!/bin/bash

set -u 
set -e
set -o pipefail

top="/raid/data/daga01/fairseq_train"

outdir="${top}/lm_models/my_LM"

train_lm(){
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
  --max-tokens 2048 --update-freq 16 \
  --max-update 50000 \
  --dataset-impl lazy

}


eval_lm() {
lang="$1"
echo "Language: $lang"
test_data="${top}/data/data-bin-32k-red-lazy-new-renamed-and-lm/${lang}"
fairseq-eval-lm ${test_data} \
	--cpu \
    --path "${outdir}/checkpoint_best.pt" \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400


}

train_lm
#eval_lm "de"
#eval_lm "en"
