#!/bin/bash



call_train_basic(){
training_data="/raid/data/daga01/data_ro-en/my_training/data-bin-mmap-roen"
checkpoints="/raid/data/daga01/data_ro-en/my_training/checkpoints"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  $(which fairseq-train) ${training_data} \
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt \
    --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --dataset-impl mmap \
 	--eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 5 \
    --save-dir ${checkpoints} \
    --warmup-updates 1000 \
    --log-file "${checkpoints}/LOG"
}

echo $(which python)

LOG="checkpoints/LOG-transf-roen"
call_train_basic > $LOG 2>&1

