#!/bin/bash


#training_data="/raid/data/daga01/fairseq_train/data-bin-32k-red-lazy-new"
#checkpoints="/raid/data/daga01/fairseq_train/checkpoints/basic-transf"

training_data="/raid/data/daga01/data/mt_train_sparse_200k/binarized"
checkpoints="/raid/data/daga01/fairseq_train/checkpoints/basic-transf-sparse-200k"

mkdir -p $checkpoints


#call_train_distributed() {
##For example, to train a large English-German Transformer model on 2 nodes each with 8 GPUs (in total 16 GPUs), run the following command on each node, replacing node_rank=0 with node_rank=1 on the second node:
## orig: --nnodes=2 --nproc_per_node=8
#
#python -m torch.distributed.launch --nproc_per_node=4 \
#    --nnodes=1 --node_rank=0 --master_addr="192.168.1.1" \
#    --master_port=1234 \
#    CUDA_VISIBLE_DEVICES=1,2,3,4  $(which fairseq-train) ${training_data} \
#    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
#    --lr 0.0005 --min-lr 1e-09 \
#    --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#    --max-tokens 3584  --save-dir ${checkpoints}
#}


call_simplest(){
CUDA_VISIBLE_DEVICES=1,2,3,4 fairseq-train "$training_data" \
    --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir "$checkpoints"
}

call_simplest_small(){
fairseq-train "$training_data" \
    --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 100 \
    --arch fconv_iwslt_de_en --save-dir "$checkpoints" --cpu --max-epoch 2 --dataset-impl raw
}

call_train_small(){
# 8 GPU cumul/update-freq 16 ?= 4 GPU update-freq 32
# lr=5e-4 - orig; here I'll change to 2xlr=1e-3 as stated in the paper

fairseq-train ${training_data} \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1 \
    --lr 0.001 --min-lr 1e-09 \
    --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 100  --save-dir ${checkpoints} --keep-last-epochs 3  \
    --dataset-impl raw  --cpu --max-epoch 3
    #--ddp-backend no_c10d --update-freq 32
}

call_train_big(){
# 8 GPU cumul/update-freq 16 ?= 4 GPU update-freq 32 
# lr=5e-4 - orig; here I'll change to 2xlr=1e-3 as stated in the paper

CUDA_VISIBLE_DEVICES=1,2,3,4  $(which fairseq-train) ${training_data} \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.001 --min-lr 1e-09 \
    --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584  --save-dir ${checkpoints} --keep-last-epochs 3  \
    --dataset-impl lazy 
    #--ddp-backend no_c10d --update-freq 32 
}


call_train_big_wide(){
# lr=5e-4 - orig; here I don't change to 2xlr=1e-3 as stated in the 2018 paper because my batches are not so big

CUDA_VISIBLE_DEVICES=1,2,3,4  $(which fairseq-train) ${training_data} \
    --arch transformer_vaswani_wmt_en_de_big_wide --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-09 \
    --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584  --save-dir ${checkpoints} --keep-last-epochs 10  \
    --dataset-impl lazy --max-epoch 45
    #--update-freq 32 --max-epoch 3
    #--num-workers 4 
    #--update-freq 8 
}



call_train_basic(){

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  $(which fairseq-train) ${training_data} \
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
 	--eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ${checkpoints}  --dataset-impl lazy 
}

call_train_sparse(){

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  $(which fairseq-train) ${training_data} \
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt \
    --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
 	--eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --patience 7 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ${checkpoints}  --dataset-impl lazy 
}


echo $(which python)

LOG="checkpoints/LOG-sparse"
call_train_sparse > $LOG 2>&1

#LOG="/raid/data/daga01/fairseq_train/LOG_R2_32k_wide"
#time -p call_train_big_wide > $LOG 2>&1
