#!/bin/bash

# We jump into the submission dir
cd ${SLURM_SUBMIT_DIR}
MASTER=`echo $SLURM_JOB_NODELIST | cut -d"," -f1 | sed 's/[][]//g' | cut -d "-" -f 1,2`
NUM_GPUS=4; # number of gpu
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1 
export USE_OPENMP=1  # prevents openblas to override OMP_NUM_THREADS

# But if you are using conda (uncomment the lines below)
. /netscratch/jalota/miniconda3/etc/profile.d/conda.sh
conda activate fsq012
python3 --version
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# conda install -c conda-forge cudatoolkit==11.7.0
export CUDA_HOME=/usr/local/cuda
nvcc --version
var=$(which nvcc)
echo "var: ${var}"
echo "cuda home: ${CUDA_HOME}"
nvidia-smi

echo "ngpus: ${NUM_GPUS}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "master: ${MASTER}"

# # Step 1. preprocess the data!
# python3 preprocess.py --dir /netscratch/jalota/datasets/europarl/ --out /netscratch/jalota/datasets/europarl-ppd/
# python3 preprocess.py --dir /netscratch/jalota/datasets/motra/en_de/dev --out /netscratch/jalota/datasets/motra-preprocessed/en_de/dev/
# python3 preprocess.py --dir /netscratch/jalota/datasets/motra/en_de/test --out /netscratch/jalota/datasets/motra-preprocessed/en_de/test/
# python3 preprocess.py --dir /netscratch/jalota/datasets/motra/en_es/train --out /netscratch/jalota/datasets/motra-preprocessed/en_es/train/
# python3 preprocess.py --dir /netscratch/jalota/datasets/motra/en_es/dev --out /netscratch/jalota/datasets/motra-preprocessed/en_es/dev/
# python3 preprocess.py --dir /netscratch/jalota/datasets/motra/en_es/test --out /netscratch/jalota/datasets/motra-preprocessed/en_es/test/

# # Step 2. run preprocess.sh
# ./preprocess.sh
# cat /netscratch/jalota/datasets/europarl-ppd/europarl.txt | sacremoses -l en -j 4 normalize -c -q -d tokenize truecase -m /netscratch/jalota/eu.truemodel > /netscratch/jalota/datasets/europarl-ppd/europarl.tok.txt

# in="/netscratch/jalota/datasets/europarl-ppd/europarl.tok.txt"
# train="$in.train"
# test="$in.test"
# awk -v train="$train" -v test="$test" '{if(rand()<0.9) {print > train} else {print > test}}' $in
# Step 3. Apply bpe and then add style-labels
# ./subword.sh 

which nvcc

# ./subword-nmt/learn_bpe.py -s 8000 < /netscratch/jalota/datasets/europarl-motra/europarl-motra-train.txt > /netscratch/jalota/datasets/europarl-motra/codes.txt
# ./subword-nmt/apply_bpe.py -c /netscratch/jalota/datasets/europarl-motra/codes.txt < /netscratch/jalota/datasets/europarl-motra/europarl-motra-train.txt > /netscratch/jalota/datasets/europarl-motra/bpe/europarl.train.bpe
# ./subword-nmt/apply_bpe.py -c /netscratch/jalota/datasets/europarl-motra/codes.txt < /netscratch/jalota/datasets/europarl-ppd/europarl.valid > /netscratch/jalota/datasets/europarl-motra/bpe/europarl.valid.bpe
# ./subword-nmt/apply_bpe.py -c /netscratch/jalota/datasets/europarl-motra/codes.txt < /netscratch/jalota/datasets/europarl-ppd/europarl.test > /netscratch/jalota/datasets/europarl-motra/bpe/europarl.test.bpe

# cd fb_fsq/fairseq/fairseq_cli/
# python3 preprocess.py \
#     --only-source \
#     --trainpref /netscratch/jalota/datasets/europarl-motra/bpe/europarl.train.bpe \
#     --validpref /netscratch/jalota/datasets/europarl-motra/bpe/europarl.valid.bpe \
#     --testpref /netscratch/jalota/datasets/europarl-motra/bpe/europarl.test.bpe \
#     --destdir /netscratch/jalota/datasets/data-bin/europarl-motra/subword-nmt/europarl \
#     --workers 60

export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA=/netscratch/jalota/datasets/data-bin/subword-nmt/europarl
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:16024
export CUDA_LAUNCH_BLOCKING=1
# cd ..
cd fb_fsq/fairseq/
# torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \

# torchrun --nnodes=$NUM_NODES \
# --nproc_per_node=${NUM_GPUS} --max_restarts=3 --rdzv_id=$JOB_ID \
# --rdzv_backend=c10d --rdzv_endpoint=$HOST_NODE_ADDR:5905 \
torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
train.py $DATA \
--mask 0.3 --tokens-per-sample 512 --fp16 --fp16-init-scale 4 \
--total-num-update 500000 --max-update 500000 --checkpoint-activations \
--warmup-updates 10000 --task denoising --save-interval 1 \
--max-source-positions 512 \
--max-target-positions 512 \
--arch transformer --optimizer adam --lr-scheduler polynomial_decay \
--lr 0.0004 --dropout 0.1 --criterion cross_entropy --max-tokens 16048 \
--weight-decay 0.01 --attention-dropout 0.1 --share-all-embeddings \
--clip-norm 0.1 --skip-invalid-size-inputs-valid-test --log-format json \
--log-interval 1000 --save-interval-updates 5000 --keep-interval-updates 1 \
--update-freq 8 --seed 4 --distributed-world-size $NUM_GPUS \
--keep-best-checkpoints 10 \
--mask-length span-poisson --replace-length 1 --encoder-learned-pos \
--decoder-learned-pos --rotate 0.0 --mask-random 0.1 --save-dir /netscratch/jalota/checkpoints/subword-nmt-bpe-transformer_gpu4_cpu56 \
--permute-sentences 1 --insert 0.0 --poisson-lambda 3.5 \
--dataset-impl mmap --num-workers ${SLURM_CPUS_PER_TASK}

# --bpe subword_nmt --optimizer cpu_adam --cpu-offload --ddp-backend fully_sharded
