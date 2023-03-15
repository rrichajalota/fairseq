#!/bin/bash

#SBATCH --nodes=1               # Number of nodes or servers. See: http://koeln.kl.dfki.de:3000/d/slurm-resources/resources?orgId=1&refresh=15s
#SBATCH --ntasks-per-node=1     # Number of task in each node, we want 1 
#SBATCH --cpus-per-task=48     # We want 4 cores for this job.
#SBATCH --mem-per-cpu=8gb     # each core to have 16 Gb RAM
#SBATCH --gres=gpu:4            # We want 4 GPUs in each node for this job.
#SBATCH --time=10-0:00         # Run this task no longer than 10 days. 
#SBATCH --job-name=v2_pretrain_bart_thrd1_cpu2
#SBATCH --partition=V100-32GB,RTXA6000,RTX3090,A100-40GB,A100-80GB,A100-PCI,V100-16GB,RTX2080Ti
#SBATCH --output=logs/v2_pretrain_transformer_gpu1_thrd1_cpu12%A.logs

echo "#############################"
date
echo "Current dir: " ${SLURM_SUBMIT_DIR}
echo "Hostname: `hostname`"

# Print the task details.
echo "Job ID: ${SLURM_JOBID}"
echo "SLURM array task ID:  ${SLURM_ARRAY_TASK_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}" 
echo "Cluster name: ${SLURM_CLUSTER_NAME}"
echo "Partition name: ${SLURM_JOB_PARTITION}" 
echo "num nodes: ${SLURM_JOB_NUM_NODES}"
echo "Using: `which python`"
echo -e "#############################\n"

NGPUS=4; # number of gpu
NCPUS_PER_TASK=48; # number of cpu per task
# MEM=50000 # memory increase this if needed

srun -v \
--container-mounts=/netscratch/jalota:/netscratch/jalota,/ds:/ds:ro,"`pwd`":"`pwd`",/home/jalota/:/home/jalota/ \
--container-image=/netscratch/$USER/containers/cuda:11.7.0-devel-ubuntu22.04.sqsh \
--container-workdir="`pwd`" --no-container-remap-root --gpu-bind=none \
--gpus=$NGPUS \
bash {1gpu|multi-gpu}_bart_run_pretrain.sh

#--cpus-per-task=$NCPUS_PER_TASK
# /netscratch/$USER/containers/cuda:11.7.1-devel-ubuntu22.04
# /netscratch/$USER/containers/cuda:11.7.0-devel-ubuntu18.04.sqsh
# --container-save=/netscratch/$USER/containers/jalota_pytorch22.12.sqsh \ 
# /netscratch/enroot/nvcr.io_nvidia_pytorch_22.12-py3.sqsh -n1 --mem=$MEM
#  RTXA6000,V100-32GB,RTX3090  # Run this only in these mentioned GPUs. If you don't have any choice over GPUs, remove this parameter.