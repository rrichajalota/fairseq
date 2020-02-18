#!/bin/bash



top="/raid/data/daga01/fairseq_train"

call_big_model_toytest() {
model="${top}/checkpoints/snapshot/checkpoint38.pt"
testdir="${top}/mini-test-bin"
python generate.py $testdir --cpu --path $model --batch-size 8 --beam 2 --dataset-impl lazy
}




call_toy_model_toytest(){
model="${top}/checkpoints/toyset/checkpoint_best.pt"
testdir="${top}/data-bin-toyset"

python generate.py $testdir --cpu --path $model --batch-size 8 --beam 2 --dataset-impl raw
}



call_big_model_toytest
