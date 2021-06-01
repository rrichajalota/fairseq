#!/bin/bash

set -u 
set -e
set -o pipefail

#top="/raid/data/daga01/fairseq_train"
#outdir="/raid/data/daga01/fairseq_out"
top="/raid/data/daga01"


call_basic_model_toytest() {
model="${top}/checkpoints/basic-transf/checkpoint_best.pt"
testdir="${top}/data/data-bin-32k-red-lazy-new"
#testdir="${top}/data/data-bin-32k-red-lazy-new-shorter-minitest-5st"
#python generate.py $testdir --cpu --path $model --batch-size 8 --beam 10 --dataset-impl lazy --sacrebleu --remove-bpe sentencepiece
CUDA_VISIBLE_DEVICES=2 python generate.py $testdir --path $model --batch-size 128 --beam 10 --dataset-impl lazy --sacrebleu --remove-bpe sentencepiece --nbest 10 
}


call_basic_model() {
model="${top}/checkpoints/basic-transf/checkpoint_best.pt"
#testdir="${top}/data/data-bin-32k-red-lazy-new-2"
testdir="${top}/data/data-bin-32k-red-lazy-new-short"
fairseq-generate $testdir --cpu --path $model --beam 5 --dataset-impl lazy
#--sacrebleu
#python generate.py $testdir --cpu --path $model --batch-size 8 --beam 2 --dataset-impl lazy
}


fairseq_test_mycode(){
src="$1"
tgt="$2"
#testdir="/raid/data/daga01/fairseq_train/data/data-bin-32k-red-lazy-new-shorter-minitest-5st"
#testdir="/raid/data/daga01/fairseq_train/data/data-bin-32k-red-lazy-new-renamed"
testdir="/raid/data/daga01/fairseq_train/data/data-bin-32k-red-lazy-new-${src}-${tgt}"
modeldir="/raid/data/daga01/fairseq_train/checkpoints/basic-transf/checkpoint_best.pt"
resdir="/raid/data/daga01/fairseq_out_cmpm2m100_tagged"
mkdir -p $resdir
resdata="${resdir}/generated-mymodel-${src}-${tgt}"


CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-generate $testdir \
    --path $modeldir \
    --batch-size 128 \
    --beam 5   \
    --nbest 1 \
    --remove-bpe 'sentencepiece' \
    --sacrebleu \
    --tokenizer moses -s $src -t $tgt \
    --dataset-impl lazy \
    --results-path $resdata

    #--print-alignment \
    #--beam 10 --nbest 10 
    #--batch-size 2 \ bei m2_100
    #--path "${modeldir}/12b_last_chk_6_gpus.pt" \
    #--fixed-dictionary "${modeldir}/model_dict.128k.txt" \
}



################## M2M_100 ##################################################
#datadir="${top}/data_yoruba"
datadir="${top}/data/mtWMT14/my_test"
spmdir="${top}/my_m2m_out/m2m_spm128k_model"
bindir="${top}/my_m2m_out/binarized"
outdir="${top}/my_m2m_out"

modeldir="${top}/m2m_100"
filename="newstest2014"

########## SPM
spm_m2m(){
src="$1"
tgt="$2"
mkdir -p $spmdir
for lang in $src $tgt ; do
    python scripts/spm_encode.py \
        --model "${modeldir}/spm.128k.model" \
        --output_format=piece \
        --inputs="$datadir/${filename}.${lang}" \
        --outputs="$spmdir/${filename}.${lang}"
done

}

############## BINARIZE
binarize_m2m(){
src="$1"
tgt="$2"
dir_out="${bindir}/${src}-${tgt}"
mkdir -p "$dir_out"
fairseq-preprocess \
    --source-lang "$src" --target-lang "$tgt" \
    --testpref ${spmdir}/${filename} \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir "${dir_out}" \
    --srcdict $modeldir/data_dict.128k.txt --tgtdict $modeldir/data_dict.128k.txt
}


############### GENERATE
call_huge_m2m(){
src="$1"
tgt="$2"


top="/raid/data/daga01"
name="my_m2m_out_scalarmean_cosine"
bindir="${top}/${name}/binarized"
outdir="${top}/${name}"

mkdir -p "$outdir"

testdir="${bindir}/${src}-${tgt}"
outfile="${outdir}/generated-${src}-${tgt}"

#--print-alignment \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-generate $testdir \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python fairseq_cli/generate.py $testdir \
    --batch-size 1 \
    --path "${modeldir}/12b_last_chk_8_gpus.pt" \
    --fixed-dictionary "${modeldir}/model_dict.128k.txt" \
    -s "$src" -t "$tgt" \
    --remove-bpe 'sentencepiece' \
    --beam 50 --nbest 50 \
    --task translation_multi_simple_epoch \
    --lang-pairs $modeldir/language_pairs.txt \
    --decoder-langtok --encoder-langtok src \
    --gen-subset test \
    --dataset-impl mmap \
    --distributed-world-size 1 --distributed-no-spawn \
    --pipeline-model-parallel \
    --pipeline-chunks 1 \
    --model-overrides '{"ddp_backend": "c10d", "pipeline_balance": "1, 6, 6, 6, 8, 6, 6, 6, 6, 1" , "pipeline_devices": "0, 4, 5, 1, 0, 2, 6, 7, 3, 0" }' \
    --pipeline-encoder-balance '[1,6,6,6,7]' \
    --pipeline-encoder-devices '[0,4,5,1,0]' \
    --pipeline-decoder-balance '[1,6,6,6,6,1]' \
    --pipeline-decoder-devices '[0,2,6,7,3,0]' \
    --skip-invalid-size-inputs-valid-test > "$outfile"


### Config for 6 GPUs
    #--path "${modeldir}/12b_last_chk_6_gpus.pt" \
    #--model-overrides '{"ddp_backend": "c10d", "pipeline_balance": "1, 9, 9, 10, 7, 7, 8, 1" , "pipeline_devices": "0, 1, 2, 0, 3 , 4, 5, 0" }' \
    #--pipeline-encoder-balance '[1,9,9,7]' \
    #--pipeline-encoder-devices '[0,1,2,0]' \
    #--pipeline-decoder-balance '[3,7,7,8,1]' \
    #--pipeline-decoder-devices '[0,3,4,5,0]' \
}



#################################### DO ##########################
#spm_m2m "en" "de"
#binarize_m2m "en" "de"
#binarize_m2m "de" "en"

call_huge_m2m "en" "de"
#call_huge_m2m "de" "en"

#fairseq_test_mycode "en" "de"
#fairseq_test_mycode "de" "en"


