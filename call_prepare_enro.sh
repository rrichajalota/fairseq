#!/bin/bash

set -u 
set -e
set -o pipefail


SRC_LANG="ro"
TGT_LANG="en"


INPUT="/raid/data/daga01/WMT2020_models/data/ro-en-test/test20.short" #input prefix, such that the file $INPUT.$SRC_LANG contains source sentences and $INPUT.$TGT_LANG contains the target sentences
OUTPUT="/raid/data/daga01/WMT2020_models/data/ro-en-out/test20.short" #output path to store generated MT
MOSES_DECODER="/raid/bin/mosesdecoder" #path to mosesdecoder installation
BPE_ROOT="/raid/bin/daga01/subword-nmt-master"  #path to subword-nmt installation
BPE="/raid/data/daga01/WMT2020_models/ro-en-models/bpecodes"  #path to BPE model
MODEL_DIR="/raid/data/daga01/WMT2020_models/ro-en-models"  #directory containing the NMT model .pt file as well as the source and target vocabularies.
TMP="/raid/data/daga01/WMT2020_models/tmp_short" #directory for intermediate temporary files
GPU="0"  #if translating with GPU, id of the GPU to use for inference


mkdir -p "$TMP"

#Preprocess the input data:

prepare(){
for LANG in $SRC_LANG $TGT_LANG; do
  perl $MOSES_DECODER/scripts/tokenizer/tokenizer.perl -threads 20 -a -l $LANG < $INPUT.$LANG > $TMP/preprocessed.tok.$LANG
  
  python $BPE_ROOT/apply_bpe.py -c ${BPE} < $TMP/preprocessed.tok.$LANG > $TMP/preprocessed.tok.bpe.$LANG
  done
}


# Binarize the data for faster translation:
preprocess(){
    fairseq-preprocess --srcdict $MODEL_DIR/dict.$SRC_LANG.txt --tgtdict $MODEL_DIR/dict.$TGT_LANG.txt --source-lang ${SRC_LANG} --target-lang ${TGT_LANG} --testpref $TMP/preprocessed.tok.bpe --destdir $TMP/bin --workers 4
}

# Translate
translate(){
    beam=10

    #CUDA_VISIBLE_DEVICES=$GPU python fairseq_cli/generate.py $TMP/bin \
    python fairseq_cli/generate.py $TMP/bin \
    --cpu \
    --path ${MODEL_DIR}/${SRC_LANG}-${TGT_LANG}.pt \
    --batch-size 10 \
    --beam $beam \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --unkpen 5 \
    --results-path $TMP/fairseq.out
    grep ^H $TMP/fairseq.out | cut -d- -f2- | sort -n | cut -f3- > $TMP/mt.out

    Post-process

    sed -r 's/(@@ )| (@@ ?$)//g' < $TMP/mt.out | perl $MOSES_DECODER/scripts/tokenizer/detokenizer.perl -l $TGT_LANG > $OUTPUT
}


#prepare

preprocess

#translate
