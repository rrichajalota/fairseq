<p align="center">
  <img src="fairseq_logo.png" width="150">
  <br />
  <br />
  <a href="https://github.com/pytorch/fairseq/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
  <a href="https://fairseq.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/fairseq/badge/?version=latest" /></a>
</p>

# Self-Supervised Neural Machine Translation
--------------------------------------------------------------------------------
This is the code used for the paper *Self-Supervised Neural Machine Translation*, which describes a joint parallel data extraction and NMT training approach. It is based on a May 2019 copy of the [Fairseq-py](https://github.com/pytorch/fairseq) repository. Be aware that it is therefore not up-to-date with current changes in the original Fairseq(-py) code.

# Requirements and Installation

All the requirements are listed in `environment.yml` and can be installed using `conda env create -f environment.yml`

* [PyTorch](http://pytorch.org/) version >= 1.13.1
* The code has been tested on Python version = 3.8.16
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library with the `--cuda_ext` and `--deprecated_fused_adam` options


## Instructions

### Data Preparation

1. Extract original and translated data from [here](https://zenodo.org/record/5596238#.Y2ObSezMJEJ). 

2. Preprocess (e.g. using [Moses scripts](https://github.com/moses-smt/mosesdecoder/tree/master/scripts)) and apply BPE encoding.

### On a high-level, to run SSNMT with pretraining:
    - **Pretraining**
        - Tokenise and preprocess the data (Europarl)
        - combine preprocessed Europarl training data with preprocessed MOTRA training data (or not)
        - apply BPE - 10k merge operations (~10.3k vocab) for EN-ALL and 11k for DE-ALL. 
        - apply fairseq-preprocess to binarise the data
        - run fairseq-train for bart-style pre-taining
    - **SSNMT**
        - Tokenise and preprocess the data (MOTRA)
        - apply the learned BPE codes from pretraining on MOTRA train-test-dev
        - binarise the data using fairseq-preprocess
        - load the pretrained model checkpoint and finetune over `translation_from_pretrained_bart` task 

    Note that, the tokenization and BPE [Byte-Pair Encoding](https://github.com/rsennrich/subword-nmt) (BPE) should remain consistent for the data used for DAE pretraining, LM training and finetuning and for the Style-Transfer model. 

### Preprocessing Data for Style Transfer
An example for preprocessing the data before training
```
cd fairseq_cli
python3 preprocess.py --destdir /netscratch/anonymous/datasets/motra-sst/ppd_w_europarl-motra-10k_no_dups/en_es_de/unsup_setup/ \
  --source-lang tr \
  --target-lang og \
  --trainpref /netscratch/anonymous/datasets/motra-preprocessed/en_es_de/train/bpe --validpref /netscratch/anonymous/datasets/motra-preprocessed/en_de/dev/europarl-motra-10k-no-dups/bpe \
  --testpref /netscratch/anonymous/datasets/motra-preprocessed/en_de/test/europarl-motra-10k-no-dups/bpe \
  --srcdict  /netscratch/anonymous/datasets/data-bin/europarl-motra/subword-nmt-10k/europarl/dict.txt \
  --tgtdict /netscratch/anonymous/datasets/data-bin/europarl-motra/subword-nmt-10k/europarl/dict.txt \
  --dataset-impl raw \
  --workers 60
```
### Train
Use `traincomp.py` to train the system. An example run on how to train SSNMT for Translationese-to-Original Style Transfer using Joint Training is shown below.

```
python3 traincomp.py /netscratch/anonymous/datasets/motra-sst/ppd_w_europarl-motra-10k_no_dups/en_es_de/unsup_setup/ \
  --arch transformer \
  --share-all-embeddings --checkpoint-activations \
  --share-decoder-input-output-embed \
  --encoder-embed-dim 512 \
  --decoder-embed-dim 512 \
  --task translation_from_pretrained_bart --langs "<tr>, <og>" \
  --update-freq 2 \
  --lr 0.0003 \
  --criterion unsupervised_augmented_label_smoothed_cross_entropy \
  --label-smoothing 0.1 --start-unsup 1200 \
  --dropout 0.2 \
  --weight-decay 0.0001 \
  --optimizer adam \
  --adam-betas '(0.9, 0.9995)' \
  --clip-norm 0.0 \
  --write-dual \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 2000 \
  --dataset-impl raw \
  --decoder-learned-pos --encoder-learned-pos \
  --max-sentences 160 --retrieval intersect \
  --max-source-positions 512 \
  --max-target-positions 512 \
  --skip-invalid-size-inputs-valid-test \
  --max-epoch 30 --keep-best-checkpoints 10 --patience 10 \
  --comp-epochs 30 --save-interval-updates 2 \
  --comparable --margin ratio \
  --verbose --faiss --index ivf \
  --representations dual --faiss-output /netscratch/anonymous/logs/from_en_all_bs@en_newLoss_htr.txt \
  --no-base --comp-log /netscratch/anonymous/logs/en_all_bs@en_newLoss_htr/ \
  --comparable-data Comparable/tr_og.list \
  --sim-measure margin --save-dir /netscratch/anonymous/checkpoints/sst/en_all_bs_no_th@newLoss_htr/ \
  --finetune-from-model /netscratch/anonymous/checkpoints/subword-nmt-10k-bpe-transformer_gpu4_cpu50/checkpoint_best.pt \
  --threshold 0 \
  --wandb-project motra_no_thresh_unsup_en \
  --num-workers 0 \
  --log-format json |tee /netscratch/anonymous/logs/train_en_all_bs@en_newLoss_htr.log

```

Run `train.py -h` for more information.

### Evaluation 

All the scripts for evaluation can be found under the `evaluation/` folder 

1. Get the style-transferred outputs a) once with --remove-bpe and b) once without (to compute average perplexities using Fairseq's pretrained TransformerLM).
```
python3 generate.py /netscratch/anonymous/datasets/data-bin/europarl-motra/subword-nmt-10k/europarl/test_bal/ \
--task translation \
--path <path-to-checkpoint>/checkpoint_best.pt \
--results-path <path> \
--beam 5 --source-lang tr --target-lang og --dataset-impl raw 
```
2. Generate intermediate data files.
```
python evaluation/gen_test_data.py --file /netscratch/anonymous/results/generations/unsup/motra-old/712551/generate-test.txt --out_dir /netscratch/anonymous/datasets/motra-preprocessed/en_de/test/unsup-generated/ --name pred_712551.tsv

# combine og file with pred file
cat /netscratch/anonymous/datasets/motra-preprocessed/en_de/test/og.tsv /netscratch/anonymous/datasets/motra-preprocessed/en_de/test/unsup-generated/pred_712551.tsv > /netscratch/anonymous/datasets/motra-preprocessed/en_de/test/gen_tsvs/gen_712551.tsv

# shuffle the test file
shuf -o /netscratch/anonymous/datasets/motra-preprocessed/en_de/test/gen_tsvs/gen_712551.tsv < /netscratch/anonymous/datasets/motra-preprocessed/en_de/test/gen_tsvs/gen_712551.tsv

python evaluation/extract_ref_hyp.py --file /netscratch/anonymous/results/generations/unsup/motra-old/712551/generate-test.txt --name 712551.tsv

python new/fairseq/evaluation/gen_fsq_ppl_data.py --file /netscratch/anonymous/results/generations/unsup/motra-old/712551_ppl/generate-test.txt --out_dir /netscratch/anonymous/test_perplexity/ --exp 712551
```

3. Evaluate LM perplexity

Note: copy dict.txt from the preprocessed FAIRSEQ_DATA to  

```
python3 eval_lm.py /netscratch/anonymous/test_perplexity/712551/ --path /netscratch/anonymous/checkpoints/transformer_lm_en_finetuned/checkpoint_best.pt --quiet --output-word-stats --gen-subset test --max-sentences 500 --skip-invalid-size-inputs-valid-test --dataset-impl raw --fp16 --sample-break-mode eos --context-window 50
```

4. Meausre BERT-Score
```
python3 evaluation/compute_bertscore.py --file /netscratch/anonymous/datasets/motra-preprocessed/en_de/test/src_hyp/712551.tsv --model roberta-base
```

5. Run Translationese Classifier
```
python3 evaluation/binary_classification.py --model /netscratch/anonymous/checkpoints/binaryClassification_balanced/ --test /netscratch/anonymous/datasets/motra-preprocessed/en_de/test/gen_tsvs/gen_712551.tsv
```

6. Run Qualitative Analysis
```
python3 evaluation/qualitative_analysis.py --file /netscratch/anonymous/datasets/motra-preprocessed/en_de/test/src_hyp/712551.tsv
```




 
![Model](fairseq.gif)
