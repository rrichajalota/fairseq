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


## Installation

1. Clone this repository and make sure you have [Anaconda Python](https://www.anaconda.com/distribution/) installed.

2. Create a virtual environment with Python 3.6 and activate

3. Install the following packages...
   * [Pytorch](https://pytorch.org/) with CUDA support (if available)
   * Configargparse
   
## Instructions

In order to train the system without any paralell data, you will need plenty of comparable data as well as pre-trained word embeddings. It is vital that the comparable data used for extraction uses the same [Byte-Pair Encoding](https://github.com/rsennrich/subword-nmt) (BPE) as the data that the word embeddings were trained on.

### Training unsupervised embeddings

1. Accumulate large amounts of monolingual data in your L1 and L2 languages.

2. Preprocess (e.g. using [Moses scripts](https://github.com/moses-smt/mosesdecoder/tree/master/scripts)) and apply BPE encoding.

3. Train [Word2Vec](https://github.com/tmikolov/word2vec) monolingual embeddings for L1 as well as for L2 data.

4. Map monolingual embeddings into common space using [Vecmap](https://github.com/tmikolov/word2vec) and a seed dictionary of numerals.

5. Once the embeddings are mapped into a common space, merge the source and target embeddings into a single file (you'll need to write a small script for doing that).

### Prepare comparable corpus

1. Collect a comparable corpus, ensuring that an article in L1 talks about the same topic as the corresponding article in L2.

2. Preprocess data, such that the final BPE encoding is the same as the one used for the monolingual data that the embeddings were trained on.

3. Now, you should have a directory containing all your pre-processed comparable documents. For the system to know which ones are related to each other, create a `list-file`. In each line, it contains the absolute path to a document in L1 which should be mapped to a document in L2. Use a tab between the L1 and L2 document. As such:

4. Also, create a concatenated version of all the comparable documents. We will call these two files `corpus.L1` and `corpus.L2` for now. These are only needed to create the OpenNMT-py format vocabulary files.

### Create OpenNMT corpus files

1. Run preprocess.py on the `corpus.L1/L2` files. Do not forget to set the vocabulary size high enough to cover all words in the vocabulary. Also, it is *vital* to the system to have a shared vocabulary, so do not forget to set the -share_vocab tag. A sample run command can be found under `comparableNMT/run/examples/run_preproc.sh`

2. Convert embeddings into OpenNMT format using embeddings_to_torch.py in the main directory. A sample run can be found under `comparableNMT/run/examples/run_embeds.sh`

### Train

Use train.py to train the system. Note that not all options available for training in OpenNMT-py are compatible with this code. For example, training on more than one GPU is currently not possible and batch sizes need to be given in number of sentences and not tokens. The examples in `comparableNMT/run/examples/run_train_XXXX.sh` show how to train the LSTM or Transformer models described in the paper. For best performance, look at `comparableNMT/run/examples/run_train_tf_margP.sh`

#### Comparable training options

```
--comparable: use joint parallel data extraction and training (mandatory)
--comparable_data: path to list-file (mandatory)
--comp_log path to where extraction logs will be written (mandatory)
--comp_epochs: number of epochs to pass over the comparable data (mandatory)
--no_base: do not use any parallel data to pre-train your model (recommended)
--fast: reduces the search space to the first batch in each document (substantially faster)
--threshold: supply a similarity score that needs to be passed by candidates (not necessary if you are using dual representations)
--second: use medium permissibility (margR)
--sim_measure: similarity measure used (default: margin-based)
--k: number of k-nearest neighbors to use when scoring (default: 4)
--cove_type: the type of sentence representation creation to use (default: sum)
--representations: dual uses both representation types, while hidden-only and embed-only use C_h and C_e respectively. In case of not using dual, it is recommended to set a threshold. (default: dual)
--max_len: maximum length of sequences to train on
--write_dual: this will also write logs for those sentence pairs accepted by one of the two representations only (can be used with dual representations)
--no_swaps: do not perform random swaps in the src-tgt direction during training (not recommended)
--comp_example_limit: supply a maximum number of unique pairs to extract (not recommended)
```

Run `train.py -h` for more information.

   
### What's New:

- November 2019: [VizSeq released (a visual analysis toolkit for evaluating fairseq models)](https://facebookresearch.github.io/vizseq/docs/getting_started/fairseq_example)
- November 2019: [CamemBERT model and code released](examples/camembert/README.md)
- November 2019: [BART model and code released](examples/bart/README.md)
- November 2019: [XLM-R models and code released](examples/xlmr/README.md)
- September 2019: [Nonautoregressive translation code released](examples/nonautoregressive_translation/README.md)
- August 2019: [WMT'19 models released](examples/wmt19/README.md)
- July 2019: fairseq relicensed under MIT license
- July 2019: [RoBERTa models and code released](examples/roberta/README.md)
- June 2019: [wav2vec models and code released](examples/wav2vec/README.md)

### Features:

Fairseq provides reference implementations of various sequence-to-sequence models, including:
- **Convolutional Neural Networks (CNN)**
  - [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/conv_lm/README.md)
  - [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
  - [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
  - [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
  - [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
- **LightConv and DynamicConv models**
  - [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
- **Long Short-Term Memory (LSTM) networks**
  - Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)
- **Transformer (self-attention) networks**
  - Attention Is All You Need (Vaswani et al., 2017)
  - [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
  - [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
  - [Adaptive Input Representations for Neural Language Modeling (Baevski and Auli, 2018)](examples/language_model/transformer_lm/README.md)
  - [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
  - [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
  - [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
  - [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md )
- **Non-autoregressive Transformers**
  - Non-Autoregressive Neural Machine Translation (Gu et al., 2017)
  - Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement (Lee et al. 2018)
  - Insertion Transformer: Flexible Sequence Generation via Insertion Operations (Stern et al. 2019)
  - Mask-Predict: Parallel Decoding of Conditional Masked Language Models (Ghazvininejad et al., 2019)
  - [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)


**Additionally:**
- multi-GPU (distributed) training on one machine or across multiple machines
- fast generation on both CPU and GPU with multiple search algorithms implemented:
  - beam search
  - Diverse Beam Search ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424))
  - sampling (unconstrained, top-k and top-p/nucleus)
- large mini-batch training even on a single GPU via delayed updates
- mixed precision training (trains faster with less GPU memory on [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores))
- extensible: easily register new models, criterions, tasks, optimizers and learning rate schedulers

We also provide [pre-trained models for translation and language modeling](#pre-trained-models-and-examples)
with a convenient `torch.hub` interface:
```python
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
en2de.translate('Hello world', beam=5)
# 'Hallo Welt'
```
See the PyTorch Hub tutorials for [translation](https://pytorch.org/hub/pytorch_fairseq_translation/)
and [RoBERTa](https://pytorch.org/hub/pytorch_fairseq_roberta/) for more examples.

![Model](fairseq.gif)

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library with the `--cuda_ext` and `--deprecated_fused_adam` options

To install fairseq from source and develop locally:
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable .
```

