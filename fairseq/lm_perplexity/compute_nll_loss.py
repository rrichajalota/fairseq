"""
Measures the perplexity of sentences using a trained language model.
"""
import logging
import math
import os
import sys
from argparse import Namespace
from typing import Iterable, List, Optional

import torch
from omegaconf import DictConfig
from fairseq.data import LMContextWindowDataset, MonolingualDataset
import fairseq
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.models.transformer import TransformerModel
from fairseq.data.data_utils import batch_by_size,filter_by_size
from fairseq.data.iterators import EpochBatchIterator
from fairseq import utils

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("lm_perplexity.compute_nll_loss")

def eval_lm(
    model: fairseq.models.FairseqModel,
    source_dictionary: fairseq.data.Dictionary,
    batch_iterator: Iterable,
    post_process: Optional[str] = None,
    output_word_probs: bool = False,
    output_word_stats: bool = False,
    target_dictionary: Optional[fairseq.data.Dictionary] = None,
    softmax_batch: int = 0,
    remove_bos_token: bool = False,
    device: Optional[torch.device] = None,
):
    """
    Args:
        models (List[~fairseq.models.FairseqModel]): list of models to
            evaluate. Models are essentially `nn.Module` instances, but
            must be compatible with fairseq's `SequenceScorer`.
        source_dictionary (~fairseq.data.Dictionary): dictionary for
            applying any relevant post processing or outputing word
            probs/stats.
        batch_iterator (Iterable): yield batches of data
        post_process (Optional[str]): post-process text by removing BPE,
            letter segmentation, etc. Valid options can be found in
            fairseq.data.utils.post_process, although not all options
            are implemented here.
        output_word_probs (Optional[bool]): output words and their
            predicted log probabilities
        output_word_stats (Optional[bool]): output word statistics such
            as word count and average probability
        target_dictionary (Optional[~fairseq.data.Dictionary]): output
            dictionary (defaults to *source_dictionary*)
        softmax_batch (Optional[bool]): if BxT is more than this, will
            batch the softmax over vocab to this amount of tokens, in
            order to fit into GPU memory
        remove_bos_token (Optional[bool]): if True, confirm that the
            first token is the beginning-of-sentence symbol (according
            to the relevant dictionary) and remove it from the output
        device (Optional[torch.device]): device to use for evaluation
            (defaults to device of first model parameter)
    """
    if target_dictionary is None:
        target_dictionary = source_dictionary
    if device is None:
        device = next(model.parameters()).device #next(models[0].parameters()).device

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(target_dictionary, softmax_batch)

    score_sum = 0.0
    count = 0

    if post_process is not None:
        if post_process in {"subword_nmt", "@@ "}:
            bpe_cont = post_process.rstrip()
            bpe_toks = {
                i
                for i in range(len(source_dictionary))
                if source_dictionary[i].endswith(bpe_cont)
            }
        else:
            raise NotImplementedError(
                f"--post-process={post_process} is not implemented"
            )
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()
    models = [model]

    for sample in batch_iterator:
        if "net_input" not in sample:
            continue

        sample = utils.move_to_cuda(sample, device=device)

        gen_timer.start()
        hypos = scorer.generate(models, sample)
        # logger.info(f"hypos: {hypos}")
        gen_timer.stop(sample["ntokens"])

        for i, hypos_i in enumerate(hypos):
            # logger.info(f"hypos_i: {hypos_i}")
            hypo = hypos_i[0]
            sample_id = sample["id"][i]
            # logger.info(f"hypo: {hypo}")
            tokens = hypo["tokens"]
            tgt_len = tokens.numel()
            pos_scores = hypo["positional_scores"].float()
            # logger.info(f"target_dictionary.bos(): {target_dictionary.bos()}")
            # logger.info(f"remove_bos_token: {remove_bos_token}")

            if torch.any(hypo["positional_scores"].isnan()):
                continue

            if remove_bos_token:
                assert hypo["tokens"][0].item() == target_dictionary.bos()
                tokens = tokens[1:]
                pos_scores = pos_scores[1:]

            skipped_toks = 0
            if bpe_toks is not None:
                for i in range(tgt_len - 1):
                    if tokens[i].item() in bpe_toks:
                        skipped_toks += 1
                        pos_scores[i + 1] += pos_scores[i]
                        pos_scores[i] = 0

            inf_scores = pos_scores.eq(float("inf")) | pos_scores.eq(float("-inf"))
            if inf_scores.any():
                # logger.info(
                #     "skipping tokens with inf scores:",
                #     target_dictionary.string(tokens[inf_scores.nonzero()]),
                # )
                pos_scores = pos_scores[(~inf_scores).nonzero()]
            score_sum += pos_scores.sum().cpu()
            count += pos_scores.numel() - skipped_toks

            if output_word_probs or output_word_stats:
                w = ""
                word_prob = []
                is_bpe = False
                for i in range(len(tokens)):
                    w_ind = tokens[i].item()
                    w += source_dictionary[w_ind]
                    if bpe_toks is not None and w_ind in bpe_toks:
                        w = w[:-bpe_len]
                        is_bpe = True
                    else:
                        word_prob.append((w, pos_scores[i].item()))

                        next_prob = None
                        ind = i + 1
                        while ind < len(tokens):
                            if pos_scores[ind].item() != 0:
                                next_prob = pos_scores[ind]
                                break
                            ind += 1

                        word_stats.setdefault(w, WordStat(w, is_bpe)).add(
                            pos_scores[i].item(), next_prob
                        )
                        is_bpe = False
                        w = ""
                if output_word_probs:
                    logger.info(
                        str(int(sample_id))
                        + " "
                        + (
                            "\t".join(
                                "{} [{:2f}]".format(x[0], x[1]) for x in word_prob
                            )
                        )
                    )

    avg_nll_loss = (
        -score_sum / count / math.log(2) if count > 0 else 0
    )  # convert to base 2
    # logger.info(
    #     "Evaluated {:,} tokens in {:.1f}s ({:.2f} tokens/s)".format(
    #         gen_timer.n, gen_timer.sum, 1.0 / gen_timer.avg if gen_timer.avg > 0 else 0
    #     )
    # )

    if output_word_stats:
        for ws in sorted(word_stats.values(), key=lambda x: x.count, reverse=True):
            logger.info(ws)

    return {
        "loss": avg_nll_loss,
        "perplexity": 2**avg_nll_loss,
    }


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """increments counters for the sum of log probs of current word and next
        word (given context ending at current word). Since the next word might be at the end of the example,
        or it might be not counted because it is not an ending subword unit,
        also keeps track of how many of those we have seen"""
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return "{}\t{}\t{}\t{}\t{}\t{}".format(
            self.word,
            self.count,
            self.log_prob,
            self.is_bpe,
            self.next_word_prob,
            self.count - self.missing_next_words,
        )

class LanguageModelValidation:
    def __init__(
            self,
            path='/netscratch/jalota/checkpoints/transformer_lm_en_finetuned/',
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='/netscratch/jalota/datasets/motra-sst/ppd_w_europarl-motra-10k_no_dups/en_es_de/unsup_setup/lm_finetune/',
            device=None,
            tgt_dict=None,
            context_window=5,
            tokens_per_sample=512

    ):
        self.context_window = context_window
        self.tokens_per_sample = tokens_per_sample
        self.tgt_dict = tgt_dict
        if context_window > 0:
        # reduce tokens per sample by the required context window size
            tokens_per_sample -= context_window
        
        # Load ensemble
        obj = TransformerModel.from_pretrained(
                path,
                checkpoint_file,
                data_name_or_path,
                )
        self.model = obj.models[0]

        self.model.half() ## use fp16
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device
        self.model.to(device)

        # logger.info(
        #     "num. model params: {:,}".format(sum(p.numel() for p in self.model.parameters()))
        # )

    def eval_lm_dataloader(self,
            dataset,
            max_tokens: Optional[int] = 36000,
            batch_size: Optional[int] = None,
            max_positions: Optional[int] = None,
            num_shards: int = 1,
            shard_id: int = 0,
            num_workers: int = 0,
            data_buffer_size: int = 10,
            # ensures that every evaluated token has access to a context of at least
            # this size, if possible
            context_window: int = 0,
        ):
            
            # logger.info(f"len(dataset): {len(dataset)}")

            if context_window > 0:
                dataset = LMContextWindowDataset(
                    dataset=dataset,
                    tokens_per_sample=self.tokens_per_sample,
                    context_window=self.context_window,
                    pad_idx=self.tgt_dict.pad(),
                )
            
            # logger.info(f"len(LMdataset): {len(dataset)}")

            indices = dataset.ordered_indices()
            # logger.info(f"indices: {indices}")
            
            indices = filter_by_size(indices, dataset, max_positions=512, raise_exception=False)
            
            batch_sampler = batch_by_size(indices, dataset.num_tokens, max_sentences=80, required_batch_size_multiple=8)

            itrs = EpochBatchIterator(dataset=dataset, collate_fn=dataset.collater, batch_sampler=batch_sampler, seed=23, epoch=0, num_workers=0)
            return itrs.next_epoch_itr(shuffle=False)

            # return self.get_batch_iterator(
            #     dataset=dataset,
            #     max_tokens=max_tokens,
            #     max_sentences=batch_size,
            #     max_positions=max_positions,
            #     ignore_invalid_inputs=True,
            #     num_shards=num_shards,
            #     shard_id=shard_id,
            #     num_workers=num_workers,
            #     data_buffer_size=data_buffer_size,
            # ).next_epoch_itr(shuffle=False)

    def get_lm_perplexity(self, dataset, batch_size):

        dataset = dataset
        batch_size = batch_size
        # Load dataset splits
    
        itr = self.eval_lm_dataloader(
            dataset=dataset,
            max_tokens=36000,
            batch_size=batch_size,
            max_positions=utils.resolve_max_positions(self.model.max_positions()
            ),
            context_window=self.context_window,
        )
        # *[model.max_positions() for model in models]

        itr = progress_bar.progress_bar(
            itr, log_format='json',
            log_interval=100,
            default_log_format='tqdm',
        )

        results = eval_lm(
            model=self.model,
            source_dictionary=self.tgt_dict,
            batch_iterator=itr,
            target_dictionary=self.tgt_dict,
        )

        # logger.info(
        #     "Loss (base 2): {:.4f}, Perplexity: {:.2f}".format(
        #         results["loss"], results["perplexity"]
        #     )
        # )

        return results