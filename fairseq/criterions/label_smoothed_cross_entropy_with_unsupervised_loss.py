"""
Classes and methods used for multi-task tuning on Supervised style-transfer and unsupervised LM and cosine similarity losses. 
"""

import torch.nn.functional as F
import torch.nn as nn
from fairseq import criterions
from dataclasses import dataclass, field
import torch
import numpy as np
from torch.nn import MSELoss, CosineSimilarity
import math
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from torch.nn.functional import gumbel_softmax
from torch.distributions import Categorical
from torch.utils.data import Dataset
from fairseq.data import LMContextWindowDataset, MonolingualDataset
import evaluate
import random
from fairseq.lm_perplexity import LanguageModel, LanguageModelValidation

from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
import logging, os, sys
from fairseq.data.data_utils import collate_tokens
from evaluate import load
from fairseq.scoring.perplexity import Perplexity

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq.criterion.UnsupervisedAugmentedCrossEntropyLoss")


class torchDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

def cross_entropy(pred, soft_targets):
    # logger.info(f"pred.size(): {pred.size()}")
    # logger.info(f"soft_targets.size(): {soft_targets.size()}")
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@dataclass
class UnsupervisedAugmentedLabelSmoothedCrossEntropyCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    lm_weight: float = field(
        default=0.5,
        metadata={"help": "weight fot per-word LM entropy."},
    )
    cosine_weight: float = field(
        default=0.5,
        metadata={"help": "weight for cosine similarity loss."},
    )
    unsupervised_weight: str = field(
        default=0.5,
        metadata={"help": "unsupervised loss weightage"},
    )
    supervised_weight: str = field(
        default=0.5,
        metadata={"help": "supervised loss weightage"},
    )
    pretrained_lm: str = field(
        default="/netscratch/jalota/checkpoints/transformer_en_hansard/",
        metadata={
            "help": "pretrained fairseq LM model to evaluate PPL during unsupervised training."
        },
    )
    pretrained_lm_dict_path: str = field(
        default="/netscratch/jalota/datasets/data-bin/canadianHansard/lm/",
        metadata={
            "help": "dict path for pretrained fairseq LM model to evaluate PPL during unsupervised training."
        },
    )
    lm_context_window: int = field(
        default=5, metadata={"help": "context window size for evaluating PPL"}
    )
    bertscore_model: str = field(
        default="roberta-base",
        metadata={
            "help": "which model to use for evaluating semantic similarity. for EN: roberta-base, DE: t5-base"
        },
    )


@register_criterion(
    "unsupervised_augmented_label_smoothed_cross_entropy",
    dataclass=UnsupervisedAugmentedLabelSmoothedCrossEntropyCriterionConfig,
)
class UnsupervisedAugmentedLabelSmoothedCrossEntropyCriterion(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
        lm_weight=1,
        cosine_weight=1,
        unsupervised_weight=1,
        supervised_weight=1,
        bertscore_model='roberta-base',
        lm_context_window=5,
        pretrained_lm_dict_path="/netscratch/jalota/datasets/data-bin/canadianHansard/lm/",
        pretrained_lm="/netscratch/jalota/checkpoints/transformer_en_hansard/",
        tau_gumbel_softmax=0.1,
        hard_gumbel_softmax=False,
        eps_gumbel_softmax=1e-10,
        soft_bert_score=False
    ):
        # 'microsoft/deberta-v3-base' t5-base
        # roberta-base for EN
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.lm_weight = torch.tensor(1)
        self.cosine_weight = torch.tensor(1)
        self.unsupervised_weight = torch.tensor(0.3)
        self.supervised_weight = torch.tensor(0.7)
        self.perplexity = Perplexity()
        self.cosine_sim = CosineSimilarity()
        self.mse_loss = MSELoss(reduction='mean')
        self.bertscore_model = bertscore_model

        self.tau_gumbel_softmax = tau_gumbel_softmax
        self.hard_gumbel_softmax = hard_gumbel_softmax
        self.eps_gumbel_softmax = eps_gumbel_softmax
        self.pretrained_lm = pretrained_lm
        self.pretrained_lm_dict_path = pretrained_lm_dict_path
        self.lm_context_window = lm_context_window
        
        # self.bert_scorer = BERTScorer(self.bert_model, soft_bert_score=soft_bert_score)  # , device='cpu')
        # self.pad_token_id = self.bert_scorer._tokenizer.convert_tokens_to_ids('[PAD]')
        # hansard: /netscratch/jalota/checkpoints/transformer_en_hansard/
        # hansard_data: /netscratch/jalota/datasets/data-bin/canadianHansard/lm/
        # de: /netscratch/jalota/checkpoints/transformer_lm_de_finetuned/
        # de_data: /netscratch/jalota/datasets/motra-sst/de/unsup_setup_raw/lm_finetuning/
        self.bertscore = evaluate.load("bertscore")
        self.lm = LanguageModel(path=self.pretrained_lm,tgt_dict=task.tgt_dict,data_name_or_path=self.pretrained_lm_dict_path)
        self.val_lm = LanguageModelValidation(path=self.pretrained_lm,tgt_dict=task.tgt_dict, context_window=self.lm_context_window,data_name_or_path=self.pretrained_lm_dict_path)
        # /netscratch/jalota/datasets/motra-sst/de/unsup_setup_raw/lm_finetuning/
        # DE: /netscratch/jalota/checkpoints/transformer_lm_de_finetuned/
        # EN: /netscratch/jalota/checkpoints/transformer_lm_en_finetuned/
        # data_name_or_path='/netscratch/jalota/datasets/motra-sst/ppd_w_europarl-motra-10k_no_dups/en_es_de/unsup_setup/lm_finetune/'

        #load("perplexity", module_type="measurement")
        
    def forward(self, model, sample, seqeunce_generator=None, tgt_dict=None,reduce=True, unsup=False, src_dict=None, train=True, only_unsupervised=False):

        logging_output = {}
        loss = 0.0
        sample_size_set = False
        # only_unsupervised = False
        if train and not only_unsupervised:
            net_output = model(**sample["sup"]["net_input"])
            loss_sum, nll_loss_sum = self.compute_loss(model, net_output, sample["sup"], reduce=reduce)
            sample_size = (
                sample['sup']["target"].size(0) if self.sentence_avg else sample['sup']["ntokens"]
            )
            # logger.info(f'sample["sup"]["net_input"]["prev_output_tokens"]: {sample["sup"]["net_input"]["prev_output_tokens"]}')
            ## take the mean of loss and nll_loss here and convert them from log base e to 2
            loss = loss_sum / sample_size / math.log(2)
            nll_loss = nll_loss_sum / sample['sup']["ntokens"] / math.log(2)
            # NOTE:
            # # we don't need to use sample_size/ntokens as denominator for the gradient
            # # here sample_size & ntokens are just used for logging
            sample_size = 1
            sample_size_set = True
            if unsup:
                loss = self.supervised_weight * loss

            logging_output = {
                "loss" : loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": 1,
                "nsentences": sample['sup']["target"].size(0),
                "sample_size": sample_size,
            }
            # sample['sup']["ntokens"]
        if unsup:
            if train:
                if not only_unsupervised:
                    sample = sample['unsup'] 
                # in case of eval, dataset is not RoundRobin, thus 'sample' can be used directly, & is not an OrderedDict!
            
            def decode(toks, src=False, escape_unk=False):
                if src:
                    s = src_dict.string(toks, bpe_symbol="subword_nmt",)
                    return s.replace("<pad>", "").rstrip()
                
                return tgt_dict.string(
                    toks,
                    bpe_symbol="subword_nmt",
                ).replace("<pad>", "").rstrip()
                
            with torch.no_grad():
                if any(sample["net_input"]["src_lengths"]) > 510:
                    logger.info(f'sample["net_input"]["src_lengths"]: {sample["net_input"]["src_lengths"]}')
                gen_out = seqeunce_generator.generate(
                        [model], sample, prefix_tokens=None, constraints=None)

                # logger.info(f"gen_out: {gen_out}")
                hyps, hyps_tok = [], []
                for i in range(len(gen_out)):
                    # s = decode(gen_out[i][0]["tokens"]).strip()
                    # if len(s) > 0:
                    #     hyps_tok.append(s)
                    hyps.append(gen_out[i][0]["tokens"]) 

                msize = max(v.size(0) for v in hyps) 
                msize = msize if msize <= 512 else 512
                # logger.info(f"msize: {msize}")

                hyps = collate_tokens(hyps, src_dict.pad(), src_dict.eos(), move_eos_to_beginning=False, left_pad=False, pad_to_length=512,pad_to_bsz=None)

                batch_size = len(hyps)

            if not train:
                # calculate bertscore and PPL straight-away! 
                refs_list = []
                hyps_tok = []
                refs = sample['net_input']['src_tokens']
                for i in range(len(refs)):
                    s = decode(refs[i]).strip()
                    hs = decode(gen_out[i][0]["tokens"]).strip()
                    if len(s.split()) > 2 and len(hs.split()) > 1:
                        hyps_tok.append(hs)
                        refs_list.append(s)
                        
                    # refs_list.append(s)

                # logger.info(f"len(refs_list): {len(refs_list)}")
                # logger.info(f"len(hyps_tok): {len(hyps_tok)}")

                # logger.info(f"refs_list: {refs_list}")
                # logger.info(f"hyps_tok: {hyps_tok}")

                sim_loss, _ = self.compute_bertLoss(hyps_tok, refs_list) 

                # ppl_results = self.perplexity.compute(data=hyps_tok, model_id='/netscratch/jalota/checkpoints/gpt2-finetuned-motra/', batch_size=len(hyps_tok), add_start_token=True)
                hyps_cpu, gen_sizes = [], []
                for h in hyps:
                    # if h.size(0) <= 512:
                    hyps_cpu.append(h.cpu())
                    gen_sizes.append(msize)

                # hyps = [h.cpu() for h in hyps]
                # logger.info(f"len(hyps_cpu): {len(hyps_cpu)}")
                # logger.info(f"gen_sizes: {gen_sizes}")

                genData = torchDataset(data_list=hyps_cpu)
                # gen_sizes = [msize for _ in range(len(genData))]
                gen_data = MonolingualDataset(genData, gen_sizes, src_vocab=tgt_dict, fixed_pad_length=512)

                ppl_results = self.val_lm.get_lm_perplexity(gen_data, batch_size)

                # logger.info(f"ppl: {ppl_results['mean_perplexity']}")
                # gpt2-finetuned-motra-de-40epochs/ - DE
                # gpt2-finetuned-motra/ - EN

                mean_per_word_entropy = ppl_results['loss']
                # math.log2(ppl_results['mean_perplexity'])

                unsupervised_loss = 1.0 * sim_loss + 1.0 * mean_per_word_entropy
                loss += self.unsupervised_weight * unsupervised_loss 
                logging_output["loss"] = loss
                logging_output["sim_loss"] = sim_loss
                logging_output["mean_per_word_entropy"] = mean_per_word_entropy
                logging_output["lm_ppl"] = ppl_results['perplexity']
                logging_output["unsupervised_loss"] = unsupervised_loss

            else:
                # use the hyps to create prev_output_tokens
                # a shifted version of hyps for feeding the
                # previous output token(s) into the next decoder step
                sample = self.prepare_second_pass_input(sample, tgt_dict, src_dict, hyps)

                # logger.info(f"sample: {sample}")

                # logger.info(f"enable gradient")
                # second-pass through the decoder in training mode
                with torch.enable_grad():
                    net_output = model(**sample['net_input'])
                    lprobs = model.get_normalized_probs(net_output, log_probs=True)
                        
                    gsm_samples = gumbel_softmax(lprobs, tau=self.tau_gumbel_softmax, hard=self.hard_gumbel_softmax,eps=self.eps_gumbel_softmax, dim=-1)

                lm_out = self.lm.get_lm_out_from_decoder_inp(gsm_samples)

                lm_loss = self.compute_loss(model, lm_out, gsm_samples, unsup=unsup, reduce=reduce)

                sample_size = gsm_samples.size()[1]
                ## take the mean of loss and nll_loss here and convert them from log base e to 2
                lm_loss = lm_loss / math.log(2)

                sim_loss = self.get_similarity_loss(model, gsm_samples, sample, src_dict.pad())
            
                unsupervised_loss = 1.0 * sim_loss + 1.0 * lm_loss

                loss += self.unsupervised_weight * unsupervised_loss   
                logging_output["loss"] = loss.data
                logging_output["lm_loss"] = lm_loss.data
                # logging_output["lm_nll_loss"] = lm_nll_loss.data
                logging_output["unsupervised_loss"] = unsupervised_loss.data
                logging_output["sim_loss"] = sim_loss.data
                logging_output["unsup_nsentences"] = 1
        
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if not sample_size_set:
            sample_size= 1

        return loss, sample_size, logging_output
    

    def get_similarity_loss(self, model, preds_tensor, sample, pad_token_id):
        
        emb_matrix = model.encoder.embed_tokens.weight

        # get bert embeddings from tensor 
        batch_size, max_seq_len, vocab_size = preds_tensor.size()
        emb_size = emb_matrix.size()[-1]

        with torch.autocast("cuda"):
            preds_tensor_embs = torch.mm(preds_tensor.contiguous().view(-1, vocab_size), emb_matrix)
            preds_tensor_embs = preds_tensor_embs.view(-1, max_seq_len, emb_size)

            # logger.info(f"preds_tensor_embs: {preds_tensor_embs.dtype}")

            with torch.no_grad():
                source_emb = model.encoder.forward(sample['net_input']['src_tokens'])
                preds_enc_emb = model.encoder.forward(preds_tensor_embs)

        source_sent_repr = torch.sum(source_emb['encoder_out'][0], dim=0)
        output_sent_repr = torch.sum(preds_enc_emb['encoder_out'][0], dim=0)
        target_labels = torch.ones(source_sent_repr.shape[0], dtype=source_sent_repr.dtype).cuda()
        #cosineLoss = torch.nn.CosineEmbeddingLoss(reduction='mean') 
        # cos_sim_loss = cosineLoss(source_sent_repr, output_sent_repr, target_labels)
        cosine_out = self.cosine_sim(source_sent_repr, output_sent_repr)
        # logger.info(f"cosine_out: {cosine_out}")
        # similarity_labels = torch.FloatTensor(np.array([1]*len(source_sent_repr)), dtype=source_sent_repr.dtype).cuda()
        # if similarity_labels is not None:
        similarity_loss = self.mse_loss(cosine_out, target_labels.view(-1))
        # logger.info(f'cos_sim_loss: {cos_sim_loss}')
        # logger.info(f"similarity_loss: {similarity_loss}")
        return similarity_loss
    
    def compute_loss(self, model, net_output, sample, unsup=False, reduce=True):
        if not unsup:
            lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
            loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            )
            return loss, nll_loss
        else:
            lm_out = net_output
            decoder_out = sample
            lm_probs = model.get_normalized_probs(lm_out, log_probs=True)
            if self.ignore_prefix_size > 0:
                # lprobs: B x T x C
                lm_probs = lm_probs[:, self.ignore_prefix_size :, :].contiguous()
                decoder_out = decoder_out[:, self.ignore_prefix_size :].contiguous()
            lprobs = lm_probs
            target = decoder_out # as per the eqn in the paper Yang et. al. 2019
            return cross_entropy(lm_probs, decoder_out)
        

    def prepare_second_pass_input(self, sample, tgt_dict, src_dict, hyps):
        prev_output_tokens = collate_tokens(
                    hyps,
                    tgt_dict.pad(),
                    tgt_dict.eos(),
                    left_pad=False,
                    move_eos_to_beginning=True,
                    pad_to_length=512,
                    pad_to_multiple=1
                )
        # logger.info(f"prev_output_tokens: {prev_output_tokens}")

        # logger.info(f"tgt_dict.eos():{tgt_dict.eos()}")
                
        src_lengths = sample["net_input"]["src_lengths"]
        src_tokens = sample["net_input"]["src_tokens"]
        # logger.info(f"src_lengths: {src_lengths}")
                
        # sort by descending src lengths 
        src_lengths, sort_order = src_lengths.sort(descending=True)
                
        sample['id'] = sample['id'].index_select(0, sort_order)
        sample["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(0, sort_order)
        sample["net_input"]["src_lengths"] = src_lengths
        sample["net_input"]["src_tokens"] = src_tokens.index_select(0, sort_order)
        
        return sample
    
    def compute_bertLoss(self, preds_list, refs_list, reduce=True):
        # logger.info(f"len(refs_list): {len(refs_list)}")
        # logger.info(f"len(preds_list): {len(preds_list)}")
        results = self.bertscore.compute(predictions=preds_list, references=refs_list, model_type=self.bertscore_model)
        avg_f1 = sum(results['f1'])/len(results['f1'])
        bert_loss = 1-avg_f1
        return bert_loss, avg_f1

    def compute_cosineSimilarityLoss(self, model, sample, hyps, train):
        # with torch.no_grad():
        if not train:
            with torch.no_grad():
                source_emb = model.encoder.forward(sample['net_input']['src_tokens'].cuda())
                gen_out_emb = model.encoder.forward(hyps) 

                source_sent_repr = torch.sum(source_emb['encoder_out'][0], dim=0)
        
                output_sent_repr = torch.sum(gen_out_emb['encoder_out'][0], dim=0).cuda()
                target_labels = torch.ones(source_sent_repr.shape[0], dtype=source_sent_repr.dtype).cuda()
                #cosineLoss = torch.nn.CosineEmbeddingLoss(reduction='mean') 
                # cos_sim_loss = cosineLoss(source_sent_repr, output_sent_repr, target_labels)
                cosine_out = self.cosine_sim(source_sent_repr, output_sent_repr)
                # similarity_labels = torch.FloatTensor(np.array([1]*len(source_sent_repr)), dtype=source_sent_repr.dtype).cuda()
                # if similarity_labels is not None:
                similarity_loss = self.mse_loss(cosine_out, target_labels.view(-1))
                # logger.info(f'cos_sim_loss: {cos_sim_loss}')
                return similarity_loss #cos_sim_loss
        else:
            source_emb = model.encoder.forward(sample['net_input']['src_tokens'].cuda())
            gen_out_emb = model.encoder.forward(hyps)

            source_sent_repr = torch.sum(source_emb['encoder_out'][0], dim=0)
            # logger.info(f"source_sent_repr: {source_sent_repr}")
            
            output_sent_repr = torch.sum(gen_out_emb['encoder_out'][0], dim=0).cuda()

            # logger.info(f"output_sent_repr: {output_sent_repr}")
            target_labels = torch.ones(source_sent_repr.shape[0], dtype=source_sent_repr.dtype).cuda()
            # cosineLoss = torch.nn.CosineEmbeddingLoss(reduction='mean') 
            # cos_sim_loss = cosineLoss(source_sent_repr, output_sent_repr, target_labels)
            cosine_out = self.cosine_sim(source_sent_repr, output_sent_repr)
            # similarity_labels = torch.FloatTensor(np.array([1]*len(source_sent_repr)), dtype=source_sent_repr.dtype).cuda()
            similarity_loss = self.mse_loss(cosine_out, target_labels.view(-1))
            
            return similarity_loss #cos_sim_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        # super().reduce_metrics(logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 1) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 1) for log in logging_outputs)
        sim_loss = sum(log.get("sim_loss", 0) for log in logging_outputs)
        mean_per_word_entropy = sum(log.get("mean_per_word_entropy", 0) for log in logging_outputs)
        unsupervised_loss = sum(log.get("unsupervised_loss", 0) for log in logging_outputs)
        unsup_nsentences = sum(log.get("unsup_nsentences", 1) for log in logging_outputs)
        lm_loss = sum(log.get("lm_loss", 0) for log in logging_outputs)
        lm_ppl = sum(log.get("lm_ppl", 0) for log in logging_outputs)
        # lm_nll_loss = sum(log.get("lm_nll_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        ) # loss and nll_loss are already in base 2! 
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens, ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        # metrics.log_derived(
        #     "lm_ppl", lm_ppl, unsup_nsentences,
        # )
        metrics.log_scalar(
            "sim_loss", sim_loss, unsup_nsentences, round=3
        )
        metrics.log_scalar(
            "lm_loss", lm_loss / sample_size, sample_size, round=3
        )
        # metrics.log_scalar(
        #     "lm_nll_loss", lm_nll_loss / ntokens, ntokens, round=3
        # )
        # metrics.log_derived(
        #     "lm_ppl", lambda meters: utils.get_perplexity(meters["lm_nll_loss"].avg)
        # )
        metrics.log_scalar(
            "mean_per_word_entropy", mean_per_word_entropy, unsup_nsentences, round=3
        )
        metrics.log_scalar(
            "unsupervised_loss", unsupervised_loss, unsup_nsentences, round=3
        )
        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

   # average_entropy = 0.0
                # refs=sample['net_input']['src_tokens']
                # rows, cols = refs.size()
                # # logger.info(f"refs_size: {refs.size()}")
                # # logger.info(f"gsm_samples: {gsm_samples.size()}")
                # refs_list = []
                # preds_list = []
                
                # for i in range(rows):
                #     ref_sentence = []
                #     # pred_sentence = []
                #     for j in range(cols):
                #         ref_word = model.decoder.dictionary.__getitem__(refs[i, j].cpu().detach().numpy())
                #         # pred_word = model.decoder.dictionary.__getitem__(gsm_samples[i, j].argmax().cpu().detach().numpy())
                #         # prob_entropy = Categorical(gsm_samples[i,j,:]).entropy().cpu().detach().numpy()
                        
                #         if refs[i, j] != tgt_dict.pad():
                #             # average_entropy += prob_entropy
                #             if ref_word != '</s>' or '<pad>' not in ref_word:
                #                 ref_sentence.append(ref_word)
                #             # if pred_word != '<s>' or '<pad>' not in pred_word or pred_word != '</s>':
                #             #     pred_sentence.append(pred_word)
                #     refs_list.append(" ".join(ref_sentence).replace("@@ ", "").replace("</s>", "").rstrip())
                # #     preds_list.append(" ".join(pred_sentence).replace("@@ ", "").replace("</s>", "").replace("<s>", "").rstrip())
                # # average_entropy = average_entropy / (rows*cols)

                # rows, cols, _ = gsm_samples.size()
                # for i in range(rows):
                #     pred_sentence = []
                #     for j in range(cols):
                #         pred_word = model.decoder.dictionary.__getitem__(gsm_samples[i, j].argmax().cpu().detach().numpy())
                        
                #         if pred_word != '<s>' or '<pad>' not in pred_word or pred_word != '</s>':
                #             pred_sentence.append(pred_word)
                #     preds_list.append(" ".join(pred_sentence).replace("@@ ", "").replace("</s>", "").replace("<s>", "").rstrip())
                # # average_entropy = average_entropy / (rows*cols)        

                # # inds = random.sample(range(len(refs_list)), 2)
                # # rr = [refs_list[i] for i in inds]
                # # rp = [preds_list[i] for i in inds]
                # # logger.info(f"ref_list: {rr}")
                # # logger.info(f"pred_list: {rp}")
                # # logger.info(f"avg_entropy: {average_entropy}")

                # bert_loss, avg_f1 = self.compute_bertLoss(preds_list, refs_list)

                # ppl_results = self.perplexity.compute(data=preds_list, model_id='/netscratch/jalota/checkpoints/gpt2-finetuned-motra/', batch_size=len(preds_list), add_start_token=True)

            # if not train:
            #     logger.info(f"sample: {sample}")
            #     with torch.no_grad():
            #         net_output = model(**sample['net_input'])

            #         lprobs = model.get_normalized_probs(net_output, log_probs=True)
                    
            #         gsm_samples = gumbel_softmax(lprobs, tau=self.tau_gumbel_softmax, hard=self.hard_gumbel_softmax,eps=self.eps_gumbel_softmax, dim=-1)

            #         # gen_out = seqeunce_generator.generate(
            #         #     [model], sample, prefix_tokens=None, constraints=None)
            # else:
            #     # if cosine:
            #     #     gen_out = seqeunce_generator.generate(
            #     #         [model], sample, prefix_tokens=None, constraints=None)
            #     # else:
            #     net_output = model(**sample['net_input'])

            #     lprobs = model.get_normalized_probs(net_output, log_probs=True)
                    
            #     gsm_samples = gumbel_softmax(lprobs, tau=self.tau_gumbel_softmax, hard=self.hard_gumbel_softmax,eps=self.eps_gumbel_softmax, dim=-1)

            # logger.info(f"hyps: {hyps.size()}")
            # logger.info(f"shape sample['net_input']['src_tokens']: {sample['net_input']['src_tokens'].size()}")
            # logger.info(f"hyps[0]: {hyps[0]}")
                
            # encoder_out = getattr(net_output, "encoder_out") 
            # logger.info(f"hyps_tok: {hyps_tok}") 
            # /netscratch/jalota/checkpoints/gpt2-finetuned/
            # /netscratch/jalota/checkpoints/gpt2-finetuned-motra/
            # if cosine:
            #     cos_sim_loss = self.compute_cosineSimilarityLoss(model, sample, preds_list, train)
                # hyps, hyps_tok = [], []
                # for i in range(len(gen_out)):
                #     s = decode(gen_out[i][0]["tokens"]).strip()
                #     if len(s) > 0:
                #         hyps_tok.append(s)
                #     hyps.append(gen_out[i][0]["tokens"]) 
                # # [h.clone().detach() for h in hyps]
                # hyps = collate_tokens(hyps, src_dict.pad(), src_dict.eos(), left_pad=False, pad_to_length=None,pad_to_bsz=None)

                # cos_sim_loss = self.compute_cosineSimilarityLoss(model, sample, hyps, train)
                # if torch.isnan(cos_sim_loss):
                #     logger.info(f"hyps: {hyps}")
                #     logger.info(f"sample['net_input']['src_tokens']: {sample['net_input']['src_tokens']}")
                #     cos_sim_loss = torch.tensor(1e-10)

            # ppl_results = self.perplexity.compute(data=hyps_tok, model_id='/netscratch/jalota/checkpoints/gpt2-finetuned-motra/', batch_size=len(hyps_tok), add_start_token=True) # {'perplexities': [], 'mean_perplexity': float_value }
            # mean_per_word_entropy = math.log2(ppl_results['mean_perplexity'])
