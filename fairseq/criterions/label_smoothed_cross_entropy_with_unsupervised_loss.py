"""
Classes and methods used for multi-task tuning on Supervised style-transfer and unsupervised LM and cosine similarity losses. 
Authors: Rricha Jalota
"""

import torch.nn.functional as F
from fairseq import criterions
from dataclasses import dataclass, field
import torch
import math
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
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
        lm_weight=0.5,
        cosine_weight=1,
        unsupervised_weight=0.5,
        supervised_weight=1

    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.lm_weight = torch.tensor(1)
        self.cosine_weight = torch.tensor(1)
        self.unsupervised_weight = torch.tensor(1)
        self.supervised_weight = torch.tensor(1)
        self.perplexity = Perplexity()
        #load("perplexity", module_type="measurement")
        
    def forward(self, model, sample, seqeunce_generator=None, tgt_dict=None,reduce=True, unsup=False, src_dict=None, train=True):

        logging_output = {}
        loss = 0.0
        sample_size_set = False
        if train:
            net_output = model(**sample["sup"]["net_input"])
            loss_sum, nll_loss_sum = self.compute_loss(model, net_output, sample["sup"], reduce=reduce)
            sample_size = (
                sample['sup']["target"].size(0) if self.sentence_avg else sample['sup']["ntokens"]
            )
            ## take the mean of loss and nll_loss here and convert them from log base e to 2
            loss = loss_sum / sample_size / math.log(2)
            nll_loss = nll_loss_sum / sample['sup']["ntokens"] / math.log(2)
            # NOTE:
            # # we don't need to use sample_size as denominator for the gradient
            # # here sample_size is just used for logging
            sample_size = 1
            sample_size_set =  True
            if unsup:
                loss = self.supervised_weight * loss

            logging_output = {
                "loss" : loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample['sup']["ntokens"],
                "nsentences": sample['sup']["target"].size(0),
                "sample_size": sample_size,
            }

        if unsup:
            if train:
                sample = sample['unsup'] 
                # in case of eval, dataset is not RoundRobin, thus 'sample' can be used directly, & is not an OrderedDict!
                # toks.int().cpu(),
            def decode(toks, escape_unk=False):
                s = tgt_dict.string(
                    toks,
                    bpe_symbol="subword_nmt",
                )
                return s
            with torch.no_grad():
                gen_out = seqeunce_generator.generate(
                [model], sample, prefix_tokens=None, constraints=None)
            hyps, hyps_tok = [], []
            for i in range(len(gen_out)):
                s = decode(gen_out[i][0]["tokens"]).strip()
                if len(s) > 0:
                    hyps_tok.append(s)
                hyps.append(gen_out[i][0]["tokens"]) 
            # [h.clone().detach() for h in hyps]
            hyps = collate_tokens(hyps, src_dict.pad(), src_dict.eos(), left_pad=False, pad_to_length=None,pad_to_bsz=None)
            # logger.info(f"hyps: {hyps.size()}")
            # logger.info(f"shape sample['net_input']['src_tokens']: {sample['net_input']['src_tokens'].size()}")
            # logger.info(f"hyps[0]: {hyps[0]}")
                
            # encoder_out = getattr(net_output, "encoder_out") 
            # logger.info(f"hyps_tok: {hyps_tok}")
            cos_sim_loss = self.compute_cosineSimilarityLoss(model, sample, hyps, train)
            ppl_results = self.perplexity.compute(data=hyps_tok, model_id='/netscratch/jalota/checkpoints/gpt2-finetuned-motra/', batch_size=len(hyps_tok), add_start_token=True) # {'perplexities': [], 'mean_perplexity': float_value }
            mean_per_word_entropy = math.log2(ppl_results['mean_perplexity'])

            unsupervised_loss = self.cosine_weight * cos_sim_loss + self.lm_weight * mean_per_word_entropy

            loss += self.unsupervised_weight * unsupervised_loss   
            logging_output["loss"] = loss.data
            logging_output["unsupervised_loss"] = unsupervised_loss.data
            logging_output["cos_sim_loss"] = cos_sim_loss.data
            logging_output["mean_per_word_entropy"] = mean_per_word_entropy
            logging_output["unsup_nsentences"] = 1
            # sample['net_input']['src_tokens'].size(0) 
        
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        
        if sample_size_set == False:
            sample_size = 1
            # sample['net_input']['src_tokens'].size(0)

        return loss, sample_size, logging_output

    def compute_cosineSimilarityLoss(self, model, sample, hyps, train):
        # with torch.no_grad():
        if not train:
            with torch.no_grad():
                source_emb = model.encoder.forward(sample['net_input']['src_tokens'].cuda())
                gen_out_emb = model.encoder.forward(hyps) 

                source_sent_repr = torch.sum(source_emb['encoder_out'][0], dim=0)
        
                output_sent_repr = torch.sum(gen_out_emb['encoder_out'][0], dim=0).cuda()
                target_labels = torch.ones(source_sent_repr.shape[0], dtype=source_sent_repr.dtype).cuda()

                cosineLoss = torch.nn.CosineEmbeddingLoss(reduction='mean') 
                cos_sim_loss = cosineLoss(source_sent_repr, output_sent_repr, target_labels)
                
                return cos_sim_loss
        else:
            source_emb = model.encoder.forward(sample['net_input']['src_tokens'].cuda())
            gen_out_emb = model.encoder.forward(hyps)

            source_sent_repr = torch.sum(source_emb['encoder_out'][0], dim=0)
            
            output_sent_repr = torch.sum(gen_out_emb['encoder_out'][0], dim=0).cuda()
            target_labels = torch.ones(source_sent_repr.shape[0], dtype=source_sent_repr.dtype).cuda()

            cosineLoss = torch.nn.CosineEmbeddingLoss(reduction='mean') 
            cos_sim_loss = cosineLoss(source_sent_repr, output_sent_repr, target_labels)
            
            return cos_sim_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        # super().reduce_metrics(logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 1) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 1) for log in logging_outputs)
        cos_sim_loss = sum(log.get("cos_sim_loss", 0) for log in logging_outputs)
        mean_per_word_entropy = sum(log.get("mean_per_word_entropy", 0) for log in logging_outputs)
        unsupervised_loss = sum(log.get("unsupervised_loss", 0) for log in logging_outputs)
        unsup_nsentences = sum(log.get("unsup_nsentences", 1) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        ) # loss and nll_loss are already in base 2! 
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens, ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_scalar(
            "cos_sim_loss", cos_sim_loss, unsup_nsentences, round=3
        )
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
        