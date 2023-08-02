from fairseq.models.transformer import TransformerModel
from fairseq.models.transformer_lm import  TransformerLanguageModel
from fairseq.tasks.language_modeling import LanguageModelingTask
import torch
import logging 
import os, sys
import torch.nn as nn
from fairseq import tasks, checkpoint_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from argparse import Namespace

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq.lm_perplexity.lm")

# task_args= Namespace(task="language_modeling", data="/netscratch/jalota/datasets/motra-sst/ppd_w_europarl-motra-10k_no_dups/en_es_de/unsup_setup/lm_finetune/", tokens_per_sample=512, output_dictionary_size= -1, dataset_impl='mmap', future_target=False, self_target=False, past_target=False, )

# model_args = Namespace(tokens_per_sample=512, data="/netscratch/jalota/datasets/motra-sst/ppd_w_europarl-motra-10k_no_dups/en_es_de/unsup_setup/lm_finetune/", arch="transformer_lm", activation_fn='relu', dropout=0.1, attention_dropout= 0.0, activation_dropout= 0.0, relu_dropout= 0.0, decoder_embed_dim= 512, decoder_output_dim= 512, decoder_input_dim= 512, decoder_ffn_embed_dim= 2048, decoder_layers= 6, decoder_attention_heads= 8, decoder_normalize_before= False, no_decoder_final_norm= False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0.0, adaptive_softmax_factor=4.0, no_token_positional_embeddings=False, share_decoder_input_output_embed=True, character_embeddings= False, character_filters='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]', character_embedding_dim=4, char_embedder_highway_layers=2, adaptive_input= False, adaptive_input_factor= 4.0, adaptive_input_cutoff= None, tie_adaptive_weights=False, tie_adaptive_proj= False, decoder_learned_pos= False, layernorm_embedding= False, no_scale_embedding= False, checkpoint_activations= False, offload_activations= False, decoder_layerdrop= 0.0, decoder_layers_to_keep= None, quant_noise_pq= 0.0, quant_noise_pq_block_size=8, quant_noise_scalar= 0.0, min_params_to_wrap=100000000, base_layers= 0, base_sublayers=1, base_shuffle=1, scale_fc=False, scale_attn=False, scale_heads=False, scale_resids= False, decoder_xformers_att_config=None, add_bos_token= False, max_target_positions= None, tpu= False)
# /netscratch/jalota/datasets/motra-sst/ppd_w_europarl-motra-10k_no_dups/en_es_de/unsup_setup/lm_finetune/
# /netscratch/jalota/datasets/motra-sst/de/unsup_setup_raw/lm_finetuning/
class LanguageModel:
    """
    Transformer LanguageModel to compute perplexity or cross entropy. 
    """
    def __init__(
            self,
            path=None,
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='/netscratch/jalota/datasets/motra-sst/ppd_w_europarl-motra-10k_no_dups/en_es_de/unsup_setup/lm_finetune/',
            device = None,
            tgt_dict=None
    ):
        # /netscratch/jalota/datasets/motra-sst/de/unsup_setup_raw/lm_finetuning/
        obj = TransformerModel.from_pretrained(
            path,
            checkpoint_file,
            data_name_or_path,
            )
        self._model = obj.models[0]
        # logger.info("self._model.type: {self._model.__class__.__name__}")
        # print(self._model)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._emb_matrix = None
        # we want to keep the weights of the pretrained model frozen
        for name, param in self._model.named_parameters():
            # logger.info(f"name: {name}")
            if 'embed_tokens.weight' in name:
                emb_matrix = param
                self._emb_matrix = emb_matrix
            param.requires_grad = False

        old_num_tokens, old_embedding_dim = self._emb_matrix.size()

        # build new embeddings: https://huggingface.co/transformers/v2.11.0/_modules/transformers/modeling_utils.html#PreTrainedModel.resize_token_embeddings
        new_emb_matrix = torch.nn.Embedding(len(tgt_dict), old_embedding_dim).requires_grad_(False)
    
            # new_emb_matrix = init_weights(new_emb_matrix)
        nn.init.normal_(new_emb_matrix.weight, mean=0, std=old_embedding_dim**-0.5)

        # print(f"new_emb_matrix.weight: {new_emb_matrix.weight}")
        # print(f"self._emb_matrix.weight: {self._emb_matrix}")

        # Copy token embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, len(tgt_dict))
        # logger.info(f"num_tokens_to_copy: {num_tokens_to_copy}")
        new_emb_matrix.weight[:num_tokens_to_copy, :] = self._emb_matrix[:num_tokens_to_copy, :]

        self._emb_matrix = new_emb_matrix.weight

        self._model.decoder.embed_tokens.weight = new_emb_matrix.weight
        self._model.decoder.output_projection.weight = new_emb_matrix.weight

        # logger.info(f"self._model.decoder.embed_tokens.weight: {self._model.decoder.embed_tokens.weight.size()}")

        # logger.info(f"self._emb_matrix.size(): {self._emb_matrix.size()}")
        
        self._model.to(self.device)
        self._emb_matrix.to(self.device)
    

    def get_lm_out_from_decoder_inp(self, preds, batch_size=64, verbose=True):
        """
        Args:
            - :param: `preds` (torch.tensor BxTxE): predicted logits

        Return:
            Return:
            - :param: 'lm_out' (torch.tensor BxTxE): output of LM when fed predicted logits from the decoder
        """

        return self.get_lm_output(
            preds,
            verbose=verbose,
            device=self.device,
            batch_size=batch_size,
        ) 

    def get_lm_output(self, 
                    preds, verbose=False, device="cuda:0", batch_size=64):
        """
        Args:
            - :param: `preds` (torch.tensor BxTxE): predicted logits
        """
        return self.get_lm_out_from_tensor(
        preds, device=device)


    def get_lm_out_from_tensor(self, 
                                      preds_tensor, 
                                      device="cuda:0"):
        """
        Compute LM embedding in batches.

        Args:
            - :param: `preds_tensor` (torch.tensor) : preds tensor.
            - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
        """
        batch_size, max_seq_len, vocab_size = preds_tensor.size()   

        emb_size = self._emb_matrix.size()[-1]

        preds_tensor = preds_tensor.to(device)
        self._emb_matrix = self._emb_matrix.to(device)

        preds_tensor_embs = torch.mm(preds_tensor.contiguous().view(-1, vocab_size), self._emb_matrix)
        # logger.info("preds_tensor_embs.size(): {preds_tensor_embs.size()}")
        preds_tensor_embs = preds_tensor_embs.view(-1, max_seq_len, emb_size)

        # logger.info(f"model.__class__.__name__: {self._model.__class__.__name__}")

        lm_out = self._model(preds_tensor_embs) 

        # logger.info(f"lm_out: {lm_out}")
        return lm_out 

        






        
        
        







        
        