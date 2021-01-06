# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder

import numpy as np


class SequenceGenerator(object):
    def __init__(
        self,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        temperature=1.,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        #tgt dict für testen
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert temperature > 0, '--temperature must be greater than 0'

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )


    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        model = EnsembleModel(models)
        return self._generate(model, sample, **kwargs)

    @torch.no_grad()
    def _generate(
        self,
        model,
        sample,
        prefix_tokens=None,
        bos_token=None,
        **kwargs
    ):
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }
        '''
        print("len(sample)", len(sample))
        print("\n\nencoder_input.keys(): ", encoder_input.keys())
        print("encoder_input[src_tokens].shape: ", encoder_input["src_tokens"].shape)
        print("\nencoder_input: ", encoder_input)
        '''
        src_tokens = encoder_input['src_tokens']                        ################################################# src tokens
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        #print("\nsample id", sample["id"])
        #print("\nsample ntokens", sample["ntokens"])

        #print("sample src_lengths", sample["net_input"]["src_lengths"])
        #print("src_lengths: ", src_lengths)
        src_lengths_1st = src_lengths
        '''
        #print("sample src tokens", sample["net_input"]["src_tokens"])
        print("sample prev output tokens", sample["net_input"]["prev_output_tokens"])
        print("sample target first of 8", sample["target"])
        #print("check old src len", encoder_input["src_lengths"], " vers ", src_lengths)
        #print("src tokens", src_tokens)
        '''
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        #print(">>## bsz1", bsz)
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )
        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!'




        ####################### ENCODER_OUTS ########################################################################################################
        # compute the encoder output for each beam
        encoder_outs = model.forward_encoder(encoder_input)    ############### encoder_input: src_tokens, src_lengths; returns list of encoders from EnsembleModel

        encoder_out_emb_orig_1st = encoder_outs[0].encoder_out
        #print("Encoder Out orig", encoder_out_emb_orig_1st.shape)
        encoder_out_emb = encoder_out_emb_orig_1st.transpose(0, 1)
        #print("Encoder Out transposed", encoder_out_emb.shape)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1) # Test tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]) --- for beam 2 and bsz 8
        new_order = new_order.to(src_tokens.device).long()


        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order) ########### ruft index_select für dim (0: row, 1: col)je nach feld>>> encoder_padding_mask: 0, encoder_out:1, encoder_embedding:0
        ### für padding mask kopiert jede Zeile bsz times
        #print("\n\n >>> Encoder outs in sequence_generator ", encoder_outs[0].encoder_padding_mask)   #### Encoder outs in sequence_generator  TransformerEncoderOut(encoder_out=tensor([[[ 2.9137, -0.7925, -0.7197,  ..., -0.1992,
        encoder_out_emb_orig_2nd = encoder_outs[0].encoder_out
        #print("reordered shape encoder", encoder_out_emb_orig_2nd.shape)

        # initialize buffers
        #print("max len", max_len) ### 200
        scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        #print("tokens 1", tokens)
        tokens[:, 0] = self.eos if bos_token is None else bos_token ################ ersetzt das erste Element mit EOS/BOS=2
        attn, attn_buf = None, None

        # The blacklist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the blacklist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        blacklist = src_tokens.new_zeros(bsz, beam_size).eq(-1)  # forward and backward-compatible False mask ################ boolean

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS    #################### ?
        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)   ######## tokens type --- long
        #print("bbsz offsets", bbsz_offsets)   ### size [8,1] -> 0,2,4,6,8...14
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)
        #print("cand offsets", cand_offsets)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size or step == max_len:
                return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that    ####################### !!!!!!!!!???????
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
            """
            assert bbsz_idx.numel() == eos_scores.numel()
            #print("\nfinalize_hypos, step: ", step)
            #print("bbs_idx", bbsz_idx)

            # clone relevant token and attention tensors
            #print("finalize_hypos: tokens with a lot padding (here there are still all sentences), shape", tokens.shape)
            tokens_clone = tokens.index_select(0, bbsz_idx)
            #print("finalize_hypos tokens_clone = tokens.index_select(0, bbsz_idx) and shape (here are the sentences to be finalized, still padding)", tokens_clone.shape )#,tokens_clone)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            #print("finalize_hypos: tokens_clone[: , 1:step+2], step+2:still 1 pad at the end")#, tokens_clone)
            assert not tokens_clone.eq(self.eos).any()
            tokens_clone[:, step] = self.eos # replace pad with EOS (step, not step+2 because 0th BOS is cut out)
            #print("finalize_hypos: tokens_clone[:, step] = self.eos, replace last pad with EOS", tokens_clone)
            #print("finalize_hypos: attn.shape", attn.shape) ### bsz, 23 (seq_length), 202 ?(max len)
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2] if attn is not None else None
            #print("finalize_hypos: attn_clone.shape: ", attn_clone.shape)

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores   ### ????????????????!
            #print("??? finalize_hypos: Why do I need to compute scores per token position and ??? convert from cumulative to per-position scores? Substract left-shifted score???")
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                #print("finalize_hypos, self.normalize_scores True")
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            #print("finalize_hypos: finished", finished)
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)
                #print("    finalize_hypos: for f in finished ([False] * bsz) cum_unfin", cum_unfin)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                #print("*idx", idx)
                #print("*unfin_idx = idx//beam_size:", unfin_idx)
                sent = unfin_idx + cum_unfin[unfin_idx]
                #print("*sent = unfin_idx + cum_unfin[unfin_idx]:", sent)

                sents_seen.add((sent, unfin_idx))
                #print("    finalize_hypos: sents_seen:", sents_seen)

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i]
                    else:
                        hypo_attn = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': None,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                    #print("** Append hypo (tokens, score, attention, positional_scores) for sent ", sent, "to finalized[sent]")
            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx): # is_finished(): if len(finalized[sent]) == beam_size or step == max_len:
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            #print("FINALIZE_HYPOS: append to newly_finished after checking termination conditions (if len(finalized[sent]) == beam_size or step == max_len): ", newly_finished, "\n")
            return newly_finished

        reorder_state = None
        batch_idxs = None

        ###############################################################################################################################################################################################
        #print("max_len again", max_len)
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            #if (step == 16):
                #break
            #print("\n\n\n>>>>>>>>>>>>>>>>>>>>>STEP ", step)
            #print("reorder state", reorder_state) #1st is None
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    #print("batch idxs is not None", batch_idxs)
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                model.reorder_incremental_state(reorder_state) # FairseqIncrementalDecoder; ?recursive method calls child module TransformerDecoder
                encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state) 
                #print(">>## Enc Outs Iter", encoder_outs[0].encoder_out.shape)
                ###########################################################
            #print(">>>!!!!!!! Z.333 Enc Outs enc out 3rd here:", encoder_outs[0].encoder_out.shape)  ######### ENCODER EMBEDDINGS HERE!!!!!!!!!

            ###### feats von mir dazu
            #print("Tokens of forward_decoder -- tokens 2, Z.323, shape", tokens.shape)
            #print("tokens\n", tokens)### viele 1er als padding
            #print("tokens[:, :step + 1], Z.339, shape", tokens[:, :step + 1].shape, "tokens[:, :step + 1]\n", tokens[:, :step + 1])

            #print("Type Model: ", type(model))
            lprobs, avg_attn_scores, feats_tmp = model.forward_decoder(                  ###################################################################    Decoder ################
                tokens[:, :step + 1], encoder_outs, temperature=self.temperature,
            ) ### step + 1: 0th step is BOS; features_only changed by me, set to true in EnsembleModel _decode_one

            #print("\n---back in seq_gen: Z.324 feats_tmp decoder shape", feats_tmp.shape)

            # wie viele extract_features habe ich??? kann man forward_decoder und extract_features gleichzeitig abrufen?
            #test_features, _ = model.models[0].extract_features(src_tokens, src_lengths, sample['net_input']['prev_output_tokens'])  ####### Ensemble Model doesn't have extract features

            #print("lprobs != lprobs", lprobs)
            lprobs[lprobs != lprobs] = -math.inf    ############# ????????????????????? BOS darauf gesetzt???
            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            ########################  PASSIERT NICHT    #######################################
            # handle max length constraint
            if step >= max_len:         ### passiert nicht
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            #print("prefix tokens", prefix_tokens)   #### immer None
            # handle prefix tokens (possibly with different lengths)
            if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
                print("PREFIX TOKENS NOT NONE", prefix_tokens) ### immer None
                prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
                prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.pad)
                lprobs[prefix_mask] = -math.inf
                lprobs[prefix_mask] = lprobs[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
                )
                # if prefix includes eos, then we should make sure tokens and
                # scores are the same across all beams
                eos_mask = prefix_toks.eq(self.eos)
                if eos_mask.any():
                    # validate that the first beam matches the prefix
                    first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
                    eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
                    target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
                    assert (first_beam == target_prefix).all()

                    def replicate_first_beam(tensor, mask):
                        tensor = tensor.view(-1, beam_size, tensor.size(-1))
                        tensor[mask] = tensor[mask][:, :1, :]
                        return tensor.view(-1, tensor.size(-1))

                    # copy tokens, scores and lprobs from the first beam to all beams
                    tokens = replicate_first_beam(tokens, eos_mask_batch_dim)                   ####################################################
                    scores = replicate_first_beam(scores, eos_mask_batch_dim)
                    lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim)
            elif step < self.min_len:   ### min_len = 1
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf
            if self.no_repeat_ngram_size > 0:  #### why needed???????? now always 0
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if type(avg_attn_scores) is list:
                avg_attn_scores = avg_attn_scores[0]
            if avg_attn_scores is not None:
                if attn is None:    ### in step 0
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)

            self.search.set_src_lengths(src_lengths)        ######################### SEARCH set_src_lengths ####################
            #print("src_lengths:", src_lengths)

            if self.no_repeat_ngram_size > 0:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                    return gen_ngrams[bbsz_idx].get(ngram_index, [])

                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
                else:
                    banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                for bbsz_idx in range(bsz * beam_size):
                    lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf




            #test = scores.view(bsz, beam_size, -1)[:, :, :step]
            #print("lprobs size", lprobs.shape)          ### lprobs size torch.Size([16, 29384])  #-> [8, 2, 29384]
            #print("scores size", scores.shape)          ### scores size torch.Size([16, 201]) #-> [8, 2, : max len]


            cand_scores, cand_indices, cand_beams = self.search.step(               ##################### SEARCH  #########################################
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
            )
            '''
            print("cand scores\n")#, cand_scores)
            print("cand indices\n", cand_indices)
            print("cand beams\n", cand_beams)
            '''
            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            #print("cand bbsz idx\n", cand_bbsz_idx)

            # finalize hypotheses that end in eos, except for blacklisted ones
            # or candidates with a score of -inf
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][blacklist] = 0
            #print("blacklist", blacklist)    ### bsz, beam -> Bool: here- False

            #print("eos mask", eos_mask)

            # only consider eos when it's among the top beam_size indices
            torch.masked_select(            #### nur die Zahlen, bei denen die Mask True ist; ??? und nur für die ersten 2 Spalten
                cand_bbsz_idx[:, :beam_size],
                mask=eos_mask[:, :beam_size],
                out=eos_bbsz_idx,
            )
            #print("eos bbsz_idx", eos_bbsz_idx)

            finalized_sents = set()
            if eos_bbsz_idx.numel() > 0:
                torch.masked_select(
                    cand_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_scores,
                )
                #print("eos scores", eos_scores)
                finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores)   ###########################################################################
                #print("len finalized snt", len(finalized_sents))
                #print("finalized sents", finalized_sents)
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            #print("Finalized sents", finalized_sents)
            if len(finalized_sents) > 0:
                #print(">>entering finalized")
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                #print("batch mask new", batch_mask)
                #print("cand_indices.new(finalized_sents)", cand_indices.new(finalized_sents)) ### tensor([4, 3])
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)
                #print("batch mask", batch_mask)
                #print("batch idxs", batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                '''
                print("eos mask new", eos_mask)
                print("cand_beams new", cand_beams)
                print("bbsz offsets", bbsz_offsets)
                '''
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                #print("cand bbsz idx new\n", cand_bbsz_idx)
                #print("cand indices new\n", cand_indices)
                if prefix_tokens is not None:
                    #print("prefix tokens is not none", prefix_tokens)
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    #print("attn is not None")
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
                #print(">>##bsz2", bsz)
            else:
                batch_idxs = None
                #print("batch idxs ", batch_idxs, "in step ", step)

            # Set active_mask so that values > cand_size indicate eos or
            # blacklisted hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.
            '''
            print("eos mask new 2\n", eos_mask)
            print("blacklist 2", blacklist)
            print("cand offsets", cand_offsets)
            '''
            active_mask = buffer('active_mask')
            eos_mask[:, :beam_size] |= blacklist
            #print("eos_mask 3 after |= blacklist", eos_mask)
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )
            #print("active mask\n", active_mask)

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(new_blacklist, active_hypos)
            )
            #print("new_blacklist 1", new_blacklist)
            #print("cand_size", cand_size)

            # update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(cand_size)[:, :beam_size]
            assert (~blacklist).any(dim=1).all()
            '''
            print("again blacklist 2", blacklist)
            print("new active hypos", active_hypos)

            print("?REORDER cand_bbsz_idx acc to index=active_hypos")
            print("cand_bbsz_idx", cand_bbsz_idx)
            print("active_hypos", active_hypos)
            '''
            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)
            #print("active bbsz idx view:", active_bbsz_idx)
            #print("active scores view:", active_scores)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            #print("tokens buf\n", (tokens_buf != tokens))   ######## ??????????????????????
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )
            #print("scores buf", scores_buf)
            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens ########## ????????
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx


        ######################
        #print("\n\n\n\n\n###############################################################################")
        #print("BUFFERS: ", buffers.keys())
        #encoder_out_emb_orig_1st = model.forward_encoder(encoder_input)[0].encoder_out   ############### encoder_input: src_tokens, src_lengths; returns list of encoders from EnsembleModel
        src_mask = (src_tokens.ne(self.pad) & src_tokens.ne(self.eos))

        prev_output_tokens = sample['net_input']['prev_output_tokens']
        gold_tgt_mask = (prev_output_tokens.ne(self.pad) & prev_output_tokens.ne(self.eos))
        #print("tgt mask", gold_tgt_mask)

        #################
        #print("\n\n\n\n------In finalized")


        # sort by score descending
        def emb_tok2sent(args):
            return args.mean(0)
            #return args.sum(0)

        #distance_type = "cosine_similarity"
        distance_type = "euclidean"
        def distance_funct(v1, v2, dim=0, distance_type=distance_type):
            distance = None
            if distance_type == "cosine_similarity":
                distance = torch.nn.functional.cosine_similarity(v1, v2, dim=dim)
            if distance_type == "euclidean":
                distance = torch.dist(v1, v2, p=2)
            return distance


        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)
            #print("KEYS finalized[sent][0].keys()", finalized[sent][0].keys())
            #### sample gold
            gold_tgt_tok = prev_output_tokens[sent][gold_tgt_mask[sent]]

            #### enc(src)
            tok_in_src = src_tokens[sent][src_mask[sent]].unsqueeze(0)
            enc_in_src= {"src_tokens": tok_in_src, "src_lengths": torch.tensor(tok_in_src.shape[1])}
            enc_out_src = model.forward_encoder(enc_in_src)
            emb_enc_src = enc_out_src[0].encoder_out.transpose(0,1).squeeze()
            semb_enc_src = emb_tok2sent(emb_enc_src)
            #print("semb_enc_src shape: ", semb_enc_src.shape)


            #print("Extract features 2")
            ### dec(src) embedding with enc_out from gold tgt"
            dec_src_f, _ = model.models[0].extract_features(gold_tgt_tok.unsqueeze(0), gold_tgt_tok.shape, enc_in_src["src_tokens"])  #################################################### src in decoder extract_features
            semb_dec_src_enc_tgtgold = emb_tok2sent(dec_src_f.squeeze())
            #print("semb_dec_src_enc_tgtgold.shape", semb_dec_src_enc_tgtgold.shape)

            #print("Extract features 3")
            ### dec(src) embedding without encoder repr
            feats_src_plus, feats_src_minus = model.forward_decoder_test(enc_in_src["src_tokens"] )  ################################## src in decoder forward_decoder
            emb_dec_src_noenc_plus = feats_src_plus.squeeze()
            #print("emb_dec_src_noenc.shape: ", emb_dec_src_noenc_plus.shape)
            semb_dec_src_noenc_plus = emb_tok2sent(emb_dec_src_noenc_plus)
            semb_dec_src_noenc_minus = emb_tok2sent(feats_src_minus.squeeze())
            #print("dist_dec_fwdec_src_noenc.shape: ", semb_dec_src_noenc_plus.shape)

            # print("Extract features 4")
            ### dec(src) embedding with repeated src in encoder_out
            dec_src_f2, _ = model.models[0].extract_features(enc_in_src["src_tokens"], enc_in_src["src_tokens"].shape[1], enc_in_src["src_tokens"])
            semb_dec_src2 = emb_tok2sent(dec_src_f2.squeeze())
            #print("semb_dec_src2.shape: ", semb_dec_src2.shape)


            for hyp in range(len(finalized[sent])):
                #print("\nTokens for sent ", sent, " beam nr ", hyp, " score: ", finalized[sent][hyp]["score"])
                hyp_tok = finalized[sent][hyp]["tokens"]
                hyp_mask = hyp_tok.ne(self.eos)
                hyp_tok = hyp_tok[hyp_mask]
                hyp_words = self.tgt_dict.string(hyp_tok)

                data_sub = dict()
                data_sub["beam"] = "hyp" + str(hyp)  ########################

                ###
                #print("\n############ forward_decoder with hyp_tok from hypothesis and enc_outs_test from src")
                #print("\nExtract features 5a")
                lprobs2, avg_attn_scores2, feats_tmp2 = model.forward_decoder(hyp_tok.unsqueeze(0), enc_out_src, temperature=self.temperature, use_incremental=False,)  ################################## tgt in decoder forward_decoder
                #print("feats_tmp2.shape: ", feats_tmp2.shape)
                emb_dec = feats_tmp2.squeeze()
                #print("-.- squeezed - emb_dec.shape: ", emb_dec.shape)
                dist_dec_fwdec = emb_tok2sent(emb_dec)
                #print("Dist shapes: semb_enc_src.shape: ", semb_enc_src.shape," dist_dec_fwdec.shape: ", dist_dec_fwdec.shape)
                dist_enc_dec_fwdec = distance_funct(semb_enc_src, dist_dec_fwdec, dim=0)
                #print("\n---- Distance:  dist_enc_dec_fwdec -> enc-dec forward_decoder: ", dist_enc_dec_fwdec) ##doppelt, welche besser?
                data_sub["maybe-nosense-dist-enc(src)-dec(hyp)"] = f'{dist_enc_dec_fwdec.item():1.3f}'


                #print("\n############ extract_features with prev_output_tokens from hypothesis")
                ###
                '''
                print("\nExtract features 5b")
                print("enc_in_src['src_lengths']: ", enc_in_src["src_lengths"])
                print("enc_in_src['src_tokens']: ", enc_in_src["src_tokens"])
                print("hyp_tok.unsqueeze(0).shape", hyp_tok.unsqueeze(0).shape)
                '''
                dec_tgt_f, _ = model.models[0].extract_features(enc_in_src["src_tokens"], enc_in_src["src_lengths"], hyp_tok.unsqueeze(0)) ################################################## tgt in decoder extract_features
                #print("drc_tf.shape: ", dec_tgt_f.shape)
                dec_tf_sq = dec_tgt_f.squeeze()
                #print("dec_tf_sq.shape: ", dec_tf_sq.shape)
                #dist_dec_extrft = dec_tf_sq.mean(0)
                dist_dec_extrft = emb_tok2sent(dec_tf_sq)
                #print("dist_dec_extrft.shape: ", dist_dec_extrft.shape)
                dist_enc_dec_extrft = distance_funct(semb_enc_src, dist_dec_extrft, dim=0)
                #print("\n---- Distance: dist_enc_dec_extrft -> enc-dec extract_features: ", dist_enc_dec_extrft) ### doppelt

                #print("\n######## extract_features second direction, with: prev_output tokens from src; tokens-to-translate from hypothesis")
                #print("Dist shapes: semb_dec_src_enc_tgtgold.shape: ", semb_dec_src_enc_tgtgold.shape, " - dist_dec_extrft.shape: ", dist_dec_extrft.shape)
                dist_dec_dec_extrft = distance_funct(semb_dec_src_enc_tgtgold, dist_dec_extrft, dim=0)
                #print("\n---- DISTANCE: dist_dec_dec_extrft -> dec-dec extract_features: ", dist_dec_dec_extrft)
                data_sub["[GOAL]:dist-dec(src+enc:gold_tgt)-dec(hyp)"] = f'{dist_dec_dec_extrft.item():1.3f}'


                #print("\n######## extract_features second direction, with: prev_output tokens from src; tokens-to-translate from tgt")
                #print("\nExtract features 7")
                dec_src_enc_hyp_f, _ = model.models[0].extract_features(hyp_tok.unsqueeze(0), hyp_tok.shape, enc_in_src["src_tokens"])
                dist_dec_src_enc_hyp = emb_tok2sent(dec_src_enc_hyp_f.squeeze())
                #print("Dist shapes: semb_dec_src_enc_tgtgold.shape: ", dist_dec_src_enc_hyp.shape, " - dist_dec_extrft.shape: ", dist_dec_extrft.shape)
                dist_dec_src_hyp_dec_hyp_extrft = distance_funct(dist_dec_src_enc_hyp, dist_dec_extrft, dim=0)
                #print("\n---- DISTANCE: dist_dec_src_hyp_dec_hyp_extrft : ", dist_dec_src_hyp_dec_hyp_extrft)
                data_sub["[try to approach GOAL]:dist-dec(src+enc:hyp)-dec(hyp)"] = f'{dist_dec_src_hyp_dec_hyp_extrft.item():1.3f}'


                #print("\n########### extract_features 2nd direction encoder_out = None for both src and tgt")
                #print("hyp_tok.shape: ", hyp_tok.unsqueeze(0).shape, "--- ")
                #print("enc_in_src['src_tokens'].shape: ", enc_in_src["src_tokens"].shape)
                #print("\nExtract features 8")
                feats_tmp4_plus, feats_tmp4_minus = model.forward_decoder_test(hyp_tok.unsqueeze(0))  ################################## tgt in decoder forward_decoder
                #print(" Feats tmp4: ", type(feats_tmp4_plus), len(feats_tmp4_plus), " shape: ", feats_tmp4_plus.shape)
                emb_dec_hyp_noenc_plus = feats_tmp4_plus.squeeze()
                #print("emb_dec_tgt_noenc.shape: ", emb_dec_hyp_noenc_plus.shape)
                #dist_dec_hyp_noenc = emb_dec_hyp_noenc.mean(0)
                dist_dec_hyp_noenc_plus = emb_tok2sent(emb_dec_hyp_noenc_plus)
                #print("dist_dec_tgt_noenc.shape: ", dist_dec_tgt_noenc.shape)
                #print("Dist shapes: dist_dec_fwdec_src_noenc.shape: ", semb_dec_src_noenc_plus.shape, "dist_dec_tgt_noenc.shape: ", dist_dec_hyp_noenc_plus.shape)
                dist_dec_dec_src_hyp_noenc2_posplus = distance_funct(semb_dec_src_noenc_plus, dist_dec_hyp_noenc_plus, dim=0)
                #print("\n---- ***noenc-2*** DISTANCE:  dist_dec_dec_src_tgt_noenc2 -> dec-dec extract_features_test, dist decoder emb of src and hyp both noenc POSPLUS: ", dist_dec_dec_src_hyp_noenc2_posplus)
                data_sub["dist-dec(src_noenc_posplus)-dec(hyp_noenc_posplus)"] = f'{dist_dec_dec_src_hyp_noenc2_posplus.item():1.3f}'

                dist_dec_hyp_noenc_minus = emb_tok2sent(feats_tmp4_minus.squeeze())
                dist_dec_dec_src_hyp_noenc2_posminus = distance_funct(semb_dec_src_noenc_minus, dist_dec_hyp_noenc_minus, dim=0)
                #print("\n---- ***noenc-2*** DISTANCE:  dist_dec_dec_src_tgt_noenc2 -> dec-dec extract_features_test, dist decoder emb of src and hyp both noenc POSMINUS: ", dist_dec_dec_src_hyp_noenc2_posminus)
                data_sub["dist-dec(src_noenc_posminus)-dec(hyp_noenc_posminus)"] = f'{dist_dec_dec_src_hyp_noenc2_posminus.item():1.3f}'


                #print("\n############ enc-enc distance btw src and tgt")
                hyp_tok_ts = hyp_tok.unsqueeze(0)
                #print("hyp_tok_ts.shape: ", hyp_tok_ts.shape)
                #print("\nExtract features 9")
                enc_in_hyp = {"src_tokens": hyp_tok_ts, "src_lengths": torch.tensor(hyp_tok_ts.shape[1])}
                enc_outs_hyp = model.forward_encoder(enc_in_hyp)

                emb_enc_hyp = enc_outs_hyp[0].encoder_out.transpose(0, 1).squeeze()
                #print("emb_enc_src.shape: ", emb_enc_hyp.shape)
                #dist_enc_hyp = emb_enc_hyp.mean(0)
                dist_enc_hyp = emb_tok2sent(emb_enc_hyp)
                #print("dist_enc_tgt shape: ", dist_enc_hyp.shape)
                dist_enc_enc = distance_funct(semb_enc_src, dist_enc_hyp, dim=0)
                #print("\n------ ***### Distance dist_enc_enc: ", dist_enc_enc.item())
                tmp = dist_enc_enc.item()
                #print(f'{tmp:1.3f}')
                data_sub["dist-enc(src)-enc(hyp)"] = f'{dist_enc_enc.item():1.3f}'



                ################    UNNECESSARY
                #print("\n########### extract_features 2nd direction encoder_out = None; Different representations, bad results")
                #print("Dist shapes: dist_dec_fwdec_src_noenc.shape: ", semb_dec_src_noenc_plus.shape, " dist_dec_fwdec.shape: ", dist_dec_fwdec.shape)
                # dist_dec_dec_src_noenc_2hyp = torch.nn.functional.cosine_similarity(dist_dec_fwdec, dist_dec_fwdec_src_noenc, dim=0)
                dist_dec_dec_src_noenc_posplus_2hyp = distance_funct(semb_dec_src_noenc_plus, dist_dec_extrft, dim=0)
                #print("\n---- ***noenc*** DISTANCE:  dist_dec_dec_src_noenc_2hyp-> enc-dec extract_features_test, dist decoder emb of src and hyp: ", dist_dec_dec_src_noenc_posplus_2hyp)
                data_sub["nosense-dist-dec(src_noenc_posplus)-dec(hyp)"] = f'{dist_dec_dec_src_noenc_posplus_2hyp:1.3f}'



                #print("\n########### extract_features 2nd direction, 2XSRC: encoder_out = src")
                dist_dec_dec_src_srcenc_2hyp = distance_funct(semb_dec_src2, dist_dec_extrft, dim=0)
                #print("\n---- ***x2*** DISTANCE:  dist_dec_dec_src_srcenc_2hyp -> enc-dec forward_decoder, dist decoder emb of src and hyp: ", dist_dec_dec_src_srcenc_2hyp)
                data_sub["nosense-dist-dec(2args<-src)-dec(hyp)"] = f'{dist_dec_dec_src_srcenc_2hyp:1.3f}'



                #print("\n############ extract_features with prev_output_tokens from hypothesis and encoder hypothesis")
                ###
                #print("hyp tok shape", hyp_tok.shape)
                dec_tgt_f_hyp, _ = model.models[0].extract_features(hyp_tok.unsqueeze(0), hyp_tok, hyp_tok.unsqueeze(0))  ################################################## tgt in decoder extract_features
                #print("drc_tf_hyp.shape: ", dec_tgt_f_hyp.shape)
                dec_tf_2hyp = dec_tgt_f_hyp.squeeze()
                #print("dec_tf_2hyp.shape: ", dec_tf_2hyp.shape)
                # dist_dec_extrft_2hyp = dec_tf_2hyp.mean(0)
                dist_dec_extrft_2hyp = emb_tok2sent(dec_tf_2hyp)
                #print("dist_dec_extrft_2hyp.shape: ", dist_dec_extrft_2hyp.shape)
                dist_dec_dec_extrft_2hyp = distance_funct(semb_dec_src2, dist_dec_extrft_2hyp,
                                                                                 dim=0)
                #print("\n---- ###*** Distance: dist_dec_dec_extrft_2hyp -> dec-dec extract_features: ",dist_dec_dec_extrft_2hyp)
                data_sub["nosense-dist-dec(2args<-src)-dec(2args<-hyp)"] = f'{dist_dec_dec_extrft_2hyp.item():1.3f}'

                #print("\n")

                ### adding for table
                #print("id_1: ", id_1, " -- id_2: ", id_2)
                finalized[sent][hyp]["distances"] = data_sub

                #data_coll.append(data_sub)
            #data_table[id_1] = data_coll
        #print("\n\n")

        # finalized: list of lists of dictionaries (1 per hypothesis); keys: tokens, score (scalar), attention, alignment (None), positional_scores
        return finalized




class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(hasattr(m, 'decoder') and isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.encoder(**encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1., use_incremental=True):
        #print(">>>>> EnsembleModel forward_decoder")
        #print("...use_incremental: ", use_incremental)
        if len(self.models) == 1:
            #print(">>>>> Is only one Ensemble Model")
            #print("in EnsembleModel forward_decoder, use_incremental: ", use_incremental)
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
                use_incremental=use_incremental,
            )
        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
                use_incremental=use_incremental,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    @torch.no_grad()
    def forward_decoder_test(self, tokens, temperature=1.):
        #print(">>>>> EnsembleModel forward_decoder")
        if len(self.models) == 1:
            #print(">>>>> Is test decoder ensemble")
            decoder_out_test_plus_pos = self.models[0].decoder.extract_features_test_posplus(tokens)
            decoder_out_test_munus_pos = self.models[0].decoder.extract_features_test_posminus(tokens)

            '''
            my_model = self.models[0]
            print("Type my_model: ", type(my_model))
            decoder_out_test = list(my_model.forward_decoder(tokens, encoder_out=None, features_only=True))
            '''
            return decoder_out_test_plus_pos, decoder_out_test_munus_pos



    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1., use_incremental=True,
    ):
        #print("*******in EnsembleModel _decode_one, use_incremental: ", use_incremental)
        if self.incremental_states is not None:
            #print(">>>>> incremental states not None", " >>>>> one Ensemble Model of type: ", type(model)) ### TransformerModel
            decoder_out_test = list(model.forward_decoder(
                tokens, encoder_out=encoder_out, incremental_state=self.incremental_states[model], features_only=True, use_incremental=use_incremental, #return_all_hiddens=True, ##### changed
            ))
            ### Meine Änderungen hier:
            #print("\nEnsembleModel Z.792 Tokens in _decode_one:\n", tokens)
            feats_tmp, extra_tmp = decoder_out_test
            #print("Shape of feats in EnsembleModel _decode_one", feats_tmp.shape)
            x = model.decoder.output_layer(feats_tmp)
            decoder_out = [x, extra_tmp]
        else:
            decoder_out_test = list(model.forward_decoder(tokens, encoder_out=encoder_out, features_only=True) )#### changed
            ### Meine Änderungen hier:
            #print("\nEnsembleModel Z.802 Tokens in _decode_one:\n", tokens)
            feats_tmp, extra_tmp = decoder_out_test
            #print("Shape of feats in _decode_one", feats_tmp.shape)
            x = model.decoder.output_layer(feats_tmp)
            decoder_out = [x, extra_tmp]


        decoder_out[0] = decoder_out[0][:, -1:, :]


        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if type(attn) is list:
            attn = attn[0]
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        ### Von mir: feats dazu
        return probs, attn, feats_tmp

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)


class SequenceGeneratorWithAlignment(SequenceGenerator):

    def __init__(self, tgt_dict, left_pad_target=False, **kwargs):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        model = EnsembleModelWithAlignment(models)
        finalized = super()._generate(model, sample, **kwargs)

        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        src_tokens, src_lengths, prev_output_tokens, tgt_tokens = \
            self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, 'full_context_alignment', False) for m in model.models):
            attn = model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [
                finalized[i // beam_size][i % beam_size]['attention'].transpose(1, 0)
                for i in range(bsz * beam_size)
            ]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            alignment = utils.extract_hard_alignment(attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos)
            finalized[i // beam_size][i % beam_size]['alignment'] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        src_tokens = src_tokens[:, None, :].expand(-1, self.beam_size, -1).contiguous().view(bsz * self.beam_size, -1)
        src_lengths = sample['net_input']['src_lengths']
        src_lengths = src_lengths[:, None].expand(-1, self.beam_size).contiguous().view(bsz * self.beam_size)
        prev_output_tokens = data_utils.collate_tokens(
            [beam['tokens'] for example in hypothesis for beam in example],
            self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam['tokens'] for example in hypothesis for beam in example],
            self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]['attn']
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        print("_decode_one called in EnsembleModelWithAlignment(EnsembleModel)")
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens,
                encoder_out=encoder_out,
                incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if type(attn) is list:
            attn = attn[0]
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn
