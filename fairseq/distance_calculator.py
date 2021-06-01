import math
from typing import Dict, List, Optional

import torch
from copy import deepcopy
import sys
from stop_words import get_stop_words
import re

import logging
logger = logging.getLogger('logger')
#handler = logging.StreamHandler(sys.stderr)

logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)
#logger.addHandler(handler)
#logging.basicConfig(level=logging.DEBUG)

# FOR REMOVING STOPWORDS IN CALCULATION
stopwords = dict()
'''
stopwords["de"] = set(get_stop_words('de'))
stopwords["en"] = set()
tmp = get_stop_words('en')
for w_en in tmp:
    sp = w_en.split("'")
    for i in sp:
        stopwords["en"].add(i)

'''

# TODO piece for sentencepiece, for other BPE algorithms change also function
'''
piece = "▁"
q1 = piece + "'"
q2 = piece + '"'
'''

### +LM
#from fairseq import hub_utils
from fairseq.models.transformer_lm import TransformerLanguageModel
# TODO: try with 2 monolingual models
#custom_lm = TransformerLanguageModel.from_pretrained('/raid/data/daga01/fairseq_train/lm_models/my_LM/', 'checkpoint_best.pt')
#custom_lm = TransformerLanguageModel.from_pretrained('/raid/data/daga01/fairseq_train/lm_models/my_LM_de', 'checkpoint_best.pt')
custom_lm = None


class DistanceCalculator():
    def __init__(
            self,
            model,
            tgt_dict,
            lm = None,
            lm_weight_4dist = 1,
            eos=None,
            normalize_scores=True,
            temperature=1.0,
    ):
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.vocab_size = len(tgt_dict)
        self.normalize_scores = normalize_scores ### ??? brauche ich???
        self.model = model
        self.model.eval()
        ### set and unset encoder ???
        self.encoder = model.encoder
        self.decoder = model.decoder
        # TODO: define constants/option dicts with possible values for distance_type and sentence_repr
        self.distance_type = "cosine_similarity" # cosine_similarity, euclidean
        self.sentence_repr = "scalar_mean"   #  scalar_mean, vector_bertscore, vector_bertscore_aligned (TODO: assert that the TransformerModel is WithAlignment )
        #TODO: get from config
        self.src_lang = "en"
        self.tgt_lang = "de"
        #TODO: pass options for remove_stopwords and lm_model
        self.remove_stopwords = False
        self.lm_model = custom_lm
        if self.lm_model:
            self.lm_model.eval()
        self.print_poc = True
        print("\n\ncosine", "scalar_mean", "None")


    def check_lm(self, tokens):
        calc = self.lm_model.score(self.check_token2word(tokens))['positional_scores']
        #print("POS LM: ", calc)
        ppl = calc.mean().neg().exp().item()
        #lm_score = calc.item()
        lm_score =  self.lm_model.score(self.check_token2word(tokens))['score'].item()
        #print("tokens: ", self.check_token2word(tokens), " ppl: ", ppl)
        return ppl, lm_score

    def filter_stop_words(self, lang, tokens):
        #TODO: evtl find index2string and string2index for BPEs
        #tokens = torch.tensor([[17, 17926, 23127]])
        words = self.tgt_dict.string(tokens).lower().split(" ")
        #print("\n###\nwords", words)
        indices = tokens.tolist()[0]

        tmp = ["", []]
        dict_list = []
        assert (len(words) == len(indices)), "Words and indices are of different length!"
        noquote = True
        for i in range(len(words)):
            if words[i].startswith(piece):
                tmp = [words[i][1:], [indices[i]]]
                #print("->", words[i][1:])
                dict_list.append(tmp)
                if words[i] == q1 or words[i] == q2:
                    noquote = False
            else:
                if noquote:
                    #print("W", words[i])
                    tmp[0] += words[i]
                    tmp[1].append(indices[i])
                    #print("tmp", tmp)
                else:
                    tmp = [words[i], [indices[i]]]
                    #print("->*", words[i])
                    dict_list.append(tmp)
                    noquote = True

        #print("dl: ",dict_list)
        str2tok = dict(dict_list)
        #print(str2tok)

        new_tokens = []
        stopword_cnt = 0
        #print("check stopwords")
        #TODO reverse logic again; not stopword and not punctuation
        for w in str2tok:
            #print("w", w)
            if w in stopwords[lang]:
                stopword_cnt += 1
                #print("+++ w is stopwword", w, " -- ", str2tok[w])
            elif re.match(r'[^\w\s]*(?!\w+)', w) :
                #print("*** is punctuation", w, " -- ", str2tok[w])
                pass
            else:
                new_tokens.extend(str2tok[w])
                #pass
                #print("not stopword", w)
        #print("new tokens", new_tokens)
        # if almost everything is stopwords
        if (len(new_tokens) == 0 or len(new_tokens) < (0.2 * len(tokens[0]))):  ### new_tokens is what remains after filtering of stopwords and puctuation; keep the sentence if it contains more than 80% punctuation
            '''
            print("\n### only stop words")
            print("tokens: ", tokens)
            print("words: ", words)
            print("len not stopwords and punct: ", len(new_tokens), " -- len tokens", len(tokens[0]) )
            '''

            return tokens
        new_tensor = torch.tensor(new_tokens)
        new_tensor = new_tensor.unsqueeze(0)
        device = tokens.get_device()
        if device >= 0:
            return new_tensor.to(tokens.device)
        return new_tensor



    def check_token2word(self, tokens):
        words = self.tgt_dict.string(tokens)
        return words

    def emb_tok2sent(self, args):
        if self.sentence_repr == "scalar_mean":
            return args.mean(0)
        elif self.sentence_repr == "vector_bertscore":
            return args

    def distance_funct(self, v1, v2, alignment=None):
        distance = None
        if self.sentence_repr == "scalar_mean":
            distance = self.distance_funct_scalar(v1, v2)
        elif self.sentence_repr == "vector_bertscore":
            distance = self.distance_bertscore(v1, v2)
        elif self.sentence_repr == "vector_bertscore_aligned":
            distance = self.distance_bertscore_aligned(v1, v2, alignment)
        return distance


    def distance_bertscore_aligned(self, src, hypo, alignment):
        distance = None
        print("src: ", src.shape)
        distance = [(self.distance_funct_scalar(src[s]), self.distance_funct_scalar(hypo[t].unsqueeze(0))) for
             s, t in alignment]

        return distance


    def distance_bertscore(self, v1, v2):
        #TODO: check the thing with "all_layers"
        '''
        print("shape v1: ", v1.shape)
        print("shape v2: ", v2.shape)
        '''
        v1.div_(torch.norm(v1, dim=-1).unsqueeze(-1))
        v2.div_(torch.norm(v2, dim=-1).unsqueeze(-1))
        '''
        print("shape v1-2: ", v1.shape)
        print("shape v2-2: ", v2.shape)
        '''
        #sim = torch.bmm(v1, v2.T) #TODO: use bmm with batch
        sim = torch.matmul(v1, v2.T)
        #print("sim.shape: ", sim.shape)
        word_p = sim.max(dim=0)[0]
        word_r = sim.max(dim=1)[0]
        '''
        print("sim.shape: ", sim.shape)
        print("word_p: ", word_p)
        print("word_p.shape: ", word_p.shape)
        '''
        # difference to original Bertscore:  I use mean instead of idf-rescaling
        P = word_p.mean(dim=0)
        R = word_r.mean(dim=0)
        F = 2 * P * R / (P + R)
        '''
        print("P: ", P)
        print("R: ", R)
        print("F: ", F)
        '''
        return F.item()


    def distance_funct_scalar(self, v1, v2, dim=0):
        distance = None
        if self.distance_type == "cosine_similarity":
            distance = torch.nn.functional.cosine_similarity(v1, v2, dim=dim)
        if self.distance_type == "euclidean":
            distance = torch.dist(v1, v2, p=2)
        return distance.item()
    '''
    def tokens_to_encoder_repr(self, tok):
        # encoder_out=x,  # T x B x C    (eg. 20 x 1 x 512 - nr. tokens x batch.size x dim)
        # encoder_padding_mask=encoder_padding_mask,  # B x T   ### kann ich das doch für batch calculation benutzen?
        # encoder_embedding=encoder_embedding,  # B x T x C
        # encoder_states=encoder_states,  # List[T x B x C]
        enc_out = self.encoder.forward(tok, torch.tensor(tok.shape[1]))["encoder_out"][0].transpose(0, 1).squeeze()  # torch.Size([20, 512])
        sent_emb = self.emb_tok2sent(enc_out)  # torch.Size([512])
        return sent_emb
    '''

    @torch.no_grad()
    def forward_encoder(self, tokens, shape):
        print("shape encoder: ", shape)
        return self.encoder(tokens, shape)


    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_out=None):
        #TODO: use incremental states if the model has such
        decoder_out, extra = self.decoder.forward(tokens, encoder_out=encoder_out, features_only=True)
        #TODO: why here call forward explicitely?
        #print("Decoder Positions", self.decoder.embed_positions)
        return decoder_out

    @torch.no_grad()
    def forward_decoder_posminus(self, tokens, encoder_out=None):
        #TODO: use incremental states if the model has such
        decoder_copy = deepcopy(self.decoder)
        #if decoder_copy is self.decoder: print("same decoder")
        decoder_copy.embed_positions = None
        decoder_out, extra = decoder_copy.forward(tokens, encoder_out=encoder_out, features_only=True)
        #TODO: why here call forward explicitely?
        #print("Decoder Posminus Positions", decoder_copy.embed_positions)
        return decoder_out

    @torch.no_grad()
    def calculate_distances(self, sample, finalized):
        #self.set_custom_lm(custom_lm)
        prev_output_tokens = sample['net_input']['prev_output_tokens']
        gold_tgt_mask = (prev_output_tokens.ne(self.pad) & prev_output_tokens.ne(self.eos))

        sample_src_tok = sample["net_input"]["src_tokens"]
        sample_src_mask = (sample_src_tok.ne(self.pad) & sample_src_tok.ne(self.eos))

        # each sentence in the batch
        for i in range(sample["id"].shape[0]):
            #logger.info(f"\n\n#####################################i: {i}")
            ### Encoder Representations (src and gold_tgt)
            #### src:
            # src_len = sample["net_input"]["src_lengths"][i] # includes eos
            src_tok = sample_src_tok[i][sample_src_mask[i]].unsqueeze(0)  # torch.Size([1, 20])
            if self.remove_stopwords:
                src_tok = self.filter_stop_words(self.src_lang, src_tok)
            src_enc_out = self.forward_encoder(src_tok, torch.tensor(src_tok.shape[1])) #torch.Size([22, 1, 512])
            #test = self.encoder.forward(src_tok, torch.tensor(src_tok.shape[1]))["encoder_out"] ### ??? same ??? actually the cos. distance btw. sent_repr of both is 1.

            semb_enc_src = self.emb_tok2sent(src_enc_out["encoder_out"][0].transpose(0, 1).squeeze(0))  # torch.Size([512])
            logger.info(f'src_tok: {self.check_token2word(src_tok)}')
            logger.info(f'src_tok: {src_tok}   ---    shape: {src_tok.shape}')
            #print("src_tok:", src_tok)
            logger.debug(f'src - tok: {src_tok.shape}, enc_out: {src_enc_out["encoder_out"][0].shape}, semb: {semb_enc_src.shape}')


            #### gold:
            gold_tok = prev_output_tokens[i][gold_tgt_mask[i]].unsqueeze(0)
            if self.remove_stopwords:
                gold_tok = self.filter_stop_words(self.tgt_lang, gold_tok)
            # only for testing
            #if self.lm_model is not None:
            #    gold_tok_probs = self.check_lm(gold_tok)
            gold_enc_out = self.encoder.forward(gold_tok, torch.tensor(gold_tok.shape[1]))  # torch.Size([20, 512])
            semb_enc_gold = self.emb_tok2sent(gold_enc_out["encoder_out"][0].transpose(0, 1).squeeze(0))  # torch.Size([512])
            logger.info(f"gold_tok: {self.check_token2word(gold_tok)}")
            logger.info(f"gold_tok: {gold_tok}    ---   shape: {gold_tok.shape}")
            logger.debug(f"gold - tok: {gold_tok.shape}, enc_out: { gold_enc_out['encoder_out'][0].shape}, semb: {semb_enc_gold.shape}")

            ### Decoder Representations (src and gold_tgt)
            gold_dec_out = self.forward_decoder(gold_tok, encoder_out=src_enc_out)
            semb_dec_gold = self.emb_tok2sent(gold_dec_out.squeeze(0))
            logger.debug(f"gold - dec_out: {gold_dec_out.shape}, semb:{semb_dec_gold.shape}")

            #### src
            src_dec_out_gold_enc = self.forward_decoder(src_tok, encoder_out=gold_enc_out)
            semb_dec_src_enc_gold = self.emb_tok2sent(src_dec_out_gold_enc.squeeze(0))
            logger.debug(f"src - src_dec_out_gold_enc: {src_dec_out_gold_enc.shape}, semb:{semb_dec_src_enc_gold.shape}")


            src_dec_posplus_out_enc_none = self.forward_decoder(src_tok)
            semb_dec_src_posplus_enc_none = self.emb_tok2sent(src_dec_posplus_out_enc_none.squeeze(0))
            logger.debug(f"src - src_dec_posplus_out_enc_none: {src_dec_posplus_out_enc_none.shape}, semb:{semb_dec_src_posplus_enc_none.shape}")

            src_dec_posminus_out_enc_none = self.forward_decoder_posminus(src_tok)
            semb_dec_src_posminus_enc_none = self.emb_tok2sent(src_dec_posminus_out_enc_none.squeeze(0))
            logger.debug(f"src - src_dec_out_enc_src: {src_dec_posminus_out_enc_none.shape}, semb:{semb_dec_src_posminus_enc_none.shape}")

            src_dec_out_enc_src = self.forward_decoder(src_tok, encoder_out=src_enc_out)
            semb_dec_src_enc_src = self.emb_tok2sent(src_dec_out_enc_src.squeeze(0))
            logger.debug(f"src - src_dec_out_enc_src: {src_dec_out_enc_src.shape}, semb:{semb_dec_src_enc_src.shape}")


            hypos = finalized[i]
            for j in range(len(hypos)):
                #print("\n######j: ", j)
                data_sub = dict()
                #data_sub["beam"] = "hyp" + str(j)
                hypo = hypos[j]
                hypo_alignment = hypo["alignment"]
                #print("in distance_calculator - HYPO! alignment: ", hypo["alignment"])
                hypo_mask = (hypo["tokens"].ne(self.eos))
                hypo_tok = hypo["tokens"][hypo_mask].unsqueeze(0)
                if self.remove_stopwords:
                    hypo_tok = self.filter_stop_words(self.tgt_lang, hypo_tok)
                if self.lm_model is not None:
                    hypo_tok_ppl, hypo_tok_lm_score = self.check_lm(hypo_tok)
                    data_sub["hypo_score_lm"] = hypo_tok_lm_score
                    data_sub["hypo_ppl_lm"] = hypo_tok_ppl
                logger.info(f'hyp_tok {j}: {self.check_token2word(hypo["tokens"])}')
                logger.info(f'hyp_tok {j}: {hypo["tokens"]}   ---   shape: {hypo["tokens"].shape}')
                logger.info(f'hyp_tok_alignment {j}: {hypo["alignment"]}')
                #print("### hypo alignment len: ", len(hypo["alignment"]))
                #print("### hypo token len: ", len(hypo_tok[0]))

                '''
                m = [(src_tok.squeeze()[s],hypo_tok.squeeze()[t]) for s,t in hypo["alignment"] ]
                print("m: ", m)
                #print("m0: ", src_tok[0][0].unsqueeze(0) )
                m = [(self.check_token2word(src_tok[0][s].unsqueeze(0)), self.check_token2word(hypo_tok[0][t].unsqueeze(0))) for s, t in hypo["alignment"]]
                print("m: ", m)
                '''

                # Encoder Repr
                #print("<<< ERROR HERE: ", hypo_tok.shape, type(hypo_tok.shape), type(hypo_tok.shape[1]))
                hyp_enc_out = self.forward_encoder(hypo_tok, torch.tensor(hypo_tok.shape[1]))
                semb_enc_hyp = self.emb_tok2sent(hyp_enc_out["encoder_out"][0].transpose(0, 1).squeeze(0) )
                logger.debug(
                    f'hyp -hyp_enc_out["encoder_out"][0]: {hyp_enc_out["encoder_out"][0].shape}, semb_enc_hyp: {semb_enc_hyp.shape}')

                # Decoder Repr
                hyp_dec_out_enc_src = self.forward_decoder(hypo_tok, src_enc_out)
                semb_dec_hyp_enc_src = self.emb_tok2sent(hyp_dec_out_enc_src.squeeze(0))
                logger.debug(f"hyp - dec_out_enc_src: {hyp_dec_out_enc_src.shape}, semb_dec_out_enc_src: {semb_dec_hyp_enc_src.shape}")

                #### Distances
                # 1. Goal: dec(src)-dec(gold)
                data_sub["[GOAL]:dist-dec(src+enc:gold_tgt)-dec(hyp)"] = self.distance_funct(semb_dec_src_enc_gold, semb_dec_hyp_enc_src, alignment=hypo_alignment)   #######################
                logger.info(f'DIST - [GOAL]:dist-dec(src+enc:gold_tgt)-dec(hyp): {data_sub["[GOAL]:dist-dec(src+enc:gold_tgt)-dec(hyp)"]}')

                # 2. Try to approach Goal: dec(src)-dec(hyp)
                src_dec_out_enc_hyp = self.forward_decoder(src_tok, hyp_enc_out)
                semb_dec_src_enc_hyp = self.emb_tok2sent(src_dec_out_enc_hyp.squeeze(0))
                logger.debug(
                    f"hyp - src_dec_out_enc_hyp: {src_dec_out_enc_hyp.shape}, semb_dec_out_enc_src: {semb_dec_src_enc_hyp.shape}")

                data_sub["[try_to_approach_GOAL]:dist-dec(src+enc:hyp)-dec(hyp)"] = self.distance_funct(semb_dec_src_enc_hyp, semb_dec_hyp_enc_src, alignment=hypo_alignment)
                logger.info(
                    f'DIST - [try_to_approach_GOAL]:dist-dec(src+enc:hyp)-dec(hyp): {data_sub["[try_to_approach_GOAL]:dist-dec(src+enc:hyp)-dec(hyp)"]}')

                # 3.
                hyp_dec_posplus_out_enc_none = self.forward_decoder(hypo_tok)
                semb_dec_hyp_posplus_enc_none = self.emb_tok2sent(hyp_dec_posplus_out_enc_none.squeeze(0))
                logger.debug(
                    f"hyp - hyp_dec_posplus_out_enc_none: {hyp_dec_posplus_out_enc_none.shape}, semb_dec_hyp_posplus_enc_none: {semb_dec_hyp_posplus_enc_none.shape}")

                data_sub["dist-dec(src_noenc_posplus)-dec(hyp_noenc_posplus)"] = self.distance_funct(semb_dec_src_posplus_enc_none, semb_dec_hyp_posplus_enc_none, alignment=hypo_alignment)
                logger.info(
                    f'DIST - dist-dec(src_noenc_posplus)-dec(hyp_noenc_possemb_hyp_noenc_posminusplus): {data_sub["dist-dec(src_noenc_posplus)-dec(hyp_noenc_posplus)"]}')

                # 4.
                dec_hyp_noenc_posminus = self.forward_decoder_posminus(hypo_tok)
                semb_dec_hyp_posminus_enc_none = self.emb_tok2sent(dec_hyp_noenc_posminus.squeeze(0))
                logger.debug(
                    f"hyp - dec_hyp_noenc_posminus: {dec_hyp_noenc_posminus.shape}, semb_dec_hyp_posminus_enc_none: {semb_dec_hyp_posminus_enc_none.shape}")

                data_sub["dist-dec(src_noenc_posminus)-dec(hyp_noenc_posminus)"] = self.distance_funct(semb_dec_src_posminus_enc_none, semb_dec_hyp_posminus_enc_none, alignment=hypo_alignment)
                logger.info(
                    f'DIST - dist-dec(src_noenc_posminus)-dec(hyp_noenc_posminus): {data_sub["dist-dec(src_noenc_posminus)-dec(hyp_noenc_posminus)"]}')

                # 5.
                data_sub["dist-enc(src)-enc(hyp)"] = self.distance_funct(semb_enc_src, semb_enc_hyp, alignment=hypo_alignment)
                logger.info(
                    f'DIST - dist-enc(src)-enc(hyp): {data_sub["dist-enc(src)-enc(hyp)"]}')


                ### Unnecessary
                # 6. enc(src)-dec(hyp)
                data_sub["maybe-nosense-dist-enc(src)-dec(hyp)"] = self.distance_funct(semb_enc_src, semb_dec_hyp_enc_src, alignment=hypo_alignment)
                logger.info(f'DIST - maybe-nosense-dist-enc(src)-dec(hyp): {data_sub["maybe-nosense-dist-enc(src)-dec(hyp)"]}')

                # 7.
                data_sub["nosense-dist-dec(src_noenc_posplus)-dec(hyp)"] = self.distance_funct(semb_dec_src_posplus_enc_none, semb_dec_hyp_enc_src, alignment=hypo_alignment)
                logger.info(
                    f'DIST - nosense-dist-dec(src_noenc_posplus)-dec(hyp): {data_sub["nosense-dist-dec(src_noenc_posplus)-dec(hyp)"]}')
                # 8.
                data_sub["nosense-dist-dec(2args<-src)-dec(hyp)"] = self.distance_funct(semb_dec_src_enc_src, semb_dec_hyp_enc_src, alignment=hypo_alignment)
                logger.info(
                    f'DIST - nosense-dist-dec(2args<-src)-dec(hyp): {data_sub["nosense-dist-dec(2args<-src)-dec(hyp)"]}')

                # 9.
                semb_dec_hyp_enc_hyp = self.emb_tok2sent(self.forward_decoder(hypo_tok, hyp_enc_out).squeeze(0))
                logger.debug(
                    f"hyp - dec_hyp_enc_hyp: direkt, semb_dec_hyp_enc_hyp: {semb_dec_hyp_enc_hyp.shape}")
                data_sub["nosense-dist-dec(2args<-src)-dec(2args<-hyp)"] = self.distance_funct(semb_dec_src_enc_src, semb_dec_hyp_enc_hyp, alignment=hypo_alignment)
                logger.info(
                    f'DIST - nosense-dist-dec(2args<-src)-dec(2args<-hyp): {data_sub["nosense-dist-dec(2args<-src)-dec(2args<-hyp)"]}')

                ### proof of concept: dist hyp-gold - enc, dec, noenc-posplus, noenc-posminus

                poc = self.print_poc
                if (poc):
                # 10. hyp-gold enc
                    data_sub["poc-dict-enc(hyp)-enc(gold)"] = self.distance_funct(semb_enc_hyp, semb_enc_gold)
                    logger.info(
                        f'DIST - poc-dict-enc(hyp)-enc(gold): {data_sub["poc-dict-enc(hyp)-enc(gold)"]}')

                    # 11. hyp-gold dec
                    data_sub["poc-dist-dec(hyp)-dec(gold)"] = self.distance_funct(semb_dec_hyp_enc_src, semb_dec_gold)
                    logger.info(
                        f'DIST -poc-dist-dec(hyp)-dec(gold): {data_sub["poc-dist-dec(hyp)-dec(gold)"]}')

                    # 12. hyp-gold noenc-posplus
                    semb_dec_gold_posplus_enc_none = self.emb_tok2sent(self.forward_decoder(gold_tok).squeeze(0))
                    logger.debug(
                        f"hyp - dec_hyp_enc_hyp: direkt, semb_dec_hyp_enc_hyp: {semb_dec_gold_posplus_enc_none.shape}")
                    data_sub["poc-dist-dec(hyp_noenc_posplus)-dec(gold_noenc_posplus)"] = self.distance_funct(semb_dec_hyp_posplus_enc_none, semb_dec_gold_posplus_enc_none)
                    logger.info(
                        f'DIST - poc-dist-dec(hyp_noenc_posplus)-dec(gold_noenc_posplus): {data_sub["poc-dist-dec(hyp_noenc_posplus)-dec(gold_noenc_posplus)"]}')

                    # 13. hyp-gold noenc-posminus
                    semb_dec_gold_posminus_enc_none = self.emb_tok2sent(self.forward_decoder_posminus(gold_tok).squeeze(0))
                    data_sub["poc-dist-dec(hyp_noenc_posminus)-dec(gold_noenc_posminus)"] = self.distance_funct(semb_dec_hyp_posminus_enc_none, semb_dec_gold_posminus_enc_none)
                    logger.info(
                        f'DIST - poc-dist-dec(hyp_noenc_posminus)-dec(gold_noenc_posminus): {data_sub["poc-dist-dec(hyp_noenc_posminus)-dec(gold_noenc_posminus)"]}')

                finalized[i][j]["distances"] = data_sub
        return finalized