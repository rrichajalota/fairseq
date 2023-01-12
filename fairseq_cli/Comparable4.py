"""
Classes and methods used for training and extraction of parallel pairs
from a comparable dataset.
Author: Alabi Jesujoba
"""
#import tracemalloc
#import gc
import re
import itertools
import random
from collections import defaultdict
import torch
from fairseq.data import (
    MonolingualDataset,
    LanguagePairDataset
)
from fairseq.data.data_utils import load_indexed_dataset,numpy_seed,batch_by_size,filter_by_size
from fairseq.data.iterators import EpochBatchIterator, GroupedIterator
from fairseq import (
    checkpoint_utils, metrics, progress_bar, utils
)

def get_src_len(src, use_gpu, device=""):
    if use_gpu:
        if device=="mps":
            return torch.tensor([src.size(0)], device="mps") #.cuda()
        else:
            return torch.tensor([src.size(0)]).cuda()
    else:
        return torch.tensor([src.size(0)])

def indexPhraseData(phrases, dictionary, append_eos, reverse_order):
    tokens_list = []
    sizes = []
    for line in phrases:
        # self.lines.append(line.strip('\n'))
        tokens = dictionary.encode_line(
            line, add_if_not_exist=False,
            append_eos=append_eos, reverse_order=reverse_order,
        ).long()
        tokens_list.append(tokens)
        sizes.append(len(tokens))
    return tokens_list, sizes

#this method is to remove spaces added within strings when dict.string is used.
#it removed remove spaces between characters and consecutive spaces
def removeSpaces(s):
  k = re.sub(' (?! )',"",s)
  k = re.sub(' +', ' ', k)
  return k

# get noun phrases with tregex using stanza
def noun_phrases(_client, _text):
    pattern = "NP"
    matches = _client.tregex(_text, pattern)
    s = "\n".join(["\t" + sentence[match_id]['spanString'] for sentence in matches['sentences'] for match_id in sentence])
    phrases = [x.strip() for x in s.split('\n\t')]
    return  phrases

def extract_phrase(tree_str, label):
    phrases = []
    trees = Tree.fromstring(tree_str)
    for tree in trees:
        for subtree in tree.subtrees():
            if subtree.label() == label:
                t = subtree
                t = ' '.join(t.leaves())
                phrases.append(t)

    return phrases

def read_vocabulary(vocab_file, threshold=20):
    """read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
    """
    vocabulary = set()

    for line in vocab_file:
        word, freq = line.strip('\r\n ').split(' ')
        freq = int(freq)
        if threshold == None or freq >= threshold:
            vocabulary.add(word)

    return vocabulary


class PhraseBank():
    """
    Class that saves the sentence pairs from which we want to extract phrases
    Args:
        candidate(tuple(src,tgt,score))
        args(argparse.Namespace): option object

    """

    def __init__(self, tasks, phrase_length):
        self.tasks = tasks
        self.sourcesent = set()
        self.targetsent = set()
        self.phrase_length = phrase_length
        self.lsrc = []
        self.ltgt = []
        self.nlp_src = None
        self.nlp_tgt = None
        '''self.use_gpu = False
        if args.cpu == False:
            self.use_gpu = True
        else:
            self.use_gpu = False
        '''

    def add_example(self, src, tgt):
        """ Add an example from a batch to the PairBank (self.pairs).
        Args:
            src(torch.Tensor): src sequence (size(seq))
            tgt(torch.Tensor): tgt sequence(size(tgt))
            fields(list(str)): list of keys of fields
        """
        # Get example from src/tgt and remove original padding
        self.sourcesent.add(str(src))
        self.targetsent.add(str(tgt))

    def getexamples(self):
        return self.sourcesent, self.targetsent

    def getexampleslen(self):
        return len(self.sourcesent), len(self.targetsent)

    def remove_from_phrase_candidates(self, seq, side):
        hash_key = hash(str(seq))
        # print(len(self.bt_candidates))
        if side == 'src':
            self.lsrc.extend([x for x in self.sourcesent if x[0] == hash_key])
            self.sourcesent = set([x for x in self.sourcesent if x[0] != hash_key])
        elif side == 'tgt':
            self.ltgt.extend([x for x in self.targetsent if x[0] == hash_key])
            self.targetsent = set([x for x in self.targetsent if x[0] != hash_key])
        # print(len(self.bt_candidates))
        # print('........')
        return None

    def convert2string(self, side):
        lstString = []
        if side == 'src':
            lstString = [removeSpaces(' '.join(self.tasks.src_dict.string(x[1], bpe_symbol='@@ '))).replace("<fr>",'').strip() for x in self.sourcesent]
        elif side == 'tgt':
            lstString = [removeSpaces(' '.join(self.tasks.tgt_dict.string(x[1], bpe_symbol='@@ '))).replace("<en>",'').strip() for x in self.targetsent]
            self.resetData()
        return lstString

    def resetSource(self):
        self.sourcesent = set()

    def resetTarget(self):
        self.targetsent = set()

    def setparsers(self, nlp_src, nlp_tgt):
        self.nlp_src = nlp_src
        self.nlp_tgt = nlp_tgt

    def setclients(self, nlp_src, nlp_tgt):
        self.client_src = nlp_src
        self.client_tgt = nlp_tgt

    def bpe(self, src, tgt):
        self.srcbpe = src
        self.tgtbpe = tgt

    def setLang(self, s, t):
        self.s = s
        self.t = t

    def extractPhrasesSNL(self, sentences, side='src'):
        if side == 'src':
            #phrases = [list(set(extract_phrase(self.nlp_src.parse(x), 'NP'))) for x in sentences]
            #phrases = [noun_phrases(self.client_src,x,_annotators="tokenize,ssplit,pos,lemma,parse") for x in sentences]
            phrases = [noun_phrases(self.client_src,x) for x in sentences] #,_annotators="tokenize,ssplit,pos,lemma,parse"
        elif side == 'tgt':
            #phrases = [list(set(extract_phrase(self.nlp_tgt.parse(x), 'NP'))) for x in sentences]
            phrases = [noun_phrases(self.client_tgt,x) for x in sentences] #,_annotators="tokenize,ssplit,pos,lemma,parse"


        phrases = list(itertools.chain(*phrases))
        if side == 'src':
            return ["<"+self.t+"> "+self.srcbpe.process_line(item) for item in phrases if len(item.split()) >= self.phrase_length]
        elif side == 'tgt':
            #print("From target", ["<"+self.s+"> "+self.tgtbpe.process_line(item) for item in phrases if len(item.split()) >= self.phrase_length] )
            return ["<"+self.s+"> "+self.tgtbpe.process_line(item) for item in phrases if len(item.split()) >= self.phrase_length]

    def resetData(self):
        self.sourcesent = set()
        self.targetsent = set()


class PairBank():
    """
    Class that saves and prepares parallel pairs and their resulting
    batches.
    Args:
        batch_size(int): number of examples in a batch
        opt(argparse.Namespace): option object
    """

    def __init__(self, batcher, args):
        self.pairs = []
        self.index_memory = set()
        self.batch_size = args.max_sentences
        self.batcher = batcher
        self.use_gpu = False
        self.mps = False
        self.cuda = False
        if args.cpu == False:
            self.use_gpu = True
            if torch.backends.mps.is_available():
                self.mps = True
                self.mps_device = torch.device("mps")
            else:
                self.cuda = True
        else:
            self.use_gpu = False
        self.update_freq = args.update_freq
        self.explen = self.batch_size * self.update_freq[-1]


    def removePadding(side):
        """ Removes original padding from a sequence.
        Args:
            side(torch.Tensor): src/tgt sequence (size(seq))
        Returns:
            side(torch.Tensor): src/tgt sequence without padding
        NOTE: This only works as long as PAD_ID==1!
        """
        # Get indexes of paddings in sequence
        padding_idx = (side == 1).nonzero()
        # If there is any padding, cut sequence from first occurence of a pad
        if padding_idx.size(0) != 0:
            first_pad = padding_idx.data.tolist()[0][0]
            side = side[:first_pad]
        return side

    def add_example(self, src, tgt):
        """ Add an example from a batch to the PairBank (self.pairs).
        Args:
            src(torch.Tensor): src sequence (size(seq))
            tgt(torch.Tensor): tgt sequence(size(tgt))
            fields(list(str)): list of keys of fields
        """
        # Get example from src/tgt and remove original padding
        src = PairBank.removePadding(src)
        tgt = PairBank.removePadding(tgt)
        if self.mps:
            src_length = get_src_len(src, self.use_gpu, device="mps")
            tgt_length = get_src_len(tgt, self.use_gpu, device="mps")
        else:
            src_length = get_src_len(src, self.use_gpu)
            tgt_length = get_src_len(tgt, self.use_gpu)
        index = None
        # Create CompExample object holding all information needed for later
        # batch creation.
        # print((src,tgt))
        example = CompExample(index, src, tgt, src_length, tgt_length, index)
        # dataset, src, tgt, src_length, tgt_length, index
        # Add to pairs
        self.pairs.append(example)
        # Remember unique src-tgt combination
        self.index_memory.add(hash((str(src), str(tgt))))
        return None

    def contains_batch(self):
        """Check if enough parallel pairs found to create a batch.
        """
        return (len(self.pairs) >= self.explen)

    def no_limit_reached(self, src, tgt):
        """ Check if no assigned limit of unique src-tgt pairs is reached.
        Args:
            src(torch.Tensor): src sequence (size(seq))
            tgt(torch.Tensor): tgt sequence(size(tgt))
        """
        # src = PairBank.removePadding(src)
        # tgt = PairBank.removePadding(tgt)
        return (hash((str(src), str(tgt))) in self.index_memory or len(self.index_memory) < self.limit)

    def get_num_examples(self):
        """Returns batch size if no maximum number of extracted parallel data
        used for training is met. Otherwise returns number of examples that can be yielded
        without exceeding that maximum.
        """
        if len(self.pairs) < self.explen:
            return len(self.pairs)
        return self.explen

    def yield_batch(self):
        """ Prepare and yield a new batch from self.pairs.
        Returns:
            batch(fairseq.data.LanguagePairDataset): batch of extracted parallel data
        """
        src_examples = []
        tgt_examples = []
        src_lengths = []
        tgt_lengths = []
        indices = []
        num_examples = self.get_num_examples()

        # Get as many examples as needed to fill a batch or a given limit
        random.shuffle(self.pairs)
        for ex in range(num_examples):
            example = self.pairs.pop()
            src_examples.append(example.src)
            tgt_examples.append(example.tgt)
            src_lengths.append(example.src_length)
            tgt_lengths.append(example.tgt_length)
            indices.append(example.index)

        dataset = None
        # fields = CompExample.get_fields()
        batch = self.batcher.create_batch(src_examples, tgt_examples, src_lengths, tgt_lengths)
        # enumerate to yield batch here
        return batch


class CompExample():
    """
    Class that stores the information of one parallel data example.
    Args:
        dataset(fairseq.data): dataset object
        src(torch.Tensor): src sequence (size(seq))
        tgt(torch.Tensor): tgt sequence (size(seq))
        src_length(torch.Tensor): the length of the src sequence (size([]))
        index(torch.Tensor): the index of the example in dataset
    """
    # These should be the same for all examples (else: consistency problem)
    _dataset = None

    def __init__(self, dataset, src, tgt, src_length, tgt_length, index):
        self.src = src
        self.tgt = tgt
        self.src_length = src_length
        self.tgt_length = tgt_length
        self.index = index

        if CompExample._dataset == None:
            CompExample._dataset = dataset


class BatchCreator():
    def __init__(self, task, args):
        self.task = task
        self.args = args

    def create_batch(self, src_examples, tgt_examples, src_lengths, tgt_lengths, no_target=False):
        """ Creates a batch object from previously extracted parallel data.
                Args:
                    src_examples(list): list of src sequence tensors
                    tgt_examples(list): list of tgt sequence tensors
                    src_lenths(list): list of the lengths of each src sequence
                    tgt_lenths(list): list of the lengths of each tgt sequence
                    indices(list): list of indices of example instances in dataset
                    dataset(fairseq.data): dataset object
                Returns:
                    batch(fairseq.data.LanguagePairDataset): batch object
                """
        pairData = LanguagePairDataset(
            src_examples, src_lengths, self.task.src_dict,
            tgt_examples, tgt_lengths, self.task.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

        with numpy_seed(self.args.seed):
            indices = pairData.ordered_indices()

        batch_sampler = batch_by_size(indices, pairData.num_tokens, max_sentences=self.args.max_sentences,
                                                 required_batch_size_multiple=self.args.required_batch_size_multiple, )
        itrs = EpochBatchIterator(dataset=pairData, collate_fn=pairData.collater, batch_sampler=batch_sampler,
                                            seed=self.args.seed, epoch=0)
        indices = None
        return itrs


class Comparable():
    """
    Class that controls the extraction of parallel sentences and manages their
    storage and training.
    Args:
        model(:py:class:'fairseq.models'):
            translation model used for extraction and training
        trainer(:obj:'fairseq.trainer'):
            trainer that controlls the training process
        fields(dict): fields and vocabulary
        logger(logging.RootLogger):
            logger that reports information about extraction and training
        opt(argparse.Namespace): option object
    """

    def __init__(self, model, trainer, task, args):
        self.sim_measure = args.sim_measure
        self.threshold = args.threshold
        self.model_name = args.model_name
        self.save_dir = args.save_dir
        self.use_phrase = False #args.use_phrase
        #self.model = trainer.get_model().encoder
        self.usepos =  args.usepos
        print("Use positional encoding = ", self.usepos)
        self.trainer = trainer
        print(f"self.trainer: {self.trainer}")
        self.task = self.trainer.task
        self.encoder = self.trainer.get_model().encoder
        print(f"self.encoder: {self.encoder}")
        self.batcher = BatchCreator(task, args)
        self.similar_pairs = PairBank(self.batcher, args)
        self.accepted = 0
        self.accepted_limit = 0
        self.declined = 0
        self.total = 0
        self.args = args
        self.comp_log = args.comp_log
        self.cove_type = args.cove_type
        self.update_freq = args.update_freq
        self.k = 4
        self.trainstep = 0
        self.second = args.second
        self.representations = args.representations
        self.task = task
        self.write_dual = args.write_dual
        self.no_swaps = False  # args.no_swaps
        self.symmetric = args.symmetric
        self.add_noise = args.add_noise
        self.use_bt = False #args.use_bt
        self.stats = None
        self.progress = None
        self.src, self.tgt = "tr", "og" #args.source_lang, args.target_lang
        self.use_gpu = False
        self.mps = False
        self.cuda = False
        self.mps_device = None
        print(f"args.cpu: {args.cpu}")
        if args.cpu == False:
            self.use_gpu = True
            if torch.backends.mps.is_available():
                self.mps = True
                self.mps_device = torch.device("mps")
                self.div = 2 * torch.tensor(self.k).to(self.mps_device) #.cuda()
            else:
                self.div = 2 * torch.tensor(self.k).cuda()
                self.cuda = True
        else:
            self.use_gpu = False
            self.div = 2 * torch.tensor(self.k) #, device="mps") #.cuda()
        print(f"use_gpu: {self.use_gpu}, self.mps: {self.mps}")
        

        if self.use_phrase == True and self.args.phrase_method == 'stanford':
            from subword_nmt.apply_bpe import BPE
            # from stanfordcorenlp import StanfordCoreNLP
            from nltk.tree import Tree
            import stanza
            from stanza.server import CoreNLPClient
            self.phrases = PhraseBank(self.task, args.phrase_length)

            english_properties = {"annotators": "tokenize,ssplit,mwt,pos,lemma,parse", "tokenize.language": "en",
                                  "pos.model": "edu/stanford/nlp/models/pos-tagger/english-left3words-distsim.tagger",
                                  "parse.model": "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"}
            self.client_eng = CoreNLPClient(properties=english_properties,
                                            classpath='/raid/data/alabi/data/FMT2/fairseq/fairseq_cli/stanford-corenlp-full-2020-04-20/*',
                                            timeout=70000, memory='6G', be_quiet=False, max_char_length=100000,
                                            endpoint='http://localhost:9937')

            # French properties for 4.0.0
            french_properties = {"annotators": "tokenize,ssplit,mwt,pos,lemma,parse", "tokenize.language": "fr",
                                 "mwt.mappingFile": "edu/stanford/nlp/models/mwt/french/french-mwt.tsv",
                                 "mwt.pos.model": "edu/stanford/nlp/models/mwt/french/french-mwt.tagger",
                                 "mwt.statisticalMappingFile": "edu/stanford/nlp/models/mwt/french/french-mwt-statistical.tsv",
                                 "mwt.preserveCasing": "false",
                                 "pos.model": "edu/stanford/nlp/models/pos-tagger/french-ud.tagger",
                                 "parse.model": "edu/stanford/nlp/models/srparser/frenchSR.beam.ser.gz"}

            self.client_fr = CoreNLPClient(properties=french_properties, timeout=70000, memory='4G',
                                               classpath='/raid/data/alabi/data/FMT2/fairseq/fairseq_cli/stanford-corenlp-full-2020-04-20/*',
                                               be_quiet=False, max_char_length=100000, endpoint='http://localhost:9938')
            self.client_eng.start()
            self.client_fr.start()
            self.phrases.setclients(self.client_eng, self.client_fr)
            # create the BPE vocabulary for source and target
            self.src_vocab = read_vocabulary(open(args.src_vocab, 'r'), 20)  # read_vocabulary(args.src_vocab, 20)
            self.tgt_vocab = read_vocabulary(open(args.tgt_vocab, 'r'), 20)  # read_vocabulary(args.tgt_vocab, 20)
            self.bpecodes = open(args.bpecodes, 'r')
            self.bpesrc = BPE(self.bpecodes, vocab=self.src_vocab)
            self.bpetgt = BPE(self.bpecodes, vocab=self.tgt_vocab)
            # self.phrases.setparsers(self.nlp_src, self.nlp_tgt)
            print("Got the source BPE ", self.bpesrc)
            self.phrases.bpe(self.bpesrc, self.bpetgt)
            self.phrases.setLang(self.src, self.tgt)

    def getstring(self, vec, dict):
        words = dict.string(vec)
        return removeSpaces(' '.join(words))

    def write_sentence(self, src, tgt, status, score=None):
        """
        Writes an accepted parallel sentence candidate pair to a file.
        Args:
            src(torch.tensor): src sequence (size(seq))
            tgt(torch.tensor): tgt sequence (size(seq))
            status(str): ['accepted', 'accepted-limit', 'rejected']
            score(float): score of the sentence pair
        """
        src_words = self.task.src_dict.string(src)
        tgt_words = self.task.tgt_dict.string(tgt)
        out = 'src: {}\ttgt: {}\tsimilarity: {}\tstatus: {}\n'.format(removeSpaces(' '.join(src_words)),
                                                                      removeSpaces(' '.join(tgt_words)), score, status)
        if 'accepted' in status:
            self.accepted_file.write(out)
            # print(out)
        elif 'phrase' in status:
            self.accepted_phrase.write(out)
        elif status == 'embed_only':
            with open(self.embed_file, 'a', encoding='utf8') as f:
                f.write(out)
        elif status == 'hidden_only':
            with open(self.hidden_file, 'a', encoding='utf8') as f:
                f.write(out)
        return None

    def extract_parallel_sents(self, candidates, candidate_pool, phrasese=False):
        """
        Extracts parallel sentences from candidates and adds them to the
        PairBank (secondary filter).
        Args:
            candidates(list(tuple(torch.Tensor...)): list of src-tgt candidates
            candidate_pool(list(hash)): list of hashed C_e candidates
        """
        # print("extract parallel")
        for candidate in candidates:
            candidate_pair = hash((str(candidate[0]), str(candidate[1])))
            # For dual representation systems...
            # print("Dual representation checking")
            if candidate_pool:
                # ...skip C_h pairs not in C_e (secondary filter)
                if self.in_candidate_pool(candidate, candidate_pool) == False:
                    self.declined += 1
                    self.total += 1
                    if self.write_dual:
                        self.write_sentence(candidate[0], candidate[1],
                                            'hidden_only', candidate[2])
                    continue
            '''if self.no_swaps:
                swap = False
            # Swap src-tgt direction randomly
            else:
                swap = np.random.randint(2)
            if swap:
                src = candidate[1]
                tgt = candidate[0]
            else:
                src = candidate[0]
                tgt = candidate[1]'''

            src = candidate[0]
            tgt = candidate[1]
            score = candidate[2]

            # Apply threshold (single-representation systems only)
            if score >= self.threshold:
                # print("Score is greater than threshold")
                # Check if no maximum of allowed unique accepted pairs reached
                # if self.similar_pairs.no_limit_reached(src, tgt):
                # Add to PairBank
                self.similar_pairs.add_example(src, tgt)
                self.write_sentence(removePadding(src), removePadding(tgt), 'accepted', score)
                self.accepted += 1
                if self.symmetric:
                    self.similar_pairs.add_example(tgt, src)
                    #self.write_sentence(tgt, src, 'accepted', score)


                if self.use_phrase and phrasese is False:
                    print("checking phrases to remove.......")
                    src_rm = removePadding(src)
                    self.phrases.remove_from_phrase_candidates(src_rm, 'src')
                    tgt_rm = removePadding(tgt)
                    self.phrases.remove_from_phrase_candidates(tgt_rm, 'tgt')
                    # write the accepted phrases to file
                if self.use_phrase and phrasese is True and self.args.write_phrase:
                    self.write_sentence(removePadding(src), removePadding(tgt), 'phrase', score)

                '''if self.add_noise:
                    noisy_src = self.apply_noise(src, tgt)
                    self.similar_pairs.add_example(noisy_src, tgt)
                    self.write_sentence(noisy_src, tgt, 'accepted-noise', score)
                if self.symmetric:
                    self.similar_pairs.add_example(tgt, src)
                    self.write_sentence(tgt, src, 'accepted', score)
                    if self.add_noise:
                        noisy_tgt = self.apply_noise(tgt, src)
                        self.similar_pairs.add_example(noisy_tgt, src)
                        self.write_sentence(noisy_tgt, src, 'accepted-noise', score)
                self.accepted += 1
                if self.use_bt:
                    self.remove_from_bt_candidates(src)
                    self.remove_from_bt_candidates(tgt)
                else:
                    self.accepted_limit += 1
                    self.write_sentence(src, tgt, 'accepted-limit', score)'''
            else:
                # print("threshold not met!!!")
                self.declined += 1
            self.total += 1

        return None

    def write_embed_only(self, candidates, cand_embed):
        """ Writes C_e scores to file (if --write-dual is set).
        Args:
            candidates(list): list of src, tgt pairs (C_h)
            cand_embed(list): list of src, tgt pairs (C_e)
        """
        candidate_pool = set([hash((str(c[0]), str(c[1]))) for c in candidates])

        for candidate in cand_embed:
            candidate_pair = hash((str(candidate[0]), str(candidate[1])))
            # Write statistics only if C_e pair not in C_h
            if candidate_pair not in candidate_pool:
                src = candidate[0]
                tgt = candidate[1]
                score = candidate[2]
                self.write_sentence(src, tgt, 'embed_only', score)

    def score_sents(self, src_sents, tgt_sents):
        """ Score source and target combinations.
        Args:
            src_sents(list(tuple(torch.Tensor...))):
                list of src sentences in their sequential and semantic representation
            tgt_sents(list(tuple(torch.Tensor...))): list of tgt sentences
        Returns:
            src2tgt(dict(dict(float))): dictionary mapping a src to a tgt and their score
            tgt2src(dict(dict(float))): dictionary mapping a tgt to a src and their score
            similarities(list(float)): list of cosine similarities
            scores(list(float)): list of scores
        """
        src2tgt = defaultdict(dict)
        tgt2src = defaultdict(dict)
        similarities = []
        scores = []

        #print("At the point of unzipping the list of tuple....................")
        #unzip the list ot tiples to have two lists of equal length each sent, repre
        srcSent, srcRep = zip(*src_sents)
        tgtSent, tgtRep = zip(*tgt_sents)

        #print("Stacking the representations to cuda....................")
        #stack the representation list into a tensor and use that to compute the similarity
        if self.mps:
            srcRp=torch.stack(srcRep).to(self.mps_device) #.cuda()
            tgtRp=torch.stack(tgtRep).to(self.mps_device) #.cuda()
        elif self.cuda:
            srcRp=torch.stack(srcRep).cuda()
            tgtRp=torch.stack(tgtRep).cuda()
        else:
            srcRp = torch.stack(srcRep)
            tgtRp = torch.stack(tgtRep)

        print(f"tgtRp: {tgtRp}")

        # Return cosine similarity if that is the scoring function
        if self.sim_measure == 'cosine':
            matx = self.sim_matrix(srcRp, tgtRp)
            for i in range(len(srcSent)):
                for j in range(len(tgtSent)):
                    if srcSent[i][0] == tgtSent[j][0]:
                        continue
                    src2tgt[srcSent[i]][tgtSent[j]] = matx[i][j].tolist()
                    tgt2src[tgtSent[j]][srcSent[i]] = matx[i][j].tolist()
                    similarities.append(matx[i][j].tolist())
            return src2tgt, tgt2src, similarities, similarities
        else:
            sim_mt, sumDistSource, sumDistTarget = self.sim_matrix(srcRp, tgtRp)
            print(f"sumDistSource: {sumDistSource}")
            print(f"sim_mt: {sim_mt}")
            for i in range(len(srcSent)):
                for j in range(len(tgtSent)):
                    if srcSent[i][0] == tgtSent[j][0]:
                        continue
                    tgt2src[tgtSent[j]][srcSent[i]] = src2tgt[srcSent[i]][tgtSent[j]] = sim_mt[i][j].tolist() / (sumDistSource[i].tolist() + sumDistTarget[j].tolist())
                    #tgt2src[tgtSent[j]][srcSent[i]] = sim_mt[i][j].tolist() / (sumDistTarget[j].tolist() + sumDistSource[i].tolist())
                    #similarities.append(sim_mt[i][j].tolist() )

            # Get list of scores for statistics
        '''for src in list(src2tgt.keys()):
            scores += list(src2tgt[src].values())'''
        return src2tgt, tgt2src, similarities, scores

    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        print(f"sim_mt len in sim_matrix: {len(sim_mt)}")
        del a_n, b_n, a_norm, b_norm
        if self.sim_measure == 'cosine':
            return sim_mt
        nearestSrc = torch.topk(sim_mt, self.k, dim=1, largest=True, sorted=False, out=None)
        #sumDistSource = torch.sum(nearestSrc[0], 1)
        print(f"nearestSrc: {nearestSrc}")
        nearestTgt = torch.topk(sim_mt, self.k, dim=0, largest=True, sorted=False, out=None)
        #sumDistTarget = torch.sum(nearestTgt[0], 0)
        print(f"nearestTgt: {nearestTgt}")
        print(f"self.div: {self.div}")

        return sim_mt, torch.sum(nearestSrc[0], 1)/self.div, torch.sum(nearestTgt[0], 0)/self.div


    def get_article_coves(self, article, representation='memory', mean=False,  side='phr', use_phrase=False):
        """ Get representations (C_e or C_h) for sentences in a document.
        Args:
            article(inputters.OrderedIterator): iterator over sentences in document
            representation(str): if 'memory', create C_h; if 'embed', create C_e
            fast(boolean): if true, only look at first batch in article
            mean(boolean): if true, use mean over time-step representations; else, sum
        Returns:
            sents(list(tuple(torch.Tensor...))):
                list of sentences in their sequential (seq) and semantic representation (cove)
        """
        sents = []
        print("inside get_article_coves")
        #for k in article:#tqdm(article):
        print("next(article)")
        # print(f"next(article): {next(article)}")
        for k in article:
            print("inside article!")
            sent_repr = None
            if self.args.modeltype == "lstm":  # if the model architecture is LSTM
                lengths = k['net_input']['src_lengths']
                texts = k['net_input']['src_tokens']
                ordered_len, ordered_idx = lengths.sort(0, descending=True)
                if self.use_gpu and self.mps:
                    texts = texts[ordered_idx].to(self.mps_device)
                    ordered_len = ordered_len.to(self.mps_device)
                elif self.use_gpu and self.cuda:
                    texts = texts[ordered_idx].cuda()
                    ordered_len = ordered_len.cuda()
                else:
                    texts = texts[ordered_idx]
                with torch.no_grad():
                    output = self.encoder.forward(texts, ordered_len) # texts.cuda()

                if representation == 'memory':
                    sent_repr = output['encoder_out'][1].squeeze()
                    print("In the lstm representation",sent_repr)
                elif representation == 'embed':
                    print("Collecting Embedding")
                    hidden_embed = output['encoder_out'][0]
                    # print(hidden_embed)tr
                    if mean:
                        sent_repr = torch.mean(hidden_embed, dim=0)
                    else:
                        sent_repr = torch.sum(hidden_embed, dim=0)
            elif self.args.modeltype == "transformer":
                print("In the transformer representation")
                if representation == 'memory':
                    with torch.no_grad():
                        # print(f"k['net_input']['src_tokens']: {k['net_input']['src_tokens']}")
                        # print(f"k['net_input']['src_lengths']: {k['net_input']['src_lengths']}")
                        # encoderOut = self.encoder.forward(k['net_input']['src_tokens'].cuda(),
                        #                                   k['net_input']['src_lengths'].cuda())
                        if self.use_gpu and self.mps:
                            print("going into encoder forward")
                            encoderOut = self.encoder.forward(k['net_input']['src_tokens'].to(self.mps_device),
                                                            k['net_input']['src_lengths'].to(self.mps_device))
                        elif self.use_gpu and self.cuda:
                            print("going into encoder forward")
                            encoderOut = self.encoder.forward(k['net_input']['src_tokens'].cuda(),
                                                            k['net_input']['src_lengths'].cuda())
                        else:
                            encoderOut = self.encoder.forward(k['net_input']['src_tokens'],
                                                          k['net_input']['src_lengths'])
                    hidden_embed = getattr(encoderOut, 'encoder_out')  # T x B x C
                    if mean:
                        sent_repr = torch.mean(hidden_embed, dim=0)
                    else:
                        sent_repr = torch.sum(hidden_embed, dim=0)
                elif representation == 'embed':
                    with torch.no_grad():
                        # print(f"k['net_input']['src_tokens']: {k['net_input']['src_tokens']}")
                        # print(f"k['net_input']['src_lengths']: {k['net_input']['src_lengths']}")
                        print("going into encoder forward emb")
                        if self.usepos:
                            if self.use_gpu and self.mps:
                                input_emb,_ = self.encoder.forward_embedding(k['net_input']['src_tokens'].to(self.mps_device)) 
                            elif self.use_gpu and self.cuda:
                                input_emb,_ = self.encoder.forward_embedding(k['net_input']['src_tokens'].cuda()) 
                            else:
                                input_emb,_ = self.encoder.forward_embedding(k['net_input']['src_tokens']) 
                        else:
                            if self.use_gpu and self.mps:
                                _, input_emb = self.encoder.forward_embedding(k['net_input']['src_tokens'].to(self.mps_device)) # .to(self.mps_device)
                            elif self.use_gpu and self.cuda:
                                _, input_emb = self.encoder.forward_embedding(k['net_input']['src_tokens'].cuda())
                            else:
                                 _, input_emb = self.encoder.forward_embedding(k['net_input']['src_tokens']) 
                        print(f"type(input_emb): {type(input_emb)}")
                        
                        if self.mps:
                            input_emb = input_emb.to(self.mps_device)
                        if self.cuda:
                            input_emb = input_emb.cuda()

                    #input_emb = getattr(encoderOut, 'encoder_embedding')  # B x T x C
                    print(f"type(input_emb): {type(input_emb)}")
                    input_emb = input_emb.transpose(0, 1)
                    if mean:
                        sent_repr = torch.mean(input_emb, dim=0)
                    else:
                        sent_repr = torch.sum(input_emb, dim=0)
            if self.args.modeltype == "transformer":
                for i in range(k['net_input']['src_tokens'].shape[0]):
                    sents.append((k['net_input']['src_tokens'][i], sent_repr[i]))
                    if side == 'src' and use_phrase is True:
                        st = removePadding(k['net_input']['src_tokens'][i])
                        self.phrases.sourcesent.add((hash(str(st)), st))
                    elif side == 'tgt' and use_phrase is True:
                        st = removePadding(k['net_input']['src_tokens'][i])
                        self.phrases.targetsent.add((hash(str(st)), st))
            elif self.args.modeltype == "lstm":
                for i in range(texts.shape[0]):
                    sents.append((texts[i], sent_repr[i]))

        print(f"len(sents): {len(sents)}")
        return sents

    def get_comparison_pool(self, src_embeds, tgt_embeds):
        """ Perform scoring and filtering for C_e (in dual representation system)
        Args:
            src_embeds(list): list of source embeddings (C_e)
            tgt_embeds(list): list of target embeddings (C_e)
        Returns:
            candidate_pool(set): set of hashed src-tgt C_e pairs
            candidate_embed(list): list of src-tgt C_e pairs
        """
        # Scoring
        src2tgt_embed, tgt2src_embed, _, _ = self.score_sents(src_embeds, tgt_embeds)
        # Filtering (primary filter)
        candidates_embed = self.filter_candidates(src2tgt_embed, tgt2src_embed)
        # Create set of hashed pairs (for easy comparison in secondary filter)
        set_embed = set([hash((str(c[0]), str(c[1]))) for c in candidates_embed])
        candidate_pool = set_embed
        return candidate_pool, candidates_embed

    def in_candidate_pool(self, candidate, candidate_pool):
        candidate_pair = hash((str(candidate[0]), str(candidate[1])))
        # For dual representation systems...
        # ...skip C_h pairs not in C_e (secondary filter)
        if candidate_pair in candidate_pool:
            return True
        return False

    def filter_candidates(self, src2tgt, tgt2src, second=False):
        """ Filter candidates (primary filter), such that only those which are top candidates in
        both src2tgt and tgt2src direction pass.
        Args:
            src2tgt(dict(dict(float))): mapping src sequence to tgt sequence and score
            tgt2src(dict(dict(float))): mapping tgt sequence to src sequence and score
            second(boolean): if true, also include second-best candidate for src2tgt direction
                (medium permissibility mode only)
        Returns:
            candidates(list(tuple(torch.Tensor...)): list of src-tgt candidates
        """
        src_tgt_max = set()
        tgt_src_max = set()
        src_tgt_second = set()
        tgt_src_second = set()

        # For each src...
        for src in list(src2tgt.keys()):
            toplist = sorted(src2tgt[src].items(), key=lambda x: x[1], reverse=True)
            # ... get the top scoring tgt
            max_tgt = toplist[0]
            # Get src, tgt and score
            src_tgt_max.add((src, max_tgt[0], max_tgt[1]))
            if second:
                # If high permissibility mode, also get second-best tgt
                second_tgt = toplist[1]
                src_tgt_second.add((src, second_tgt[0], second_tgt[1]))

        # For each tgt...
        for tgt in list(tgt2src.keys()):
            toplist = sorted(tgt2src[tgt].items(), key=lambda x: x[1], reverse=True)
            # ... get the top scoring src
            max_src = toplist[0]
            tgt_src_max.add((max_src[0], tgt, max_src[1]))

        if second:
            # Intersection as defined in medium permissibility mode
            src_tgt = (src_tgt_max | src_tgt_second) & tgt_src_max
            candidates = list(src_tgt)
            return candidates

        # Intersection as defined in low permissibility
        print("Length of s2t max",len(src_tgt_max))
        print("Length of t2s max", len(tgt_src_max))
        print("Intersection = ",list(src_tgt_max & tgt_src_max))
        candidates = list(src_tgt_max & tgt_src_max)
        return candidates

    def _get_iterator(self, sent, dictn, max_position, epoch, fix_batches_to_gpus=False):
        """
        Creates an iterator object from a text file.
        Args:
            path(str): path to text file to process
        Returns:
            data_iter(.EpochIterator): iterator object
        """
        # get indices ordered by example size
        with numpy_seed(self.args.seed):
            indices = sent.ordered_indices()
        # filter out examples that are too large
        max_positions = (max_position)
        if max_positions is not None:
            indices = filter_by_size(indices, sent, max_positions, raise_exception=(not True), )
        # create mini-batches with given size constraints
        max_sentences = self.args.max_sentences  # 30
        batch_sampler = batch_by_size(indices, sent.num_tokens, max_sentences=max_sentences,
                                                 required_batch_size_multiple=self.args.required_batch_size_multiple, )
        # print(f"tuple(batch_sampler): {tuple(batch_sampler)}")
        itrs = EpochBatchIterator(dataset=sent, collate_fn=sent.collater, batch_sampler=batch_sampler, seed=self.args.seed,num_workers=self.args.num_workers, epoch=epoch)
        #data_iter = itrs.next_epoch_itr(shuffle=False, fix_batches_to_gpus=fix_batches_to_gpus)
        # print(f"itrs.state_dict: {itrs.state_dict()}")
        itrs = itrs._get_iterator_for_epoch(epoch=epoch, shuffle=True)
        # print(f"itrs.n(): {itrs.n()}")
        # print(f"itrs.first_batch(): {itrs.first_batch()}")
        # print(f"next(itrs)")
        # print(f"{next(itrs)}")

        return itrs
        #return data_iter
        #return data_loader

    def get_cove(self, memory, ex, mean=False):
        """ Get sentence representation.
        Args:
            memory(torch.Tensor): hidden states or word embeddings of batch
            ex(int): index of example in batch
            mean(boolean): if true, take mean over time-steps; else, sum
        Returns:
            cove(torch.Tensor): sentence representation C_e or C_h
        """
        # Get current example
        seq_ex = memory[:, ex, :]
        if self.cove_type == 'mean':
            cove = torch.mean(seq_ex, dim=0)
        else:
            cove = torch.sum(seq_ex, dim=0)
        return cove

    def getdata(self, articles):
        trainingSetSrc = load_indexed_dataset(articles[0], self.task.src_dict,
                                                         dataset_impl=self.args.dataset_impl, combine=False,
                                                         default='cached')
        trainingSetTgt = load_indexed_dataset(articles[1], self.task.tgt_dict,
                                                         dataset_impl=self.args.dataset_impl, combine=False,
                                                         default='cached')
        # print("read the text file ")self.args.data +
        # convert the read files to Monolingual dataset to make padding easy
        src_mono = MonolingualDataset(dataset=trainingSetSrc, sizes=trainingSetSrc.sizes,
                                      src_vocab=self.task.src_dict,
                                      tgt_vocab=None, shuffle=False, add_eos_for_other_targets=False)
        tgt_mono = MonolingualDataset(dataset=trainingSetTgt, sizes=trainingSetTgt.sizes,
                                      src_vocab=self.task.tgt_dict,
                                      tgt_vocab=None, shuffle=False, add_eos_for_other_targets=False)

        del trainingSetSrc, trainingSetTgt
        print("Monolingual data")
        print(f"src_mono.num_tokens(1): {src_mono.num_tokens(1)}")
        print(f"tgt_mono.num_tokens(1): {tgt_mono.num_tokens(1)}")
        return src_mono, tgt_mono

    def extract_phrase_train(self, srcPhrase, tgtPhrase, epoch):
        src_sents = []
        tgt_sents = []
        src_embeds = []
        tgt_embeds = []
        # load the dataset from the files for both source and target
        src_indexed, src_sizes = indexPhraseData(srcPhrase, dictionary = self.task.src_dict, append_eos = True, reverse_order = False)
        tgt_indexed, tgt_sizes = indexPhraseData(tgtPhrase, dictionary = self.task.tgt_dict, append_eos=True, reverse_order=False)
        #print(src_indexed)

        src_mono = MonolingualDataset(dataset=src_indexed, sizes=src_sizes,
                                      src_vocab=self.task.src_dict,
                                      tgt_vocab=None, shuffle=False, add_eos_for_other_targets=False)
        tgt_mono = MonolingualDataset(dataset=tgt_indexed, sizes=tgt_sizes,
                                      src_vocab=self.task.tgt_dict,
                                      tgt_vocab=None, shuffle=False, add_eos_for_other_targets=False)

        # Prepare iterator objects for current src/tgt document
        src_article = self._get_iterator(src_mono, dictn=self.task.src_dict,
                                         max_position=self.args.max_source_positions, epoch=epoch,
                                         fix_batches_to_gpus=False)
        tgt_article = self._get_iterator(tgt_mono, dictn=self.task.tgt_dict,
                                         max_position=self.args.max_target_positions, epoch=epoch,
                                         fix_batches_to_gpus=False)
        # Get sentence representations
        #try:
        if self.representations == 'embed-only':
            # print("Using Embeddings only for representation")
            # C_e
            src_sents += self.get_article_coves(src_article, representation='embed', mean=False, side='src')
            tgt_sents += self.get_article_coves(tgt_article, representation='embed', mean=False, side='tgt')
        else:
            # C_e and C_h

            it1 = src_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
            src_embeds += self.get_article_coves(it1, representation='embed', mean=False, side='src',
                                                 use_phrase=self.use_phrase)
            it1 = src_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
            src_sents += self.get_article_coves(it1, representation='memory', mean=False, side='src')

            it3 = tgt_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
            tgt_embeds += self.get_article_coves(it3, representation='embed', mean=False, side='tgt',
                                                 use_phrase=self.use_phrase)
            it3 = tgt_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
            tgt_sents += self.get_article_coves(it3, representation='memory', mean=False, side='tgt')

                # return
        '''except:
            # Skip document pair in case of errors
            print("error")
            src_sents = []
            tgt_sents = []
            src_embeds = []
            tgt_embeds = []'''
        # free resources for Gabbage, not necessary tho
        #src_mono.dataset.tokens_list = None
        #src_mono.dataset.sizes = None
        #src_mono.sizes = None
        #tgt_mono.sizes = None
        del tgt_mono
        del src_mono
        #print('source = ',src_sents[0][0])
        #print('target = ', tgt_sents[0][0])
        # Score src and tgt sentences

        #try:
        src2tgt, tgt2src, similarities, scores = self.score_sents(src_sents, tgt_sents)
        '''except:
            # print('Error occurred in: {}\n'.format(article_pair), flush=True)
            print(src_sents, flush=True)
            print(tgt_sents, flush=True)
            src_sents = []
            tgt_sents = []
            return'''
        # print("source 2 target ", src2tgt)
        # Keep statistics
        # epoch_similarities += similarities
        # epoch_scores += scores
        src_sents = []
        tgt_sents = []

        #try:
        if self.representations == 'dual':
            # For dual representation systems, filter C_h...
            candidates = self.filter_candidates(src2tgt, tgt2src, second=self.second)
            # ...and C_e
            comparison_pool, cand_embed = self.get_comparison_pool(src_embeds,
                                                                   tgt_embeds)
            print("The number of candidates from Phrases = ", len(candidates))
            src_embeds = []
            tgt_embeds = []
            if self.write_dual:
                # print("writing the sentences to file....")
                self.write_embed_only(candidates, cand_embed)
        else:
            print("Using Embedings only for Filtering ......")
            # Filter C_e or C_h for single representation system
            candidates = self.filter_candidates(src2tgt, tgt2src)
            comparison_pool = None
        '''except:
            # Skip document pair in case of errors
            print("Error Occured!!!!")
            # print('Error occured in: {}\n'.format(article_pair), flush=True)
            src_embeds = []
            tgt_embeds = []
            return'''


        # Extract parallel samples (secondary filter)
        phrasese = True
        self.extract_parallel_sents(candidates, comparison_pool, phrasese)
        return None

    def extract_and_train(self, comparable_data_list, epoch):

        #tracemalloc.start()
        """ Manages the alternating extraction of parallel sentences and training.
        Args:
            comparable_data_list(str): path to list of mapped documents
        Returns:
            train_stats(:obj:'onmt.Trainer.Statistics'): epoch loss statistics
        """

        self.accepted_file = open('{}_accepted-e{}.txt'.format(self.comp_log, epoch), 'w+', encoding='utf8')
        if self.use_phrase == True:
                    self.accepted_phrase = open('{}_accepted_phrase-e{}.txt'.format(self.comp_log, epoch), 'w+',
                                                encoding='utf8')
        self.status_file = '{}_status-e{}.txt'.format(self.comp_log, epoch)
        if self.write_dual:
            self.embed_file = '{}_accepted_embed-e{}.txt'.format(self.comp_log,
                                                                 epoch)
            self.hidden_file = '{}_accepted_hidden-e{}.txt'.format(self.comp_log,
                                                                   epoch)

        epoch_similarities = []
        epoch_scores = []
        src_sents = []
        tgt_sents = []
        src_embeds = []
        tgt_embeds = []

        # Go through comparable data
        with open(comparable_data_list, encoding='utf8') as c:
            comp_list = c.read().split('\n')
            #num_articles = len(comp_list)
            cur_article = 0
            for article_pair in comp_list:
                cur_article += 1
                articles = article_pair.split(' ')
                print(f"len(articles): {len(articles)}")
                # Discard malaligned documents
                if len(articles) != 2:
                    continue
                #load the dataset from the files for both source and target
                src_mono, tgt_mono = self.getdata(articles)
                # Prepare iterator objects for current src/tgt document
                print(f"self.task.src_dict: {self.task.src_dict}")
                print(f"self.args.max_source_positions: {self.args.max_source_positions}")
                src_article = self._get_iterator(src_mono, dictn=self.task.src_dict,
                                                 max_position=self.args.max_source_positions, epoch=epoch, fix_batches_to_gpus=False)
                tgt_article = self._get_iterator(tgt_mono, dictn=self.task.tgt_dict,
                                                 max_position=self.args.max_target_positions, epoch=epoch, fix_batches_to_gpus=False)

                # Get sentence representations
                try:
                    if self.representations == 'embed-only':
                        print("Using Embeddings only for representation")
                        # C_e
                        src_sents += self.get_article_coves(src_article, representation='embed', mean=False)
                        tgt_sents += self.get_article_coves(tgt_article, representation='embed', mean=False)
                    else:
                        # C_e and C_h
                        '''it1, it2 = itertools.tee(src_article)
                        it3, it4 = itertools.tee(tgt_article)'''

                        it1 = src_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
                        src_embeds += self.get_article_coves(it1, representation='embed', mean=False, side='src',
                                                         use_phrase=self.use_phrase)
                        it1 = src_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
                        src_sents += self.get_article_coves(it1, representation='memory', mean=False, side='src')

                        it3 = tgt_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
                        tgt_embeds += self.get_article_coves(it3, representation='embed', mean=False, side='tgt',
                                                         use_phrase=self.use_phrase)
                        it3 = tgt_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
                        tgt_sents += self.get_article_coves(it3, representation='memory', mean=False, side='tgt')

                        #return
                except:
                    #Skip document pair in case of errors
                    print("error")
                    src_sents = []
                    tgt_sents = []
                    src_embeds = []
                    tgt_embeds = []
                    continue
                src_mono.dataset.tokens_list = None
                src_mono.dataset.sizes = None
                src_mono.sizes = None
                tgt_mono.sizes = None
                del tgt_mono
                del src_mono

                if len(src_sents) < 15 or len(tgt_sents) < 15:
                    #print("Length LEss tahn 15")
                    continue
                print("Proceeding")
                # Score src and tgt sentences
                print("In all we have got ", len(src_sents), "source sentences and ", len(tgt_sents), "target")
                try:
                    src2tgt, tgt2src, similarities, scores = self.score_sents(src_sents, tgt_sents)
                except:
                    print('Error occurred in: {}\n'.format(article_pair), flush=True)
                    print("src_sents")
                    # print(src_sents, flush=True)
                    print("tgt_sents")
                    # print(tgt_sents, flush=True)
                    src_sents = []
                    tgt_sents = []
                    continue
                print("source 2 target ", src2tgt)
                # Keep statistics
                #epoch_similarities += similarities
                #epoch_scores += scores
                src_sents = []
                tgt_sents = []


                try:
                    if self.representations == 'dual':
                        # For dual representation systems, filter C_h...
                        candidates = self.filter_candidates(src2tgt, tgt2src, second=self.second)
                        # ...and C_e
                        comparison_pool, cand_embed = self.get_comparison_pool(src_embeds,
                                                                               tgt_embeds)
                        src_embeds = []
                        tgt_embeds = []
                        if self.write_dual:
                            #print("writing the sentences to file....")
                            self.write_embed_only(candidates, cand_embed)
                    else:
                            print("Using Embedings only for Filtering ......")
                            # Filter C_e or C_h for single representation system
                            candidates = self.filter_candidates(src2tgt, tgt2src)
                            comparison_pool = None
                except:
                    # Skip document pair in case of errors
                    print("Error Occured!!!!")
                    print('Error occured in: {}\n'.format(article_pair), flush=True)
                    src_embeds = []
                    tgt_embeds = []
                    continue


                # Extract parallel samples (secondary filter)
                self.extract_parallel_sents(candidates, comparison_pool)
                # if phrase extraction is to be used

                if self.use_phrase and (len(self.phrases.sourcesent) >= 30 and len(self.phrases.targetsent) >= 30):
                    #print("enough phrases meeen. ", len(self.phrases.sourcesent), " and  ",
                     #     len(self.phrases.targetsent), " .................")
                    #extract the sentences rejected and convert to string
                    srcSent = self.phrases.convert2string('src')
                    tgtSent = self.phrases.convert2string('tgt')
                    #print(tgtSent)
                    #if the phrase extraction method is stanford, then go ahead and get the phrases.
                    if self.args.phrase_method == 'stanford':

                        sourcePhrase = self.phrases.extractPhrasesSNL(srcSent,'src')
                        #print('extracted phrases = ',sourcePhrase)
                        #print("Completed the english phrases!!!!!!")

                        targetPhrase = self.phrases.extractPhrasesSNL(tgtSent, 'tgt')
                        #print(targetPhrase)
                        self.extract_phrase_train(sourcePhrase, targetPhrase, epoch)
                        self.phrases.resetData()

                    #print("The length of the sourcePhrase = ",len(sourcePhrase))
                    #print("The length of the targetPhrase = ",len(targetPhrase))

                    ##convert the sentences to string and perform NP extraction
                    ##print("source = ", list(self.phrases.sourcesent)[0])

                    # extract phrases using Stanford NLP/n-grams


                #print("pair bank  = ",len((self.similar_pairs.pairs)))
                # Train on extracted sentences
                self.train(epoch)
                del src2tgt, tgt2src
                #gc.collect()
                # Add to leaky code within python_script_being_profiled.py


                '''snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                print("[ Top 10 ]")
                for stat in top_stats[:30]:
                    print(stat)
                    '''

                # Train on remaining partial batch
            if len((self.similar_pairs.pairs)) > 0:
                print("batching and training")
                self.train(epoch, last=True)

        self.accepted_file.close()
        if self.use_phrase == True:
                    self.accepted_phrase.close()

        # log end-of-epoch stats
        #stats = get_training_stats(metrics.get_smoothed_values('train'))
        #self.progress.print(stats, tag='train', step=num_updates)

        # log end-of-epoch stats
        num_updates = self.trainer.get_num_updates()
        stats = get_training_stats(metrics.get_smoothed_values('train'))
        self.progress.print(stats, tag='train', step=num_updates)
        # reset epoch-level meters
        metrics.reset_meters('train')
        return None
    '''
    @metrics.aggregate('train')
    def trainRest(self, epoch):
        itrs = self.similar_pairs.yield_batch()
        itr = itrs.next_epoch_itr(shuffle=True, fix_batches_to_gpus=False)
        itr = GroupedIterator(itr, 1)
        self.progress = progress_bar.build_progress_bar(
            self.args, itr, epoch, no_progress_bar='simple',
        )
        for samples in self.progress:
            log_output = self.trainer.train_step(samples)
            num_updates = self.trainer.get_num_updates()
            if log_output is None:
                continue
            # log mid-epoch stats
            stats = get_training_stats(metrics.get_smoothed_values('train'))
            self.progress.log(stats, tag='train', step=num_updates)
            self.progress.print(stats, tag='train', step=num_updates)

        print("done")
        #del itrs, itr
    '''

    @metrics.aggregate('train')
    def train(self, epoch, last=False):
        # Check if enough parallel sentences were collected
        if last is False:
            while self.similar_pairs.contains_batch():
                # print("IT has batch.....")
                # try:
                itrs = self.similar_pairs.yield_batch()
                itr = itrs.next_epoch_itr(shuffle=True, fix_batches_to_gpus=False)
                itr = GroupedIterator(itr, self.update_freq[-1])

                self.progress = progress_bar.build_progress_bar(
                    self.args, itr, epoch, no_progress_bar='simple',
                )

                for samples in self.progress:
                    with metrics.aggregate('train_inner'):
                        print("Size of the samples = ",len(samples))
                        log_output = self.trainer.train_step(samples)
                        num_updates = self.trainer.get_num_updates()
                        if log_output is None:
                            continue
                    # log mid-epoch stats
                    stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
                    self.progress.print(stats, tag='train_inner', step=num_updates)
                    self.progress.log(stats, tag='train_inner', step=num_updates)
                    metrics.reset_meters('train_inner')
        else:
            # numberofex = self.similar_pairs.get_num_examples()
            itrs = self.similar_pairs.yield_batch()
            itr = itrs.next_epoch_itr(shuffle=True, fix_batches_to_gpus=False)

            itr = GroupedIterator(itr, self.update_freq[-1])
            self.progress = progress_bar.build_progress_bar(
                self.args, itr, epoch, no_progress_bar='simple',
            )
            for samples in self.progress:
                with metrics.aggregate('train_inner'):
                    log_output = self.trainer.train_step(samples)
                    num_updates = self.trainer.get_num_updates()
                    if log_output is None:
                        continue
                # log mid-epoch stats
                stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
                self.progress.print(stats, tag='train_inner', step=num_updates)
                self.progress.log(stats, tag='train_inner', step=num_updates)
                metrics.reset_meters('train_inner')

    def validate(self, epoch, subsets):
        """Evaluate the model on the validation set(s) and return the losses."""

        if self.args.fixed_validation_seed is not None:
            # set fixed seed for every validation
            utils.set_torch_seed(self.args.fixed_validation_seed)

        valid_losses = []
        for subset in subsets:
            # Initialize data iterator
            itr = self.task.get_batch_iterator(
                dataset=self.task.dataset(subset),
                max_tokens=self.args.max_tokens_valid,
                max_sentences=self.args.max_sentences_valid,
                max_positions=utils.resolve_max_positions(
                    self.task.max_positions(),
                    self.trainer.get_model().max_positions(),
                ),
                ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=self.args.required_batch_size_multiple,
                seed=self.args.seed,
                num_shards=self.args.distributed_world_size,
                shard_id=self.args.distributed_rank,
                num_workers=self.args.num_workers,
            ).next_epoch_itr(shuffle=False)
            progress = progress_bar.build_progress_bar(
                self.args, itr, epoch,
                prefix='valid on \'{}\' subset'.format(subset),
                no_progress_bar='simple'
            )

            # create a new root metrics aggregator so validation metrics
            # don't pollute other aggregators (e.g., train meters)
            with metrics.aggregate(new_root=True) as agg:
                for sample in progress:
                    self.trainer.valid_step(sample)

            # log validation stats
            stats = get_valid_stats(self.args, self.trainer, agg.get_smoothed_values())
            progress.print(stats, tag=subset, step=self.trainer.get_num_updates())

            valid_losses.append(stats[self.args.best_checkpoint_metric])
        return valid_losses

    def save_comp_chkp(self, epoch):
        dirs = self.save_dir + '/' + self.model_name + '_' + str(epoch) + self.src + "-" + self.tgt + ".pt"
        self.trainer.save_checkpoint(dirs, {"train_iterator": {"epoch": epoch}})

def get_valid_stats(args, trainer, stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats

def get_training_stats(stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats

def removePadding(side):
    """ Removes original padding from a sequence.
    Args:
        side(torch.Tensor): src/tgt sequence (size(seq))
    Returns:
        side(torch.Tensor): src/tgt sequence without padding
    NOTE: This only works as long as PAD_ID==1!
    """
    # Get indexes of paddings in sequence
    padding_idx = (side == 1).nonzero()
    # If there is any padding, cut sequence from first occurence of a pad
    if padding_idx.size(0) != 0:
        first_pad = padding_idx.data.tolist()[0][0]
        side = side[:first_pad]
    return side