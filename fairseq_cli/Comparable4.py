"""
Classes and methods used for training and extraction of parallel pairs
from a comparable dataset.
"""
import tracemalloc
#import gc
import re
import itertools
import random
import faiss
import faiss.contrib.torch_utils
import numpy as np
from collections import defaultdict
import torch
import time
from pathlib import Path
from fairseq.data import (
    MonolingualDataset,
    LanguagePairDataset
)
from tqdm import tqdm
from fairseq.data.data_utils import load_indexed_dataset,numpy_seed,batch_by_size,filter_by_size
from fairseq.data.iterators import EpochBatchIterator, GroupedIterator
from fairseq import (
    checkpoint_utils, utils
)
from fairseq.logging import meters, metrics, progress_bar
from omegaconf import DictConfig, OmegaConf
import argparse
import os, sys
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
from fairseq.trainer import Trainer
from fairseq.distributed import utils as distributed_utils
torch.manual_seed(10)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# debug_mode = "error"
# torch.cuda.set_sync_debug_mode(debug_mode)

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.comparable")

def get_src_len(src, use_gpu, device=""):
    if use_gpu:
        if device=="mps":
            return torch.tensor([src.size(0)], device="mps") #.cuda()
        else:
            return torch.tensor([src.size(0)]).cuda()
    else:
        return torch.tensor([src.size(0)])

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

    def resetData(self):
        self.sourcesent = set()
        self.targetsent = set()


# class RejectedBank():
#     """
#     Class that saves and prepares rejected monostylistic SRC sentences and their resulting
#     batches.
#     Args:
#         batch_size(int): number of examples in a batch
#         opt(argparse.Namespace): option object
#     """
#     pass


class PairBank():
    """
    Class that saves and prepares parallel pairs and their resulting
    batches.
    Args:
        batch_size(int): number of examples in a batch
        opt(argparse.Namespace): option object
    """

    def __init__(self, batcher, cfg):
        self.pairs = []
        self.index_memory = set()
        self.batch_size = cfg.dataset.batch_size #max_sentences
        self.batcher = batcher
        self.use_gpu = False
        self.mps = False
        self.cuda = False
        if cfg.common.cpu == False:
            self.use_gpu = True
            if torch.backends.mps.is_available():
                self.mps = True
                self.mps_device = torch.device("mps")
            else:
                self.cuda = True
        else:
            self.use_gpu = False
        self.update_freq = cfg.optimization.update_freq
        self.explen = self.batch_size * self.update_freq[-1]

    def __len__(self):
        return len(self.pairs)

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
            src_len = example.src_length.item()
            tgt_len = example.tgt_length.item()
            # print(f"example.src_length: {src_len}")
            src_examples.append(example.src)
            tgt_examples.append(example.tgt)
            src_lengths.append(src_len) # example.src_length
            tgt_lengths.append(tgt_len) # example.tgt_length
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
    def __init__(self, task, cfg, trainer):
        self.task = task
        self.cfg = cfg
        self.trainer = trainer

    def create_batch(self, src_examples, tgt_examples, src_lengths, tgt_lengths, no_target=False, shard_batch_itr=False):
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
        # print(f"src_lengths type: {type(src_lengths)}")
        # src_lengths = src_lengths.detach().cpu().numpy()
        # tgt_lengths = tgt_lengths.detach().cpu().numpy()
        pairData = LanguagePairDataset(
            src_examples, src_lengths, self.task.src_dict,
            tgt_examples, tgt_lengths, self.task.tgt_dict,
            left_pad_source=self.cfg.task.left_pad_source,
            left_pad_target=self.cfg.task.left_pad_target
        )
        # max_source_positions=self.cfg.task.max_source_positions,
        # max_target_positions=self.cfg.task.max_target_positions,

        with numpy_seed(self.cfg.common.seed):
            indices = pairData.ordered_indices()

        batch_sampler = batch_by_size(indices, pairData.num_tokens, 
        max_sentences=self.cfg.comparable.max_sentences, required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple, )
        itrs = EpochBatchIterator(dataset=pairData, collate_fn=pairData.collater,
         batch_sampler=batch_sampler, seed=self.cfg.common.seed, epoch=0, num_workers=self.cfg.dataset.num_workers,num_shards=self.trainer.data_parallel_world_size if shard_batch_itr else 1, shard_id=self.trainer.data_parallel_rank if shard_batch_itr else 0,)
        indices = None
        return itrs


def knn(x, y, k, use_gpu, index='flat'):
    '''
    small query batch, small index: CPU is typically faster
    small query batch, large index: GPU is typically faster
    large query batch, small index: could go either way
    large query batch, large index: GPU is typically faster
    '''
    return knnGPU(x, y, k, index) if use_gpu else knnCPU(x, y, k, index)

def knn_v2(x, y, k, use_gpu, index='flat'):
    '''
    small query batch, small index: CPU is typically faster
    small query batch, large index: GPU is typically faster
    large query batch, small index: could go either way
    large query batch, large index: GPU is typically faster
    string_factory = "L2norm,OPQ16_64,IVF30000_HNSW32,PQ16" # Flat
    '''
    return knnGPU_v2(x, y, k, index) if use_gpu else knnCPU(x, y, k, index)

def knnGPU_v2(x, y, k, index='flat', faiss_verbose=True, batch_size=100000, train_size=1170000, use_float16=True):
    ngpus = faiss.get_num_gpus()
    dim = 512 #x.shape[1]
    string_factory = "OPQ16_64,IVF30000,PQ16"
    logger.info(f"index string_factory: {string_factory}")
    #"OPQ32,IMI2x8,PQ32" "OPQ32,IVF256,PQ32"
    #"PCA64,IVF30000,Flat"
    #"IVF200,Flat" # -- works for < 1M indices
    #"OPQ16_64,IVF30000,PQ16" # OPQ is a CPU-based vector transform
    # IVF30000,PQ16
    idx = faiss.index_factory(dim, string_factory, faiss.METRIC_INNER_PRODUCT)
    # https://www.pinecone.io/learn/composite-indexes/
    ivf = faiss.extract_index_ivf(idx)
    res = faiss.StandardGpuResources()
    res.setDefaultNullStreamAllDevices()
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.verbose = True
    co.indicesOptions = faiss.INDICES_CPU
    co.common_ivf_quantizer = False
    co.usePrecomputed = False
    co.useFloat16 = use_float16
    co.useFloat16CoarseQuantizer = True
    if ngpus == 1:
        gpu_idx = faiss.index_cpu_to_gpu(res, 0, idx)
    else: # multiple gpus
        res_list = [res for _ in range(ngpus)]
        gpu_idx = faiss.index_cpu_to_gpu_multiple_py(resources=res_list, index=idx, co=co)
        # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/491
        # inputs to gpu_idx have to be on cpu and inside the function, they would be moved back to gpu!

    # logger.info("Created faiss index of type {}".format(type(gpu_idx)))
    faiss_verbose = None

    # Set verbosity level
    if faiss_verbose is not None:
        if hasattr(gpu_idx, "index") and gpu_idx.index is not None:
            gpu_idx.index.verbose = faiss_verbose
        if hasattr(gpu_idx, "quantizer") and gpu_idx.quantizer is not None:
            gpu_idx.quantizer.verbose = faiss_verbose
        if hasattr(gpu_idx, "clustering_index") and gpu_idx.clustering_index is not None:
            gpu_idx.clustering_index.verbose = faiss_verbose

    # Train
    # logger.info("training index")

    ys, xs = [], []

    for i in tqdm(range(0, len(y), batch_size)):
        yb = torch.stack(list(y[i : i + batch_size]), dim=0)
        yb = yb.type(torch.float32).cpu() # convert to float32 to move to cpu!
        yb = torch.nn.functional.normalize(yb, p=2, dim=1)
        # logger.info(f"yb.size: {yb.size()}")
        ys.append(yb)
    
    y = torch.cat(ys, dim=0)
    
    train_vecs = y
    if train_size is not None:
        # train_vecs = y[:train_size].cpu() 
        with torch.no_grad():
            indices = torch.tensor(random.sample(range(y.size()[0]), train_size))
            indices = torch.tensor(indices)
            train_vecs = train_vecs[indices]
    logger.info("Training the index with {} randomly-sampled vectors".format(len(train_vecs)))
    gpu_idx.train(train_vecs)

    # Add vectors
    logger.info("Adding {} vectors to the faiss index".format(len(y)))
    gpu_batch_size=200000
    for i in tqdm(range(0, len(y), gpu_batch_size)):
        vecs = y[i : i + batch_size]
        vecs = vecs.type(torch.float32).cpu() # convert to float32 to move to cpu! 
        gpu_idx.add(vecs)
    
    # send batched queries to the full faiss index
    # size of sim and inds should be equal to len(y) 
    logger.info("Querying {} vectors to the faiss index".format(len(x)))
    sim, ind = [], []
    faiss.GpuParameterSpace().set_index_parameter(ivf, "nprobe", 50)
    # faiss.ParameterSpace().set_index_parameter(ivf, "nprobe", 50)
    for i in tqdm(range(0, len(x), gpu_batch_size)):
        xb = torch.stack(list(x[i : i + gpu_batch_size]), dim=0)
        xb = xb.type(torch.float32).cpu() # convert to float32 to move to cpu!
        xb = torch.nn.functional.normalize(xb, p=2, dim=1)
        xs.append(xb)
        bsim, bind = gpu_idx.search(xb, k) # x[i : i + batch_size].cpu()
        sim.append(bsim)
        # print(f"len(sim): {len(sim)}")
        ind.append(bind)
    
    # logger.info(f"concat results..")
    # logger.info(f"sim[0].size(): {sim[0].size()}")
    rsim = torch.cat(sim, dim=0).cpu() # along the rows
    rind = torch.cat(ind, dim=0).cpu()
    x = torch.cat(xs, dim=0)
    # logger.info(f"similarity results size: {rsim.size()}")

    # logger.info(f"type(rsim): {type(rsim)} type(rind): {type(rsim)}")

    return rsim, rind, x, y

    
def knnCPU(x, y, k, index='flat'):
    start=time.time()
    dim = x.shape[1]
    m = 8 # number of centroid IDs in final compressed vectors
    bits = 8 # number of bits in each centroid
    nlist = 100  # how many cells
    if index == 'ivf':
        # quantizer = faiss.IndexFlatIP(dim)
        # idx = faiss.IndexIVFFlat(quantizer, dim, nlist)
        idx = faiss.index_factory(dim, "IVF200,Flat", faiss.METRIC_INNER_PRODUCT)
        idx.train(y)
        # print(f"idx.is_trained: {idx.is_trained}") 40000
    elif index == 'hnsw': 
        idx = faiss.index_factory(dim, "PCA64,IVF30000_HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
        idx.train(y)
    elif index =='pq':
        # quantizer = faiss.IndexFlatIP(dim)
        # idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)
        idx = faiss.index_factory(dim, "IVF200,PQ16", faiss.METRIC_INNER_PRODUCT)
        idx.train(y)
    else:
        idx = faiss.IndexFlatIP(dim)

    # print(f"num embeddings indexed: {idx.ntotal}")
    idx.add(y)
    sim, ind = idx.search(x, k)
    # print(f"sim[:3]: {sim[:3]}")
    # print(f"ind: {ind}")
    # print(f"time taken to build the index: {time.time()-start} secs")
    return sim, ind, x, y

def knnGPU(x, y, k, index='flat', mem=48*1024*1024*
           1024):
    # d = srcRep.shape[1]
    # print(f"d: {d}")

    # 1. take a query vector xq 2. identify the cell it belongs to
    # 3. use IndexFlat2 to search btw query vector & all other vectors 
    # belonging to that specific cell
    # '''
    # PQ = Product Quantization. IVF reduces the scope of our search, PQ approximates
    # distance/similarity calculation. 
    # 1. split OG vector into several subvectors. 
    # 2. for each set of subvector, perform a clustering operation - creating multiple centroids 
    # for each sub-vector set. 
    # 3. In the vector of sub-vecs, replace each sub-vec with the ID of its nearest set-specific centroid
    # '''
    # https://github.com/facebookresearch/LASER/blob/main/source/mine_bitexts.py
    print(f"faiss.get_num_gpus(): {faiss.get_num_gpus()}")
    ngpus = faiss.get_num_gpus()
    dim = x.shape[1]
    m = 8 # number of centroid IDs in final compressed vectors
    bits = 8 # number of bits in each centroid
    nlist = 100  # how many cells
    res = faiss.StandardGpuResources()
    res.setDefaultNullStreamAllDevices()
    co = faiss.GpuClonerOptions()
    batch_size = mem // (dim*4)
    print(f"batch_size: {batch_size}")
    if batch_size > x.shape[0]:
        batch_size = 64 #x.shape[0] // 10000
        print(f"batch_size: {batch_size}")
    
    sim = np.zeros((x.shape[0], k), dtype=np.float32)
    ind = np.zeros((x.shape[0], k), dtype=np.int64)
    for xfrom in range(0, x.shape[0], batch_size):
        xto = min(xfrom + batch_size, x.shape[0]) # to_src_ind
        bsims, binds = [], []
        for yfrom in range(0, y.shape[0], batch_size):
            yto = min(yfrom + batch_size, y.shape[0]) # to_trg_ind
            logger.info('{}-{}  ->  {}-{}'.format(xfrom, xto, yfrom, yto))
            if index == 'ivf': # below 1M vectors
                idx = faiss.index_factory(dim, "IVF1200,Flat", faiss.METRIC_INNER_PRODUCT)
                idx = faiss.index_cpu_to_gpu(res, 0, index=idx)
                idx.train(y)
            elif index =='pq':
                # quantizer = faiss.IndexFlatIP(dim)
                # idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)
                idx = faiss.index_factory(dim, "IVF1200,PQ16", faiss.METRIC_INNER_PRODUCT)
                idx = faiss.index_cpu_to_all_gpus(index=idx, co=co)
                idx.train(y)
            elif index == 'hnsw': # for 1M-10M vectors 
                idx = faiss.index_factory(dim, "PCA64,IVF30000_HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
                idx_ivf = faiss.extract_index_ivf(idx)
                res = [faiss.StandardGpuResources() for _ in range(ngpus)]
                #clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(idx_ivf.d), co, ngpu=1)
                # clustering_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(idx_ivf.d))
                # index_cpu_to_gpu_multiple_py(resources, index, co=None, gpus=None)
                #clustering_index = faiss.index_cpu_to_all_gpus(res=res, index=faiss.IndexFlatIP(idx_ivf.d),  co=co)
                clustering_index = faiss.index_cpu_to_gpu_multiple_py(resources=res, index=faiss.IndexFlatIP(idx_ivf.d), co=co)
                idx_ivf.clustering_index = clustering_index
                idx.train(y)
            else:
                idx = faiss.IndexFlatIP(dim)
                idx = faiss.index_cpu_to_all_gpus(index=idx, co=co)
            # quantizer = faiss.IndexFlatL2(d)
            # idx = faiss.IndexIVFFlat(quantizer, d, nlist)
            # #idx = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
            # idx.train(srcRep)
            # print(f"idx.is_trained: {idx.is_trained}")
            # idx.add(srcRep)
            # print(f"num embeddings indexed: {idx.ntotal}")

            # idx.nprobe = 1 # to increase the search scope
            # # large nprobe values = slower but more accurate search 
    
            idx.add(y[yfrom:yto]) # added trg_batch = batch_size to the index  
            bsim, bind = idx.search(x[xfrom:xto], min(k, yto-yfrom)) # find k nearest neighbours for the batched queries
            bsims.append(bsim)
            binds.append(bind + yfrom)
            del idx
        bsims = np.concatenate(bsims, axis=1)
        binds = np.concatenate(binds, axis=1)
        aux = np.argsort(-bsims, axis=1)
        for i in range(xfrom, xto):
            for j in range(k):
                sim[i, j] = bsims[i-xfrom, aux[i-xfrom, j]]
                ind[i, j] = binds[i-xfrom, aux[i-xfrom, j]]
    return sim, ind

def score(x, y, fwd_mean, bwd_mean, margin):
    return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)

def score_candidates(x, y, candidate_inds, fwd_mean, bwd_mean, margin, verbose=False):
    if verbose:
        print(' - scoring {:d} candidates'.format(x.shape[0]))
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = candidate_inds[i, j]
            scores[i, j] = score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin)
    return scores


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


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

    def __init__(self, model, trainer, task, cfg):
        self.sim_measure = cfg.comparable.sim_measure
        self.threshold = cfg.comparable.threshold
        self.use_threshold = cfg.comparable.use_threshold
        self.model_name = cfg.comparable.model_name
        self.save_dir = cfg.comparable.save_dir
        self.use_phrase = cfg.comparable.use_phrase
        self.model = model #trainer.get_model().encoder
        self.usepos =  cfg.comparable.usepos
        self.temp = cfg.comparable.temp
        Path(f"/netscratch/jalota/pickle/{self.temp}/").mkdir(parents=True, exist_ok=True)
        # print("Use positional encoding = ", self.usepos)
        self.trainer = trainer
        # print(f"self.trainer: {self.trainer}")
        self.task = self.trainer.task
        self.encoder = self.trainer.get_model().encoder
        # print(f"self.encoder: {self.encoder}")
        self.batch_size = cfg.comparable.max_sentences
        self.batcher = BatchCreator(task, cfg, trainer)
        self.similar_pairs = PairBank(self.batcher, cfg)
        self.accepted = 0
        self.accepted_limit = 0
        self.declined = 0
        self.total = 0
        self.cfg = cfg
        self.comp_log = cfg.comparable.comp_log
        Path(self.comp_log).mkdir(parents=True, exist_ok=True)
        self.cove_type = cfg.comparable.cove_type
        self.update_freq = cfg.optimization.update_freq
        self.k = cfg.comparable.k #20 #cfg.comparable.k
        self.trainstep = 0
        self.second = cfg.comparable.second
        self.representations = cfg.comparable.representations
        # self.task = task
        self.write_dual = cfg.comparable.write_dual
        self.no_swaps = cfg.comparable.no_swaps
        self.symmetric = cfg.comparable.symmetric
        self.add_noise = cfg.comparable.add_noise
        self.use_bt = cfg.comparable.use_bt
        self.stats = None
        self.progress = None
        self.src, self.tgt = "tr", "og" #args.source_lang, args.target_lang
        self.use_gpu = False
        self.mps = False
        self.cuda = False
        self.mps_device = None
        self.log_interval = cfg.common.log_interval #5
        self.margin = cfg.comparable.margin
        self.verbose = cfg.comparable.verbose
        self.mode = cfg.comparable.mode
        self.faiss = cfg.comparable.faiss
        self.retrieval = cfg.comparable.retrieval
        self.faiss_use_gpu = cfg.comparable.faiss_use_gpu
        self.faiss_output = cfg.comparable.faiss_output
        self.index=cfg.comparable.index
        # print(f"args.cpu: {args.cpu}")
        if cfg.common.cpu == False:
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
            # return out
        elif status == 'hidden_only':
            with open(self.hidden_file, 'a', encoding='utf8') as f:
                f.write(out)
            # return out
        return None

    def extract_parallel_sents(self, candidates, candidate_pool, phrasese=False, use_threshold=False):
        """
        Extracts parallel sentences from candidates and adds them to the
        PairBank (secondary filter).
        Args:
            candidates(list): list of src, tgt pairs (C_h) # memory reps
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
                
                elif self.in_candidate_pool(candidate, candidate_pool) and not use_threshold:
                    src = candidate[0]
                    tgt = candidate[1]
                    score = candidate[2]

                    self.similar_pairs.add_example(src, tgt)
                    self.write_sentence(removePadding(src), removePadding(tgt), 'accepted', score)
                    self.accepted += 1
                    if self.symmetric:
                        self.similar_pairs.add_example(tgt, src)
                        self.write_sentence(tgt, src, 'accepted', score)
                    self.total += 1
                
                elif use_threshold or self.in_candidate_pool(candidate, candidate_pool):
                # Apply threshold (single-representation systems only)

                    src = candidate[0]
                    tgt = candidate[1]
                    score = candidate[2]

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
                            self.write_sentence(tgt, src, 'accepted', score)
                    else:
                        # print("threshold not met!!!")
                        self.declined += 1
                    self.total += 1
                else: # doesnt match thresold or not in candidate-pool
                    continue
        
        return None

    def write_embed_only(self, candidates, cand_embed):
        """ Writes C_e scores to file (if --write-dual is set).
        Args:
            candidates(list): list of src, tgt pairs (C_h) # memory reps
            cand_embed(list): list of src, tgt pairs (C_e) # embed reps
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

    def faiss_sent_scoring_v2(self, src_sents, tgt_sents):
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
        start = time.time()
        
        srcSent, srcRep = zip(*src_sents)
        # print(f"srcSent: {srcSent}")
        tgtSent, tgtRep = zip(*tgt_sents)
        # print(f"tgtSent: {tgtSent}")

        # print("faiss sent scoring")

        if self.faiss_use_gpu:
            # https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
            ngpus = faiss.get_num_gpus()
            logger.info(f"number of GPUs: {ngpus}")

        # srcSent2ind = {sent:i for i, sent in enumerate(srcSent)}
        # tgtSent2ind = {sent:i for i, sent in enumerate(tgtSent)}

        # logger.info(f"len srcRep: {len(srcRep)}")
        # logger.info(f"srcRep: {srcRep}")

        

        # x = torch.stack(list(srcRep), dim=0) #torch.cat(srcRep, dim=1) # concat along the rows
        # y = torch.stack(list(tgtRep), dim=0)
        if self.faiss_use_gpu:
            x, y = srcRep, tgtRep

            # logger.info(f"len x: {x.size()}")
            # logger.info(f"x: {x}")
        else:
            x= np.asarray([rep.detach().cpu().numpy() for rep in srcRep])
            y= np.asarray([rep.detach().cpu().numpy() for rep in tgtRep])
            
            # print(f"normalising x.dtype : {x.dtype}")
            faiss.normalize_L2(x)
            faiss.normalize_L2(y)
            # logger.info("done torch normalizing")
            # logger.info(f"x.size(): {x.size()}")
        # https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209/9
        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        # y = torch.nn.functional.normalize(y, p=2, dim=1)
        #F.normalize(x, p=2, dim=1)

        candidates = []

        # torch.from_numpy(a)
        
        # calculate knn in both directions
        if self.retrieval != 'bwd':
            if self.verbose:
                print(' - perform {:d}-nn source against target'.format(self.k))
            x2y_sim, x2y_ind, x, y = knn_v2(x, y, self.k, self.faiss_use_gpu, self.index)
            if self.faiss_use_gpu:
                x2y_sim = x2y_sim.numpy() #.detach().cpu().numpy()
                x2y_ind = x2y_ind.numpy() # .detach().cpu()
            x2y_mean = x2y_sim.mean(axis=1)
            #x2y_mean = torch.mean(x2y_sim, 1)

            # print(f"x2y_sim.shape: {x2y_sim.shape}")
            # print(f"x2y_ind.shape: {x2y_ind.shape}")

        if self.retrieval != 'fwd':
            if self.verbose:
                print(' - perform {:d}-nn target against source'.format(self.k))
            y2x_sim, y2x_ind, _, _ = knn_v2(y, x, self.k, self.faiss_use_gpu, self.index)
            # logger.info(f"type(y2x_sim): {type(y2x_sim)}, type(y2x_ind): {type(y2x_ind)}")
            if self.faiss_use_gpu:
                y2x_sim = y2x_sim.numpy() # .detach().cpu()
                y2x_ind = y2x_ind.numpy() # .detach().cpu().numpy()
            y2x_mean = y2x_sim.mean(axis=1)
            # y2x_mean = torch.mean(y2x_sim, 1)

        # margin function
        if self.margin == 'absolute':
            margin = lambda a, b: a
        elif self.margin == 'distance':
            margin = lambda a, b: a - b
        else:  # args.margin == 'ratio':
            margin = lambda a, b: a / b

        # print(f"margin: {margin}")

        fout = open(self.faiss_output, mode='w', encoding='utf8', errors='surrogateescape')

        src_inds=list(range(len(srcSent)))
        trg_inds=list(range(len(tgtSent)))

        if self.mode == 'search':
            if self.verbose:
                print(' - Searching for closest sentences in target')
                print(' - writing alignments to {:s}'.format(self.faiss_output))
            scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin, self.verbose)
            best = x2y_ind[np.arange(x.shape[0]), scores.argmax(axis=1)]

            print(f"best: {best}")

            nbex = x.shape[0]
            ref = np.linspace(0, nbex-1, nbex).astype(int)  # [0, nbex)
            err = nbex - np.equal(best.reshape(nbex), ref).astype(int).sum()
            print(' - errors: {:d}={:.2f}%'.format(err, 100*err/nbex))
            for i in src_inds:
                print(tgtSent[best[i]], file=fout)

        elif self.mode == 'score':
            for i, j in zip(src_inds, trg_inds):
                s = score(x[i], y[j], x2y_mean[i], y2x_mean[j], margin)
                src = srcSent[i]
                tgt = tgtSent[j]
                src_words = self.task.src_dict.string(src)
                tgt_words = self.task.tgt_dict.string(tgt)
                out = 'src: {}\ttgt: {}\tsimilarity: {}\n'.format(removeSpaces(' '.join(src_words)),
                                                                            removeSpaces(' '.join(tgt_words)), s)
                print(out, file=fout)

        elif self.mode == 'mine':
            if self.verbose:
                logger.info(' - mining for parallel data')
            fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin, self.verbose)
            bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin, self.verbose)
            fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
            # print(f"fwd_best: {fwd_best}")
            bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]
            # print(f"bwd_best: {bwd_best}")
            if self.verbose:
                logger.info(' - writing alignments to {:s}'.format(self.faiss_output))
                if self.threshold > 0:
                    logger.info(' - with threshold of {:f}'.format(self.threshold))
            if self.retrieval == 'fwd':
                for i, j in enumerate(fwd_best):
                    s = fwd_scores[i].max()
                    src = srcSent[i]
                    tgt = tgtSent[j]
                    src_words = self.task.src_dict.string(src)
                    tgt_words = self.task.tgt_dict.string(tgt)
                    out = 'src: {}\ttgt: {}\tsimilarity: {}\n'.format(removeSpaces(' '.join(src_words)),
                                                                                removeSpaces(' '.join(tgt_words)), s)
                    print(out, file=fout)
                    # print(fwd_scores[i].max(), srcSent[i], tgtSent[j], sep='\t', file=fout)
                    candidates.append((srcSent[i], tgtSent[j], s))
            if self.retrieval == 'bwd':
                for j, i in enumerate(bwd_best):
                    s = bwd_scores[j].max()
                    src = srcSent[i]
                    tgt = tgtSent[j]
                    src_words = self.task.src_dict.string(src)
                    tgt_words = self.task.tgt_dict.string(tgt)
                    out = 'src: {}\ttgt: {}\tsimilarity: {}\n'.format(removeSpaces(' '.join(src_words)),
                                                                                removeSpaces(' '.join(tgt_words)), s)
                    print(out, file=fout)
                    # print(bwd_scores[j].max(), srcSent[i], tgtSent[j], sep='\t', file=fout)
                    candidates.append((srcSent[i], tgtSent[j], s))
            if self.retrieval == 'intersect':
                for i, j in enumerate(fwd_best):
                    if bwd_best[j] == i:
                        s = fwd_scores[i].max()
                        src = srcSent[i]
                        tgt = tgtSent[j]
                        src_words = self.task.src_dict.string(src)
                        tgt_words = self.task.tgt_dict.string(tgt)
                        out = 'src: {}\ttgt: {}\tsimilarity: {}\n'.format(removeSpaces(' '.join(src_words)),
                                                                                    removeSpaces(' '.join(tgt_words)), s)
                        print(out, file=fout)
                        # print(fwd_scores[i].max(), srcSent[i], tgtSent[j], sep='\t', file=fout)
                        candidates.append((srcSent[i], tgtSent[j], s))
            if self.retrieval == 'max':
                indices = np.stack((np.concatenate((np.arange(x.shape[0]), bwd_best)),
                                    np.concatenate((fwd_best, np.arange(y.shape[0])))), axis=1)
                scores = np.concatenate((fwd_scores.max(axis=1), bwd_scores.max(axis=1)))
                seen_src, seen_trg = set(), set()
                for i in np.argsort(-scores):
                    src_ind, trg_ind = indices[i]
                    if not src_ind in seen_src and not trg_ind in seen_trg:
                        seen_src.add(src_ind)
                        seen_trg.add(trg_ind)
                        if scores[i] > self.threshold:
                            s = scores[i]
                            src = srcSent[src_ind]
                            tgt = tgtSent[trg_ind]
                            src_words = self.task.src_dict.string(src)
                            tgt_words = self.task.tgt_dict.string(tgt)
                            out = 'src: {}\ttgt: {}\tsimilarity: {}\n'.format(removeSpaces(' '.join(src_words)),
                                                                                        removeSpaces(' '.join(tgt_words)), s)
                            print(out, file=fout)
                            # print(scores[i], srcSent[src_ind], tgtSent[trg_ind], sep='\t', file=fout)
                            candidates.append((srcSent[src_ind], tgtSent[trg_ind], scores[i]))

        fout.close()
        logger.info(f"time taken by faiss sent scoring: {time.time()-start} seconds.")
        # logger.info(f"num candidates: {len(candidates)}")
        return candidates

    
    def faiss_sent_scoring(self, src_sents, tgt_sents):
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
        start = time.time()
        
        srcSent, srcRep = zip(*src_sents)
        # print(f"srcSent: {srcSent}")
        tgtSent, tgtRep = zip(*tgt_sents)
        # print(f"tgtSent: {tgtSent}")

        print("faiss sent scoring")

        if self.faiss_use_gpu:
            # https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
            ngpus = faiss.get_num_gpus()
            logger.info(f"number of GPUs: {ngpus}")

        # srcSent2ind = {sent:i for i, sent in enumerate(srcSent)}
        # tgtSent2ind = {sent:i for i, sent in enumerate(tgtSent)}

        x= np.asarray([rep.detach().cpu().numpy() for rep in srcRep])
        y= np.asarray([rep.detach().cpu().numpy() for rep in tgtRep])
        
        print(f"normalising x.dtype : {x.dtype}")
        faiss.normalize_L2(x)
        faiss.normalize_L2(y)

        logger.info("done faiss normalizing")

        candidates = []

        # torch.from_numpy(a)
        
        # calculate knn in both directions
        if self.retrieval != 'bwd':
            if self.verbose:
                print(' - perform {:d}-nn source against target'.format(self.k))
            x2y_sim, x2y_ind = knn(x, y, min(y.shape[0], self.k), self.faiss_use_gpu, self.index)
            x2y_mean = x2y_sim.mean(axis=1)
            # logger.info(f"x2y_sim.shape: {x2y_sim.shape}")
            logger.info(f"x2y_ind.shape: {x2y_ind.shape}")

        if self.retrieval != 'fwd':
            if self.verbose:
                print(' - perform {:d}-nn target against source'.format(self.k))
            y2x_sim, y2x_ind = knn(y, x, min(x.shape[0], self.k), self.faiss_use_gpu, self.index)
            y2x_mean = y2x_sim.mean(axis=1)
            logger.info(f"y2x_ind.shape: {y2x_ind.shape}")

        # margin function
        if self.margin == 'absolute':
            margin = lambda a, b: a
        elif self.margin == 'distance':
            margin = lambda a, b: a - b
        else:  # args.margin == 'ratio':
            margin = lambda a, b: a / b

        # print(f"margin: {margin}")

        fout = open(self.faiss_output, mode='w', encoding='utf8', errors='surrogateescape')

        src_inds=list(range(len(srcSent)))
        trg_inds=list(range(len(tgtSent)))

        if self.mode == 'search':
            if self.verbose:
                print(' - Searching for closest sentences in target')
                print(' - writing alignments to {:s}'.format(self.faiss_output))
            scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin, self.verbose)
            best = x2y_ind[np.arange(x.shape[0]), scores.argmax(axis=1)]

            print(f"best: {best}")

            nbex = x.shape[0]
            ref = np.linspace(0, nbex-1, nbex).astype(int)  # [0, nbex)
            err = nbex - np.equal(best.reshape(nbex), ref).astype(int).sum()
            print(' - errors: {:d}={:.2f}%'.format(err, 100*err/nbex))
            for i in src_inds:
                print(tgtSent[best[i]], file=fout)

        elif self.mode == 'score':
            for i, j in zip(src_inds, trg_inds):
                s = score(x[i], y[j], x2y_mean[i], y2x_mean[j], margin)
                src = srcSent[i]
                tgt = tgtSent[j]
                src_words = self.task.src_dict.string(src)
                tgt_words = self.task.tgt_dict.string(tgt)
                out = 'src: {}\ttgt: {}\tsimilarity: {}\n'.format(removeSpaces(' '.join(src_words)),
                                                                            removeSpaces(' '.join(tgt_words)), s)
                print(out, file=fout)

        elif self.mode == 'mine':
            if self.verbose:
                print(' - mining for parallel data')
            fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin, self.verbose)
            bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin, self.verbose)
            fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
            # print(f"fwd_best: {fwd_best}")
            bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]
            # print(f"bwd_best: {bwd_best}")
            if self.verbose:
                print(' - writing alignments to {:s}'.format(self.faiss_output))
                if self.threshold > 0:
                    print(' - with threshold of {:f}'.format(self.threshold))
            if self.retrieval == 'fwd':
                for i, j in enumerate(fwd_best):
                    s = fwd_scores[i].max()
                    src = srcSent[i]
                    tgt = tgtSent[j]
                    src_words = self.task.src_dict.string(src)
                    tgt_words = self.task.tgt_dict.string(tgt)
                    out = 'src: {}\ttgt: {}\tsimilarity: {}\n'.format(removeSpaces(' '.join(src_words)),
                                                                                removeSpaces(' '.join(tgt_words)), s)
                    print(out, file=fout)
                    # print(fwd_scores[i].max(), srcSent[i], tgtSent[j], sep='\t', file=fout)
                    candidates.append((srcSent[i], tgtSent[j], s))
            if self.retrieval == 'bwd':
                for j, i in enumerate(bwd_best):
                    s = bwd_scores[j].max()
                    src = srcSent[i]
                    tgt = tgtSent[j]
                    src_words = self.task.src_dict.string(src)
                    tgt_words = self.task.tgt_dict.string(tgt)
                    out = 'src: {}\ttgt: {}\tsimilarity: {}\n'.format(removeSpaces(' '.join(src_words)),
                                                                                removeSpaces(' '.join(tgt_words)), s)
                    print(out, file=fout)
                    # print(bwd_scores[j].max(), srcSent[i], tgtSent[j], sep='\t', file=fout)
                    candidates.append((srcSent[i], tgtSent[j], s))
            if self.retrieval == 'intersect':
                for i, j in enumerate(fwd_best):
                    if bwd_best[j] == i:
                        s = fwd_scores[i].max()
                        src = srcSent[i]
                        tgt = tgtSent[j]
                        src_words = self.task.src_dict.string(src)
                        tgt_words = self.task.tgt_dict.string(tgt)
                        out = 'src: {}\ttgt: {}\tsimilarity: {}\n'.format(removeSpaces(' '.join(src_words)),
                                                                                    removeSpaces(' '.join(tgt_words)), s)
                        print(out, file=fout)
                        # print(fwd_scores[i].max(), srcSent[i], tgtSent[j], sep='\t', file=fout)
                        candidates.append((srcSent[i], tgtSent[j], s))
            if self.retrieval == 'max':
                indices = np.stack((np.concatenate((np.arange(x.shape[0]), bwd_best)),
                                    np.concatenate((fwd_best, np.arange(y.shape[0])))), axis=1)
                scores = np.concatenate((fwd_scores.max(axis=1), bwd_scores.max(axis=1)))
                seen_src, seen_trg = set(), set()
                for i in np.argsort(-scores):
                    src_ind, trg_ind = indices[i]
                    if not src_ind in seen_src and not trg_ind in seen_trg:
                        seen_src.add(src_ind)
                        seen_trg.add(trg_ind)
                        if scores[i] > self.threshold:
                            s = scores[i]
                            src = srcSent[src_ind]
                            tgt = tgtSent[trg_ind]
                            src_words = self.task.src_dict.string(src)
                            tgt_words = self.task.tgt_dict.string(tgt)
                            out = 'src: {}\ttgt: {}\tsimilarity: {}\n'.format(removeSpaces(' '.join(src_words)),
                                                                                        removeSpaces(' '.join(tgt_words)), s)
                            print(out, file=fout)
                            # print(scores[i], srcSent[src_ind], tgtSent[trg_ind], sep='\t', file=fout)
                            candidates.append((srcSent[src_ind], tgtSent[trg_ind], scores[i]))

        fout.close()
        logger.info(f"time taken by faiss sent scoring: {time.time()-start} seconds.")
        # logger.info(f"num candidates: {len(candidates)}")
        return candidates

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

        srcSent, srcRep = zip(*src_sents)
        tgtSent, tgtRep = zip(*tgt_sents)

        #print("At the point of unzipping the list of tuple....................")
        #unzip the list ot tiples to have two lists of equal length each sent, repre
        
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

        # print(f"tgtRp: {tgtRp}")
        # print(f"self.sim_measure: {self.sim_measure}")

        # Return cosine similarity if that is the scoring function
        if self.sim_measure == 'cosine':
            matx = self.sim_matrix(srcRp, tgtRp)
            # print(f"going into double loop")
            for i in range(len(srcSent)):
                for j in range(len(tgtSent)):
                    #print(f"i: {i}, j: {j}")
                    if srcSent[i][0] == tgtSent[j][0]:
                        continue
                    src2tgt[srcSent[i]][tgtSent[j]] = matx[i][j].tolist() # for each sent in SRC -> every TGT is assigned a score  
                    tgt2src[tgtSent[j]][srcSent[i]] = matx[i][j].tolist() 
                    # src2tgt = { "dis a src sent": {"dis a tg": 0.2, "dis s a TRG": 0.6, "dis": 0.12} }
                    similarities.append(matx[i][j].tolist())
            return src2tgt, tgt2src, similarities, similarities
        else:
            sim_mt, sumDistSource, sumDistTarget = self.sim_matrix(srcRp, tgtRp)
            # sim_mt, nearestSrc, nearestTgt = self.sim_matrix(srcRp, tgtRp)
            # sumDistSource = torch.sum(nearestSrc, 1).cuda() /self.div
            # sumDistTarget = torch.sum(nearestTgt, 0).cuda() /self.div
            # print(f"sumDistSource device: {sumDistSource.get_device()}")
            # print(f"sim_mt: {sim_mt}")

            # print(f"going into double loop")
            for i in range(len(srcSent)): # m 
                for j in range(len(tgtSent)): # n
                    #print(f"i: {i}, j: {j}")
                    if srcSent[i][0] == tgtSent[j][0]:
                        continue
                    # assign margin scores
                    tgt2src[tgtSent[j]][srcSent[i]] = src2tgt[srcSent[i]][tgtSent[j]] = sim_mt[i][j].tolist() / (sumDistSource[i].tolist() + sumDistTarget[j].tolist())
                    #tgt2src[tgtSent[j]][srcSent[i]] = sim_mt[i][j].tolist() / (sumDistTarget[j].tolist() + sumDistSource[i].tolist())
                    similarities.append(sim_mt[i][j].tolist())

            # Get list of scores for statistics
        '''for src in list(src2tgt.keys()):
            scores += list(src2tgt[src].values())'''
        # print(f"finished with the double loop. going out of score_sents.")
        return src2tgt, tgt2src, similarities, scores

    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1)).detach().cpu()
        # print(f"sim_mt: {sim_mt}")
        # print(f"sim_mt shape in sim_matrix: {sim_mt.shape}")
        del a_n, b_n, a_norm, b_norm
        if self.sim_measure == 'cosine':
            return sim_mt.cuda()
        # print(f"self.k: {self.k}")
        # print("nearestSrc")
        # print(torch.topk(sim_mt, self.k, dim=1, largest=True, sorted=False, out=None))
        nearestSrc = torch.topk(sim_mt, self.k, dim=1, largest=True, sorted=False, out=None)
        #sumDistSource = torch.sum(nearestSrc[0], 1)
        # print(f"nearestSrc: {nearestSrc}")
        # print("nearestTgt")
        nearestTgt = torch.topk(sim_mt, self.k, dim=0, largest=True, sorted=False, out=None)
        #sumDistTarget = torch.sum(nearestTgt[0], 0)
        # print(f"nearestTgt: {nearestTgt}")
        # print(f"device self.div: {self.div.get_device()}")
        sim_mt = sim_mt.cuda()
        # print(f"after sim_mt: {sim_mt}")
        # return sim_mt, nearestSrc[0], nearestTgt[0]
        c = torch.sum(nearestSrc[0], 1)/self.div.detach().cpu()
        d = torch.sum(nearestTgt[0], 0)/self.div.detach().cpu()
        # print(f"torch.sum(nearestSrc[0], 1): {c.shape}")
        # print(f"torch.sum(nearestTgt[0], 0): {d.shape}")

        return sim_mt , c.cuda(), d.cuda()
        # return sim_mt, torch.sum(nearestSrc[0], 1)/self.div, torch.sum(nearestTgt[0], 0)/self.div
        # return sim_mt, c, d


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
        # print(f"self.cfg.task.arch: ")
        # print(f"{self.cfg.task.arch}")
        #for k in article:#tqdm(article):
        # print("next(article)")
        id = 0
        # print(f"next(article): {next(article)}")
        # print(f"len(article): {len(article)}")
        for k in article:
            # print("inside article!")
            # print(f"k['net_input']['src_tokens']: {k['net_input']['src_tokens']}")
            # print(f"self.cfg.model.arch: {self.cfg.model.arch}")
            # print(f"article id: {id}")
            # if id == 3013:
            #     print("skipping 3013")
            #     continue
            # print(f"k['net_input']['src_tokens']: {k['net_input']['src_tokens']}")
            sent_repr = None
            if self.cfg.model.arch == "lstm":  # if the model architecture is LSTM 
                # replaced self.cfg.task.arch
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
                    # print("In the lstm representation",sent_repr)
                elif representation == 'embed':
                    # print("Collecting Embedding")
                    hidden_embed = output['encoder_out'][0]
                    # print(hidden_embed)tr
                    if mean:
                        sent_repr = torch.mean(hidden_embed, dim=0)
                    else:
                        sent_repr = torch.sum(hidden_embed, dim=0)
            elif self.cfg.model.arch == "transformer":
                # print("In the transformer representation")
                if representation == 'memory':
                    with torch.no_grad():
                        # print(f"k['net_input']['src_tokens']: {k['net_input']['src_tokens']}")
                        # print(f"k['net_input']['src_lengths']: {k['net_input']['src_lengths']}")
                        # encoderOut = self.encoder.forward(k['net_input']['src_tokens'].cuda(),
                        #                                   k['net_input']['src_lengths'].cuda())
                        if self.use_gpu and self.mps:
                            # print("going into encoder forward")
                            encoderOut = self.encoder.forward(k['net_input']['src_tokens'].to(self.mps_device),
                                                            k['net_input']['src_lengths'].to(self.mps_device))
                        elif self.use_gpu and self.cuda:
                            # print("going into encoder forward")
                            encoderOut = self.encoder.forward(k['net_input']['src_tokens'].cuda(), k['net_input']['src_lengths'].cuda())
                            # print("got encoderOut")
                        else:
                            encoderOut = self.encoder.forward(k['net_input']['src_tokens'],
                                                          k['net_input']['src_lengths'])
                    # print(f"encoderOut: {encoderOut}")
                    # print(f"len(encoderOut['encoder_out']): {len(encoderOut['encoder_out'])}")
                    hidden_embed = encoderOut['encoder_out'][0]
                    # hidden_embed = getattr(encoderOut, 'encoder_out')  # T x B x C
                    # print(f"hidden_embed: {hidden_embed}")
                    if mean:
                        sent_repr = torch.mean(hidden_embed, dim=0)
                    else:
                        sent_repr = torch.sum(hidden_embed, dim=0)
                elif representation == 'embed':
                    # print(f"k['net_input']['src_tokens']: {k['net_input']['src_tokens']}")
                    # print(f"k['net_input']['src_lengths']: {k['net_input']['src_lengths']}")
                    # print("going into encoder forward emb")
                    # print(f"self.usepos: {self.usepos}")
                    with torch.no_grad():
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
                        # print(f"type(input_emb): {type(input_emb)}")
                        # print(f"self.cuda: {self.cuda}")
                        
                        if self.mps:
                            input_emb = input_emb.to(self.mps_device)
                        if self.cuda:
                            input_emb = input_emb.cuda()

                    #input_emb = getattr(encoderOut, 'encoder_embedding')  # B x T x C
                    # print(f"input_emb.size(): {input_emb.size()}")
                    input_emb = input_emb.transpose(0, 1)
                    if mean:
                        sent_repr = torch.mean(input_emb, dim=0)
                    else:
                        sent_repr = torch.sum(input_emb, dim=0)
            if self.cfg.model.arch == "transformer":
                # print(f"inside modeltype == transformer")
                
                for i in range(k['net_input']['src_tokens'].shape[0]):
                    # print(f"i : {i}")
                    # print(f"k['net_input']['src_tokens'][i]: {k['net_input']['src_tokens'][i]}")
                    # print(f"rang(i): {range(k['net_input']['src_tokens'].shape[0])}")
                    sents.append((k['net_input']['src_tokens'][i], sent_repr[i]))

            elif self.cfg.model.arch == "lstm":
                for i in range(texts.shape[0]):
                    sents.append((texts[i], sent_repr[i]))
            # print(f"finishing {id}")
            id += 1

        # print(f"len(sents): {len(sents)}")
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
        print("candidate filtering")
        candidates_embed = self.filter_candidates(src2tgt_embed, tgt2src_embed) 
        # candidates_embed: [(src_sent_x, tgt_sent_y, score_xy)]
        # Filter candidates (primary filter), such that only those which are top candidates in
        # both src2tgt and tgt2src direction pass.
        # Create set of hashed pairs (for easy comparison in secondary filter)
        set_embed = set([hash((str(c[0]), str(c[1]))) for c in candidates_embed])
        candidate_pool = set_embed # unique set of hashed (src_sent_x, tgt_sent_y) pairs 
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
        i = 0

        # For each src...
        for src in list(src2tgt.keys()):
            # print(f"src: {src}")
            # sort the dict of dict based on sim scores
            toplist = sorted(src2tgt[src].items(), key=lambda x: x[1], reverse=True)
            # ... get the top scoring tgt
            max_tgt = toplist[0]
            # Get src, tgt and score
            src_tgt_max.add((src, max_tgt[0], max_tgt[1]))
            if second:
                # If high permissibility mode, also get second-best tgt
                second_tgt = toplist[1]
                src_tgt_second.add((src, second_tgt[0], second_tgt[1]))
            i += 1

        # For each tgt...
        i = 0
        for tgt in list(tgt2src.keys()):
            # print(f"tgt {i}")
            toplist = sorted(tgt2src[tgt].items(), key=lambda x: x[1], reverse=True)
            # ... get the top scoring src
            max_src = toplist[0]
            tgt_src_max.add((max_src[0], tgt, max_src[1]))
            i += 1 

        if second:
            # Intersection as defined in medium permissibility mode
            src_tgt = (src_tgt_max | src_tgt_second) & tgt_src_max
            candidates = list(src_tgt)
            return candidates

        # Intersection as defined in low permissibility
        print("Length of s2t max",len(src_tgt_max))
        print("Length of t2s max", len(tgt_src_max))
        # print("Intersection = ",list(src_tgt_max & tgt_src_max))
        candidates = list(src_tgt_max & tgt_src_max)
        return candidates # [(src_x, tgt_y, score_xy)]

    def _get_iterator(self, sent, dictn, max_position, epoch, fix_batches_to_gpus=False, shard_batch_itr=False,disable_iterator_cache=False):
        """
        Creates an iterator object from a text file.
        Args:
            path(str): path to text file to process
        Returns:
            data_iter(.EpochIterator): iterator object
        """
        # get indices ordered by example size
        with numpy_seed(self.cfg.common.seed):
            indices = sent.ordered_indices()
        # filter out examples that are too large
        max_positions = (max_position)
        if max_positions is not None:
            indices = filter_by_size(indices, sent, max_positions, raise_exception=(not True), )
        # create mini-batches with given size constraints
        print(f"self.cfg.comparable.max_sentences: {self.cfg.comparable.max_sentences}")
        max_sentences = self.cfg.comparable.max_sentences  # 30
        print(f"max_sentences: {max_sentences}")
        print(f"self.cfg.dataset.num_workers: {self.cfg.dataset.num_workers}")
        print(f"sent.num_tokens: {sent.num_tokens}")

        batch_sampler = batch_by_size(indices, sent.num_tokens, max_sentences=max_sentences, required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple, )
        # print(f"tuple(batch_sampler): {tuple(batch_sampler)}")
        itrs = EpochBatchIterator(dataset=sent, collate_fn=sent.collater, batch_sampler=batch_sampler, seed=self.cfg.common.seed,num_workers=self.cfg.dataset.num_workers, epoch=epoch, num_shards=self.trainer.data_parallel_world_size if shard_batch_itr else 1, shard_id=self.trainer.data_parallel_rank if shard_batch_itr else 0,)
        #data_iter = itrs.next_epoch_itr(shuffle=False, fix_batches_to_gpus=fix_batches_to_gpus)
        # print(f"itrs.state_dict: {itrs.state_dict()}")
        # print(f"itrs: {itrs}")
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
        logger.info(f"self.cfg.dataset.dataset_impl: raw")
        trainingSetSrc = load_indexed_dataset(articles[0], self.task.src_dict,
                                                         dataset_impl='raw', combine=False,
                                                         default='cached')
        trainingSetTgt = load_indexed_dataset(articles[1], self.task.tgt_dict,
                                                         dataset_impl='raw', combine=False,
                                                         default='cached')
        logger.info(f"trainingSetSrc: {trainingSetSrc}")
        # print("read the text file ")self.args.data +
        # convert the read files to Monolingual dataset to make padding easy
        src_mono = MonolingualDataset(dataset=trainingSetSrc, sizes=trainingSetSrc.sizes, src_vocab=self.   task.src_dict, tgt_vocab=None, shuffle=False, add_eos_for_other_targets=False)
        tgt_mono = MonolingualDataset(dataset=trainingSetTgt, sizes=trainingSetTgt.sizes, src_vocab=self.task.tgt_dict, tgt_vocab=None, shuffle=False, add_eos_for_other_targets=False)
        del trainingSetSrc, trainingSetTgt
        # print("Monolingual data")
        # print(f"src_mono.num_tokens(1): {src_mono.num_tokens(1)}")
        # print(f"tgt_mono.num_tokens(1): {tgt_mono.num_tokens(1)}")
        return src_mono, tgt_mono

    def extract_and_train(self, comparable_data_list, epoch):

        tracemalloc.start()
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
            for ap, article_pair in enumerate(comp_list):
                print(f"on article {ap}")
                cur_article += 1
                articles = article_pair.split(' ')
                # if ap >= 1:
                #     src_sents = torch.load(f"/netscratch/jalota/pickle/{self.temp}/src_sents.pt")
                #     src_embeds = torch.load(f"/netscratch/jalota/pickle/{self.temp}/src_embeds.pt")
                #     tgt_sents = torch.load(f"/netscratch/jalota/pickle/{self.temp}/tgt_sents.pt")
                #     tgt_embeds = torch.load(f"/netscratch/jalota/pickle/{self.temp}/tgt_embeds.pt")
                # print(f"articles: {articles}")
                # print(f"len(articles): {len(articles)}")
                # Discard malaligned documents
                if len(articles) != 2:
                    continue
                #load the dataset from the files for both source and target
                src_mono, tgt_mono = self.getdata(articles)
                # Prepare iterator objects for current src/tgt document
                print(f"self.task.src_dict: {self.task.src_dict}")
                print(f"self.cfg.max_source_positions: {self.cfg.task.max_source_positions}")
                print(f"get iterator")
                src_article = self._get_iterator(src_mono, dictn=self.task.src_dict, max_position=self.cfg.task.max_source_positions, epoch=epoch, fix_batches_to_gpus=False)
                tgt_article = self._get_iterator(tgt_mono, dictn=self.task.tgt_dict, max_position=self.cfg.task.max_target_positions, epoch=epoch, fix_batches_to_gpus=False)

                # Get sentence representations
                try:
                    if self.representations == 'embed-only':
                        print("Using Embeddings only for representation")
                        # C_e
                        itr_src = src_article._get_iterator_for_epoch(epoch=epoch, shuffle=True)
                        itr_tgt = tgt_article._get_iterator_for_epoch(epoch=epoch, shuffle=True)
                        print(f"src article, rep=embed")
                        src_sents += self.get_article_coves(itr_src, representation='embed', mean=False)
                        # print(f"tgt article, rep=embed")
                        # torch.save(src_sents, "/netscratch/jalota/pickle/exp2/src_sents.pt")
                        tgt_sents += self.get_article_coves(itr_tgt, representation='embed', mean=False)
                        # torch.save(tgt_sents, "/netscratch/jalota/pickle/exp2/tgt_sents.pt")
                    else:
                        # C_e and C_h
                        '''it1, it2 = itertools.tee(src_article)
                        it3, it4 = itertools.tee(tgt_article)'''
                        print(f"src article, rep=embed")
                        it = src_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
                        src_embeds += self.get_article_coves(it, representation='embed', mean=False, side='src', use_phrase=self.use_phrase)
                        # torch.save(src_embeds, f"/netscratch/jalota/pickle/{self.temp}/src_embeds.pt")
                        print(f"src article, rep=memory")
                        # del src_embeds
                        it = src_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
                        src_sents += self.get_article_coves(it, representation='memory', mean=False, side='src')
                        # torch.save(src_sents, f"/netscratch/jalota/pickle/{self.temp}/src_sents.pt")
                        # del src_sents
                        print(f"tgt article, rep=embed")
                        it = tgt_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
                        tgt_embeds += self.get_article_coves(it, representation='embed', mean=False, side='tgt', use_phrase=self.use_phrase)
                        # torch.save(tgt_embeds, f"/netscratch/jalota/pickle/{self.temp}/tgt_embeds.pt")
                        # del tgt_embeds
                        print(f"tgt article, rep=memory")
                        it = tgt_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
                        print(f"got it4")
                        tgt_sents += self.get_article_coves(it, representation='memory', mean=False, side='tgt')
                        # torch.save(tgt_sents, f"/netscratch/jalota/pickle/{self.temp}/tgt_sents.pt")
                        # del tgt_sents
                        print(f"done with tgt article, rep=memory")
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

                # src_sents = torch.load(f"/netscratch/jalota/pickle/{self.temp}/src_sents.pt")
                # tgt_sents = torch.load(f"/netscratch/jalota/pickle/{self.temp}/tgt_sents.pt")

                if len(src_sents) < 15 or len(tgt_sents) < 15:
                    #print("Length LEss tahn 15")
                    continue
                print("Proceeding")
                # Score src and tgt sentences
                print("In all we have got ", len(src_sents), "source sentences and ", len(tgt_sents), "target")

                # print(f"src_sents: {src_sents[:4]}")
                
                # get src2gt , tgt2src 
                try:
                    logger.info(f"self.faiss: {self.faiss}")
                    if self.faiss:
                        candidates = self.faiss_sent_scoring_v2(src_sents, tgt_sents)
                        logger.info(f"done with faiss scoring of src sents and tgt sents")
                        del src_sents
                        del tgt_sents
                    
                        # src_embeds = torch.load(f"/netscratch/jalota/pickle/{self.temp}/src_embeds.pt")
                        # tgt_embeds = torch.load(f"/netscratch/jalota/pickle/{self.temp}/tgt_embeds.pt")

                        candidates_embed = self.faiss_sent_scoring_v2(src_embeds, tgt_embeds)
                        logger.info(f"num candidates  = {len(candidates)}")
                        logger.info(f"num candidates_embed  = {len(candidates_embed)}")
                        logger.info(f"done with faiss scoring of src embeds and tgt embeds")
                        embed_comparison_pool = set_embed = set([hash((str(c[0]), str(c[1]))) for c in candidates_embed])
                        # candidates : [(src_sent_x, tgt_sent_y, score_xy)]
                        # logger.info(f"made embed_comparison_pool")
                        if self.write_dual:
                            #print("writing the sentences to file....")
                            self.write_embed_only(candidates, candidates_embed)
                        # Extract parallel samples (secondary filter)
                        logger.info(f"starting to extract parallel sents")
                        self.extract_parallel_sents(candidates, embed_comparison_pool, use_threshold=self.use_threshold)
                    else:
                        src2tgt, tgt2src, similarities, scores = self.score_sents(src_sents, tgt_sents)
                    # src2tgt = { "dis a src sent": {"dis a tg": 0.2, "dis s a TRG": 0.6, "dis": 0.12} }
                    # this score could be from margin / cosine similarity 
                    # similarities containes only sim scores (useless var)
                    # scores is a useless var
                except Exception as e:
                    print('Error occurred in: {}\n'.format(article_pair), flush=True)
                    print(f"e: {e}")
                    print("src_sents")
                    # print(src_sents, flush=True)
                    print("tgt_sents")
                    # print(tgt_sents, flush=True)
                    src_sents = []
                    tgt_sents = []
                    continue
                # print("source 2 target ", src2tgt)
                # Keep statistics
                #epoch_similarities += similarities
                #epoch_scores += scores
                src_sents = []
                tgt_sents = []

                if not self.faiss:
                    try:
                        if self.representations == 'dual':
                            # For dual representation systems, filter C_h...
                            candidates = self.filter_candidates(src2tgt, tgt2src, second=self.second)
                            # candidates : [(src_sent_x, tgt_sent_y, score_xy)]
                            # candidates generated from memory representations
                            # Filter candidates (primary filter), such that only those which are top candidates in 
                            # both src2tgt and tgt2src direction pass.
                            # ...and C_e
                            comparison_pool, cand_embed = self.get_comparison_pool(src_embeds,
                                                                                tgt_embeds)
                            # comparison_pool: unique set of hashed (src_sent_x, tgt_sent_y) pairs 
                            # cand_embed: candidates generated from embedding representations 
                            #             [(src_sent_x, tgt_sent_y, score_xy)]
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
                    self.extract_parallel_sents(candidates, comparison_pool, use_threshold=self.use_threshold)
                    # if phrase extraction is to be used

                logger.info(f"pair bank  = {len(self.similar_pairs.pairs)}")
                # Train on extracted sentences
                self.train(epoch)
                if not self.faiss:
                    del src2tgt, tgt2src
                #gc.collect()
                # Add to leaky code within python_script_being_profiled.py

                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                
            if len(self.similar_pairs.pairs) > 0:
                logger.info("batching and training")
                self.train(epoch, last=True)

        self.accepted_file.close()
        if self.use_phrase == True:
            self.accepted_phrase.close()

        # log end-of-epoch stats
        logger.info("end of epoch {} (average epoch stats below)".format(epoch))
        num_updates = self.trainer.get_num_updates()
        stats = get_training_stats(metrics.get_smoothed_values('train'))
        self.progress.print(stats, tag='train', step=num_updates)
        
        # reset epoch-level meters
        metrics.reset_meters('train')
        end_of_epoch = True
        return num_updates, end_of_epoch
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
        # is_epoch_end = False
        if last is False:
            while self.similar_pairs.contains_batch():
                # print("IT has batch.....")
                # try:
                itrs = self.similar_pairs.yield_batch()
                itr = itrs.next_epoch_itr(shuffle=True, fix_batches_to_gpus=self.cfg.distributed_training.fix_batches_to_gpus)
                itr = GroupedIterator(itr, self.update_freq[-1], skip_remainder_batch=self.cfg.optimization.skip_remainder_batch)
                if self.cfg.common.tpu:
                    itr = utils.tpu_data_loader(itr)
                self.progress = progress_bar.progress_bar(
                    itr,
                    log_format=self.cfg.common.log_format,
                    log_file=self.cfg.common.log_file,
                    log_interval=self.log_interval,
                    epoch=epoch,
                    aim_repo=(
                        self.cfg.common.aim_repo
                        if distributed_utils.is_master(self.cfg.distributed_training)
                        else None
                    ),
                    aim_run_hash=(
                        self.cfg.common.aim_run_hash
                        if distributed_utils.is_master(self.cfg.distributed_training)
                        else None
                    ),
                    aim_param_checkpoint_dir=self.cfg.checkpoint.save_dir,
                    tensorboard_logdir=(
                        self.cfg.common.tensorboard_logdir
                        if distributed_utils.is_master(self.cfg.distributed_training)
                        else None
                    ),
                    default_log_format=("tqdm" if not self.cfg.common.no_progress_bar else "simple"),
                    wandb_project=(
                        self.cfg.common.wandb_project
                        if distributed_utils.is_master(self.cfg.distributed_training)
                        else None
                    ),
                    wandb_run_name=os.environ.get(
                        "WANDB_NAME", os.path.basename(self.cfg.checkpoint.save_dir)
                    ),
                    azureml_logging=(
                        self.cfg.common.azureml_logging
                        if distributed_utils.is_master(self.cfg.distributed_training)
                        else False
                    ),
                )
                self.progress.update_config(_flatten_config(self.cfg))
                # logger.info(f"Start iterating over samples")
                for i, samples in enumerate(self.progress):
                    with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
                        "train_step-%d" % i
                    ):
                        # print("Size of the samples = ",len(samples))
                        log_output = self.trainer.train_step(samples)
                        if log_output is not None: # not OOM, overflow, ...
                             # log mid-epoch stats
                            num_updates = self.trainer.get_num_updates()
                            if num_updates % self.log_interval == 0:
                                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                                self.progress.log(stats, tag="train_inner", step=num_updates)
                                
                                # reset mid-epoch stats after each log interval
                                # the end-of-epoch stats will still be preserved
                                metrics.reset_meters('train_inner')
                # end_of_epoch = not itr.has_next()
                # is_epoch_end = end_of_epoch
        else:
            # numberofex = self.similar_pairs.get_num_examples()
            itrs = self.similar_pairs.yield_batch()
            itr = itrs.next_epoch_itr(shuffle=True, fix_batches_to_gpus=self.cfg.distributed_training.fix_batches_to_gpus)
            itr = GroupedIterator(itr, self.update_freq[-1], skip_remainder_batch=self.cfg.optimization.skip_remainder_batch)
            if self.cfg.common.tpu:
                itr = utils.tpu_data_loader(itr)
            self.progress = progress_bar.progress_bar(
                itr,
                log_format=self.cfg.common.log_format,
                log_file=self.cfg.common.log_file,
                log_interval=self.log_interval,
                epoch=epoch,
                aim_repo=(
                    self.cfg.common.aim_repo
                    if distributed_utils.is_master(self.cfg.distributed_training)
                    else None
                ),
                aim_run_hash=(
                    self.cfg.common.aim_run_hash
                    if distributed_utils.is_master(self.cfg.distributed_training)
                    else None
                ),
                aim_param_checkpoint_dir=self.cfg.checkpoint.save_dir,
                tensorboard_logdir=(
                    self.cfg.common.tensorboard_logdir
                    if distributed_utils.is_master(self.cfg.distributed_training)
                    else None
                ),
                default_log_format=("tqdm" if not self.cfg.common.no_progress_bar else "simple"),
                wandb_project=(
                    self.cfg.common.wandb_project
                    if distributed_utils.is_master(self.cfg.distributed_training)
                    else None
                ),
                wandb_run_name=os.environ.get(
                    "WANDB_NAME", os.path.basename(self.cfg.checkpoint.save_dir)
                ),
                azureml_logging=(
                    self.cfg.common.azureml_logging
                    if distributed_utils.is_master(self.cfg.distributed_training)
                    else False
                ),
            )
            self.progress.update_config(_flatten_config(self.cfg))
            logger.info("Start iterating over samples")
            for i, samples in enumerate(self.progress):
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
            # end_of_epoch = not itr.has_next()
            # is_epoch_end = end_of_epoch
        # return is_epoch_end #end_of_epoch
    
    
    def validate(self, epoch, subsets):
        """Evaluate the model on the validation set(s) and return the losses."""

        if self.cfg.dataset.fixed_validation_seed is not None:
            # set fixed seed for every validation
            utils.set_torch_seed(self.cfg.dataset.fixed_validation_seed)

        self.trainer.begin_valid_epoch(epoch)
        valid_losses = []
        for subset_idx, subset in enumerate(subsets):
            logger.info('begin validation on "{}" subset'.format(subset))

            # Initialize data iterator
            itr = self.trainer.get_valid_iterator(subset).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False  # use a fixed valid set
            )
            if self.cfg.common.tpu:
                itr = utils.tpu_data_loader(itr)
            # print(f"self.cfg.distributed_training: {self.cfg.distributed_training}")
            progress = progress_bar.progress_bar(
                itr,
                log_format=self.cfg.common.log_format,
                log_interval=self.cfg.common.log_interval,
                epoch=epoch,
                prefix=f"valid on '{subset}' subset",
                aim_repo=(
                    self.cfg.common.aim_repo
                    if distributed_utils.is_master(self.cfg.distributed_training)
                    else None
                ),
                aim_run_hash=(
                    self.cfg.common.aim_run_hash
                    if distributed_utils.is_master(self.cfg.distributed_training)
                    else None
                ),
                aim_param_checkpoint_dir=self.cfg.checkpoint.save_dir,
                tensorboard_logdir=(
                    self.cfg.common.tensorboard_logdir
                    if distributed_utils.is_master(self.cfg.distributed_training)
                    else None
                ),
                default_log_format=("tqdm" if not self.cfg.common.no_progress_bar else "simple"),
                wandb_project=(
                    self.cfg.common.wandb_project
                    if distributed_utils.is_master(self.cfg.distributed_training)
                    else None
                ),
                wandb_run_name=os.environ.get(
                    "WANDB_NAME", os.path.basename(self.cfg.checkpoint.save_dir)
                ),
            )

            # create a new root metrics aggregator so validation metrics
            # don't pollute other aggregators (e.g., train meters)
            with metrics.aggregate(new_root=True) as agg:
                for i, sample in enumerate(progress):
                    if (
                        self.cfg.dataset.max_valid_steps is not None
                        and i > self.cfg.dataset.max_valid_steps
                    ):
                        break
                    self.trainer.valid_step(sample)

            # log validation stats
            # only tracking the best metric on the 1st validation subset
            # logger.info(f"subset_idx: {subset_idx}")
            tracking_best = True
            #subset_idx == 0
            stats = get_valid_stats(self.cfg, self.trainer, agg.get_smoothed_values(), tracking_best)

            if hasattr(self.task, "post_validate"):
                # logger.info(f"post_validate = True")
                self.task.post_validate(self.trainer.get_model(), stats, agg)

            # logger.info(f"stats in {subset} subset: {stats}")
            progress.print(stats, tag=subset, step=self.trainer.get_num_updates())

            # logger.info(f"self.cfg.checkpoint.best_checkpoint_metric: {self.cfg.checkpoint.best_checkpoint_metric}")
            # logger.info(f"stats[self.cfg.checkpoint.best_checkpoint_metric]: {stats[self.cfg.checkpoint.best_checkpoint_metric]}")

            valid_losses.append(stats[self.cfg.checkpoint.best_checkpoint_metric])
        return valid_losses

    def save_comp_chkp(self, epoch):
        dirs = self.save_dir + '/' + self.model_name + '_' + str(epoch) + self.src + "-" + self.tgt + ".pt"
        self.trainer.save_checkpoint(dirs, {"train_iterator": {"epoch": epoch}})


def get_valid_stats(
    cfg: DictConfig,
    trainer: Trainer,
    stats: Dict[str, Any],
    tracking_best: bool,
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if tracking_best and hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
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
