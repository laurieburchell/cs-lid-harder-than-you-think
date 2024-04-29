#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:47:04 2023

@author: laurie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab, Vocab
from nltk.util import everygrams
from nltk.tokenize.api import TokenizerI
from tqdm import tqdm
from collections import Counter, OrderedDict, namedtuple
from typing import Iterable, Generator
import numpy as np
from sklearn.metrics import classification_report
import wandb
import preprocessor as p  # tweet preprocessing
from sacremoses import MosesPunctNormalizer
import cleantext
import emoji


class SlowtextDataset(Dataset):
    """Loads fasttext-style files for language ID"""
    def __init__(self, data_filename, tokeniser) -> None:
        super().__init__()  # inherit all methods and properties from Dataset class
        self.data_filename = data_filename
        self.sents, self.labels = [], []
        self.tokeniser = tokeniser
        self.vocabulary, self.ngram2hash_dict = None, dict()
        self.label2index_dict = dict()

    def _load_fasttext_data(self) -> tuple[list, list]:
        """Loads data from file where each line is in fastText format."""
        with open(self.data_filename, 'r') as f:
            data = [line.rstrip().split(' ', 1) for line in tqdm(
                f.readlines(), desc="loading data")]
        labels, sents = zip(*data)
        labels = [y.split('__')[2] for y in labels]  # remove fasttext labels
        # perform data cleaning on sents
        mpn = MosesPunctNormalizer()
        p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.ESCAPE_CHAR)  # too zealous cleaning removes non-Latin chars
        clean_sents = []
        for s in tqdm(sents, desc="cleaning sentences"):
            try:
                _s = p.clean(s)
                _s = cleantext.clean(_s, 
                                     extra_spaces=True, 
                                     lowercase=True,
                                     numbers=True, 
                                     reg=r"#", reg_replace="")
                _s = emoji.replace_emoji(_s, "")  
                _s = mpn.normalize(_s)  # normalise punct à la Moses
            except ValueError:
                _s = ""
            clean_sents.append(_s)
        return clean_sents, labels
    
    def yield_words(self, data_iter: Iterable) -> Generator:
        """Generator yielding all words in the provided data iterator"""
        for sent in data_iter:
            words = self.tokeniser(sent)
            yield words

    def yield_ngrams(self, words: list) -> Generator:
        """Generator yielding all n-grams in the list of words"""
        for word in words:
            word = '<' + word + '>'  # boundary symbols
            if len(word) < self.minn+2:  # not point getting grams for too-short words
                yield []
            else:
                grams = everygrams(word, min_len=self.minn, max_len=self.maxn)
                joined_grams = ["".join(x) for x in grams]  # everygrams produces tuples
                yield joined_grams
  
    def label2index(self, label: str) -> int:
        """Helper function to convert label to its index"""
        return self.label2index_dict[label]
    
    def sent2indices(self, sent: str) -> list[int]:
        """Convert sentence to list of ints representing bag of word and ngram features"""
        indices = []
        # add word indices to output
        word_list = next(self.yield_words([sent]))  # keep processing the same as when building vocab
        indices += [i for i in self.vocabulary(word_list) if i != 0]  # filter unks
        # look up n-grams in hash table
        ngram_list = next(self.yield_ngrams(word_list))
        ngram_lookups = ["<" + str(self.ngram2hash_dict.get(ngram, -1)) + ">" 
                             for ngram in ngram_list]  # gets int rep of hash
        # lookup these reps in vocab
        indices += [i for i in self.vocabulary(ngram_lookups) if i != 0]  # filter unks again
        return indices
    
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx: int) -> tuple[str, str]:
        """Returns a item from the dataset based on the given index"""
        return self.labels[idx], self.sents[idx]


class SlowtextTrainingDataset(SlowtextDataset):
    """Defines dataset for training Slowtext models"""
    def __init__(self, data_filename, tokeniser, minn, maxn, min_freq, num_buckets) -> None:
        super().__init__(data_filename, tokeniser)
        self.sents, self.labels = self._load_fasttext_data()
        self.min_freq = min_freq
        self.num_buckets = num_buckets
        self.minn = minn
        self.maxn = maxn
        self.vocabulary, self.ngram2hash_dict = self._build_vocab_and_ngram2hash()
        self.label2index_dict = self._build_label2index()

    def _build_vocab_and_ngram2hash(self) -> tuple[Vocab, dict]:
        """Builds vocabulary of words which are seen at least min_freq times plus the bucket numbers of their n-grams"""
        # count frequency of words in vocab and filter to those appearing at least min_freq times
        word_g = self.yield_words(self.sents)
        word_counter = Counter()
        for words in tqdm(word_g, desc="calculating word frequencies...",total=len(self.sents)):
            word_counter.update(words)
        words_by_freq = sorted(word_counter.items(), key=lambda x: (-x[1], x[0]))
        ordered_dict = OrderedDict(words_by_freq)  # vocab has to be built from ordered dict of words and freqs
        # build vocab of just words from OrderedDict
        v = vocab(ordered_dict, min_freq=self.min_freq, specials=["<unk>"], special_first=True)
        print(f"vocab len: {len(v)}")
        v.set_default_index(v["<unk>"])  # index will be 0

        # make hashmap for ngrams
        words = v.get_itos()  # list of included words
        ngram_g = self.yield_ngrams(words[1:])  # skip unk
        ngram2hash = dict()
        for g in tqdm(ngram_g, desc="generating hash lookup dict for n-grams...", total=len(words)):
            if len(g) > 0:
                h = {x: hash(x) % self.num_buckets for x in g}
                ngram2hash.update(h)
        # update vocabulary with string representation of hashes
        collisions = 0
        for h in tqdm(ngram2hash.values(), 
                        desc="updating vocab with string representations of n-gram hashes", 
                        total=len(ngram2hash)):
            try:
                v.append_token("<" + str(h) + ">")
            except RuntimeError:  # raised by hash collision
                collisions += 1
                continue
        print(f"there were {collisions} collisions")
        return v, ngram2hash
    
    def _build_label2index(self) -> dict:
        """Build dict of form {label: idx} to map labels to indices"""
        uniq_labels = set(self.labels)
        d = zip(list(uniq_labels), list(range(len(uniq_labels))))  # assign idx to each label
        return dict(d)
    

class SlowtextEvalDataset(SlowtextDataset):
    """Defines dataset used to evaluate SlowtextModel"""
    def __init__(self, data_filename, tokeniser, train_dataset) -> None:
        super().__init__(data_filename, tokeniser)
        self.sents, self.labels = self._load_fasttext_data()
        self.minn = train_dataset.minn   # these are defined by training data
        self.maxn = train_dataset.maxn
        self.vocabulary = train_dataset.vocabulary
        self.ngram2hash_dict = train_dataset.ngram2hash_dict
        self.label2index_dict = train_dataset.label2index_dict

    
class SlowtextTestDataset(SlowtextDataset):
    """Defines test-time dataset of sentences to classify with no labels"""
    def __init__(self, data_filename, tokeniser, train_dataset) -> None:
        super().__init__(data_filename, tokeniser)
        self.sents = self._load_test_data()
        self.minn = train_dataset.minn   # these are defined by training data
        self.maxn = train_dataset.maxn
        self.vocabulary = train_dataset.vocabulary
        self.ngram2hash_dict = train_dataset.ngram2hash_dict
        self.label2index_dict = train_dataset.label2index_dict


    def _load_test_data(self):
        """Loads test dataset from file with one sentence per line"""
        # load raw data
        with open(self.data_filename, 'r') as f:
            raw_sents = [line.strip() for line in tqdm(f.readlines(), desc="loading test data")]
        # perform data cleaning on sents
        mpn = MosesPunctNormalizer()
        p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.ESCAPE_CHAR)  # too zealous cleaning removes non-Latin chars
        clean_sents = []
        for s in tqdm(raw_sents, desc="cleaning sentences"):
            try:
                _s = p.clean(s)
                _s = cleantext.clean(_s, 
                                     extra_spaces=True, 
                                     lowercase=True,
                                     numbers=True, 
                                     reg=r"#", reg_replace="")
                _s = emoji.replace_emoji(_s, "")  
                _s = mpn.normalize(_s)  # normalise punct à la Moses
            except ValueError:
                _s = ""
            clean_sents.append(_s)
        return clean_sents
    
    def __getitem__(self, idx: int) -> tuple[str, str]:
        """Returns a item from the dataset based on the given index"""
        return "", self.sents[idx]  # keep it as dummy tuple for consistency


class SlowtextClassifier(nn.Module):
    """Reimplementation of fasttext classifer for language identification"""
    def __init__(self, train_data: SlowtextTrainingDataset, emb_dim: int) -> None:
        super().__init__()  # names of layers reflect Joulin et al. (2017)
        self.emb_dim = emb_dim
        self.A = nn.EmbeddingBag(num_embeddings=len(train_data.vocabulary),
                                 embedding_dim=emb_dim,
                                 max_norm=1.0, norm_type=2, # want to create "normalised bag of features"
                                 mode="max", sparse=False)  # better cross-class perf: 'max > 'sum' > 'mean'
        self.B = nn.Linear(in_features=emb_dim, out_features=len(train_data.label2index_dict))
        self.init_weights()

    def init_weights(self):
        """Initialises weights"""
        initrange = 1/self.emb_dim
        nn.init.uniform_(self.A.weight, a=0, b=initrange)
        nn.init.uniform_(self.B.weight, a=0, b=initrange)
        nn.init.uniform_(self.B.bias, a=0, b=initrange)

    def forward(self, _input, offsets):
        """Model defined in Joulin et al. (2017)"""
        textrep = self.A(_input, offsets)  # average word representations into text representation
        logits = self.B(textrep)  # feed text representation into linear classifier
        return logits
    

def build_dataloader(sd: SlowtextDataset, device, **params) -> DataLoader:
    """Build dataloader for slowtext data"""

    def collate_batch(batch: Iterable):
        """Packs labels and sentences into batches, keeping track of offsets"""
        label_batch, sent_batch, offsets = [], [], [0]
        for (label, sent) in batch:
            label_batch.append(sd.label2index(label))  # lookup label to get id
            # convert each sentence to bag of indices for embedding, based on words and ngrams
            processed_sent = torch.tensor(sd.sent2indices(sent), dtype=torch.int64)
            sent_batch.append(processed_sent)
            offsets.append(processed_sent.size(0))  # keep track of len of processed sentence
        # once all labels and sents in batch converted to ints, convert to tensors and move to gpu
        label_batch = torch.tensor(label_batch, dtype=torch.int64)
        sent_batch = torch.cat(sent_batch)  # concats all the tensors of sents
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)  # take cumsum to track where sents start
        return label_batch.to(device), sent_batch.to(device), offsets.to(device)
    
    return DataLoader(sd, **params, collate_fn=collate_batch)


def build_test_dataloader(sd: SlowtextDataset, device, **params) -> DataLoader:
    """Build dataloader for test data (blank label)"""

    def collate_batch(batch: Iterable):
        """Packs sentences into batches, keeping track of offsets"""
        sent_batch, offsets = [], [0]
        for (_, sent) in batch:  # expect blank label
            # convert each sentence to bag of indices for embedding, based on words and ngrams
            processed_sent = torch.tensor(sd.sent2indices(sent), dtype=torch.int64)
            sent_batch.append(processed_sent)
            offsets.append(processed_sent.size(0))  # keep track of len of processed sentence
        # once all sents in batch converted to ints, convert to tensors and move to gpu
        sent_batch = torch.cat(sent_batch)  # concats all the tensors of sents
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)  # take cumsum to track where sents start
        return sent_batch.to(device), offsets.to(device)
    
    return DataLoader(sd, **params, collate_fn=collate_batch)
    

def train_loop(dataloader: DataLoader, model: SlowtextClassifier, 
               loss_fn: torch.nn.modules.loss, optimiser: torch.optim.Optimizer,
               epoch: int):
    """Train SlowtextClassifier for one epoch"""
    num_classes = len(dataloader.dataset.label2index_dict)
    model.train()
    # iterates through batches, updating params for each
    for batch_num, (gold_labels, sents, offsets) in enumerate(dataloader):
        predicted_labels = model(sents, offsets)
        if loss_fn._get_name() == 'BCEWithLogitsLoss':  # single label targets need to be same shape&type as multi-label outputs
            # we assume probability of gold label is 1.0
            gold_labels = torch.nn.functional.one_hot(gold_labels, num_classes=num_classes).float()  # gold labels are probabilities per class
        loss = loss_fn(predicted_labels, gold_labels)
        # backprogation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


def get_predictions_and_gold_labels(dataloader: DataLoader, model: SlowtextClassifier) -> namedtuple:
    """Get single-label predictions and gold labels for sentences in dataloader using SlowtextClassifier"""
    model.eval()
    with torch.no_grad():
        # get labels
        gold_labels = []
        predictions = []
        for idx, (gold_label, sent, offsets) in enumerate(dataloader):
            gold_labels += gold_label.to("cpu")  # will be blank for test set
            logits = model(sent, offsets)
            preds = logits.argmax(axis=1)  # get single label out for each sent
            predictions += preds.to("cpu")
        # put labels on output because I'm forgetful
        ModelOutput = namedtuple('model_output', ['predictions', 'gold_labels'])
        output = ModelOutput(np.array(predictions), np.array(gold_labels))

        return output
    

def evaluate(dataloader: DataLoader, model: SlowtextClassifier, as_dict: bool=False):
    """Calculates classification report for data, returns as str (default) or dict"""
    model.eval()
    model_output = get_predictions_and_gold_labels(dataloader, model)
    # get sorted list of class labels
    sorted_class_labels = sorted(dataloader.dataset.label2index_dict, 
                                 key=lambda x: dataloader.dataset.label2index_dict[x])
    report = classification_report(model_output.gold_labels, model_output.predictions, 
                                     digits=3, target_names=sorted_class_labels,
                                     output_dict=as_dict, zero_division=0)  # I expect zero division
    
    return report