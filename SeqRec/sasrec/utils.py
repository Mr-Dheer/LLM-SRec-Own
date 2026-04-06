"""
Data utilities for SASRec and LLM-SRec.

This module provides:
1. Data loading and train/valid/test partitioning (leave-last-out protocol)
2. PyTorch Dataset classes for training, validation, and inference
3. A multi-process batch sampler (WarpSampler) for SASRec pre-training
4. Evaluation functions for computing NDCG@K and HR@K metrics

Leave-last-out evaluation protocol (Section 4 "Evaluation Protocol"):
- For each user, the last item → test set
- The second-to-last item → validation set
- All remaining items → training set
- Test/validation candidate set: 1 positive + 99 random negatives
"""

import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import os
from datetime import datetime
from pytz import timezone
from torch.utils.data import Dataset
from tqdm import tqdm

import pickle


def random_neq(l, r, s):
    """
    Sample a random integer in [l, r) that is NOT in set s.
    Used to generate negative samples (items the user hasn't interacted with).

    Args:
        l: Lower bound (inclusive).
        r: Upper bound (exclusive).
        s: Set of excluded values (user's interacted items).

    Returns:
        A random integer not in s.
    """
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    """
    Background worker function for WarpSampler.

    Continuously generates training samples and puts them in a queue.
    Each sample consists of:
    - user: randomly selected user ID
    - seq: input sequence (right-padded with zeros, most recent items at the end)
    - pos: positive items (the actual next items in the sequence)
    - neg: negative items (randomly sampled non-interacted items)

    The sequence is constructed by walking backward from the second-to-last item:
    seq[i] → pos[i] represents "given item seq[i], predict pos[i] as the next item"
    neg[i] is a random non-interacted item (negative sample for the same position)
    """
    def sample():
        # Randomly select a user with at least 2 interactions
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)   # Input sequence (left-padded with 0s)
        pos = np.zeros([maxlen], dtype=np.int32)   # Positive next items
        neg = np.zeros([maxlen], dtype=np.int32)   # Negative random items
        nxt = user_train[user][-1]                  # Start from the last interacted item
        idx = maxlen - 1                            # Fill from right to left

        ts = set(user_train[user])  # Set of all user's interacted items (for negative sampling)

        # Walk backward through the user's interaction history
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i          # Current item in the sequence
            pos[idx] = nxt        # Next item to predict
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)  # Random negative
            nxt = i               # Move backward
            idx -= 1
            if idx == -1: break   # Sequence is full

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """
    Multi-process batch sampler for SASRec pre-training.

    Spawns n_workers background processes that continuously generate batches
    of training samples and put them in a shared queue. This allows data
    loading to happen in parallel with model training.
    """
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        """Get the next batch from the queue (blocks until available)."""
        return self.result_queue.get()

    def close(self):
        """Terminate all worker processes."""
        for p in self.processors:
            p.terminate()
            p.join()


class SeqDataset(Dataset):
    """
    PyTorch Dataset for LLM-SRec training (used with DataLoader for DDP support).

    For each user, constructs:
    - seq: Input interaction sequence (right-aligned, left-padded with 0s)
    - pos: Positive next items at each position
    - neg: Random negative items at each position

    This is the same data format as WarpSampler but as a proper Dataset
    for compatibility with PyTorch's DataLoader and DistributedSampler.
    """

    def __init__(self, user_train, num_user, num_item, max_len):
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        print("Initializing with num_user:", num_user)

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        user_id = idx + 1  # User IDs are 1-indexed
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)

        nxt = self.user_train[user_id][-1]  # Start from last item
        length_idx = self.max_len - 1

        # Set of user's interactions (for negative sampling exclusion)
        ts = set(self.user_train[user_id])

        # Build sequence by walking backward through interaction history
        for i in reversed(self.user_train[user_id][:-1]):
            seq[length_idx] = i
            pos[length_idx] = nxt
            if nxt != 0: neg[length_idx] = random_neq(1, self.num_item + 1, ts)
            nxt = i
            length_idx -= 1
            if length_idx == -1: break

        return user_id, seq, pos, neg


class SeqDataset_Inference(Dataset):
    """
    PyTorch Dataset for test set evaluation in LLM-SRec.

    For each user, constructs:
    - seq: Training items + validation item (input for predicting test item)
    - pos: The test item (ground truth next item)
    - neg: 1 random negative item

    During evaluation, 99 additional negatives are sampled to form a 100-item
    candidate set (1 positive + 99 negatives) for ranking.
    """

    def __init__(self, user_train, user_valid, user_test, use_user, num_item, max_len):
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test
        self.num_user = len(use_user)
        self.num_item = num_item
        self.max_len = max_len
        self.use_user = use_user  # Subset of users to evaluate on
        print("Initializing with num_user:", self.num_user)

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        user_id = self.use_user[idx]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1

        # Include the validation item as the most recent item in the sequence
        try:
            seq[idx] = self.user_valid[user_id][0]
            idx -= 1
        except:
            idx = self.max_len - 1

        # Fill remaining positions with training items (most recent first)
        for i in reversed(self.user_train[user_id]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        # Exclude all known interacted items from negative sampling
        rated = set(self.user_train[user_id])
        rated.add(0)

        pos = self.user_test[user_id][0]  # Ground truth: the test item

        # Sample 1 negative item (more negatives added during evaluation)
        neg = []
        for _ in range(1):
            t = np.random.randint(1, self.num_item + 1)
            while t in rated: t = np.random.randint(1, self.num_item + 1)
            neg.append(t)
        neg = np.array(neg)
        return user_id, seq, pos, neg


class SeqDataset_Validation(Dataset):
    """
    PyTorch Dataset for validation set evaluation in LLM-SRec.

    Similar to SeqDataset_Inference but:
    - seq: Only training items (no validation item in input)
    - pos: The validation item (ground truth)
    - Used during training for early stopping decisions.
    """

    def __init__(self, user_train, user_valid, use_user, num_item, max_len):
        self.user_train = user_train
        self.user_valid = user_valid
        self.num_user = len(use_user)
        self.num_item = num_item
        self.max_len = max_len
        self.use_user = use_user
        print("Initializing with num_user:", self.num_user)

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        user_id = self.use_user[idx]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1

        # Fill with training items only (validation item is the target)
        for i in reversed(self.user_train[user_id]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(self.user_train[user_id])
        rated.add(0)
        pos = self.user_valid[user_id][0]  # Ground truth: the validation item

        # Sample 1 negative item
        neg = []
        for _ in range(1):
            t = np.random.randint(1, self.num_item + 1)
            while t in rated: t = np.random.randint(1, self.num_item + 1)
            neg.append(t)
        neg = np.array(neg)
        return user_id, seq, pos, neg


def data_partition(fname, args, path=None):
    """
    Load and partition the dataset into train/valid/test splits.

    Reads pre-processed text files (generated by data_preprocess.py) where each
    line contains "user_id item_id". The data is already split into train/valid/test
    files following the leave-last-out protocol.

    Args:
        fname: Dataset name (e.g., 'Movies_and_TV').
        args: Configuration arguments.
        path: Optional custom path to the data files.

    Returns:
        [user_train, user_valid, user_test, usernum, itemnum, eval_set]
        where eval_set = [set of users with valid data, set of users with test data]
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)

    # Read train, valid, and test files
    for t in ['train', 'valid', 'test']:
        if path == None:
            f = open(f'./../data_{args.dataset}/{fname}_{t}.txt', 'r')
        else:
            f = open(path + f'_{t}.txt', 'r')

        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            if t == 'train':
                user_train[u].append(i)
            elif t == 'valid':
                user_valid[u].append(i)
            elif t == 'test':
                user_test[u].append(i)

    # Sets of users that have validation/test data (for evaluation sampling)
    eval_set = [set(user_valid.keys()), set(user_test.keys())]

    return [user_train, user_valid, user_test, usernum, itemnum, eval_set]


def evaluate(model, dataset, args, mode=1, ranking=10):
    """
    Evaluate the SASRec model on the test set using NDCG@K and HR@K.

    Evaluation protocol:
    1. For each user, construct input sequence (train + valid items).
    2. Create candidate set: 1 positive (test item) + 99 random negatives.
    3. Score all candidates using the model.
    4. Compute the rank of the positive item.
    5. Calculate NDCG@K and HR@K.

    Args:
        model: Trained SASRec model.
        dataset: [train, valid, test, usernum, itemnum, eval_set].
        args: Configuration arguments.
        mode: 0 = validation set, 1 = test set.
        ranking: K for NDCG@K and HR@K (default: 10).

    Returns:
        (NDCG@K, HR@K) averaged over all evaluated users.
    """
    [train, valid, test, usernum, itemnum, eval_set] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0  # Hit Rate (HR)
    valid_user = 0.0

    # Cap evaluation at 10,000 users for efficiency
    eval_set = eval_set[mode]
    if len(eval_set) > 10000:
        users = random.sample(list(eval_set), 10000)
    else:
        users = list(eval_set)

    num_candi = 99  # Number of negative candidates
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        # Build input sequence: training items + validation item
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        try:
            seq[idx] = valid[u][0]  # Add validation item
            idx -= 1
        except:
            idx = args.maxlen - 1

        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]  # Positive item (test item) at index 0

        # Sample 99 negative items not in user's history
        his = train[u] + valid[u] + item_idx
        his = set(his)
        his.add(0)
        items = set([i for i in range(1, itemnum + 1)])
        items = list(items.difference(his))
        if len(items) > num_candi:
            samples = random.sample(items, num_candi)
            item_idx = item_idx + samples
        else:
            item_idx = item_idx + items

        # Shuffle candidate order to avoid position bias
        l_ = [i for i in range(len(item_idx))]
        random.shuffle(l_)

        # Get model predictions (negated for ascending sort)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        # Find the rank of the positive item (index 0 in original order)
        rank = predictions[l_].argsort().argsort()[l_.index(0)].item()
        valid_user += 1

        # Compute NDCG@K and HR@K
        if rank < ranking:
            NDCG += 1 / np.log2(rank + 2)  # NDCG uses log2(rank+2) since rank is 0-indexed
            HT += 1  # Hit if positive item is in top-K

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args):
    """
    Evaluate the SASRec model on the validation set.
    Same as evaluate() but uses validation items as targets instead of test items,
    and only uses training items as input (no validation item in the sequence).
    """
    [train, valid, test, usernum, itemnum, eval_set] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0

    eval_set = eval_set[0]  # Use validation set users
    if len(eval_set) > 10000:
        users = random.sample(list(eval_set), 10000)
    else:
        users = list(eval_set)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        # Build input sequence from training items only
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]  # Positive item is the validation item

        # Sample 100 negative items
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return NDCG / valid_user, HT / valid_user


def save_eval(model, dataset, args):
    """Evaluate the model and save results to a text file."""
    model.eval()

    with torch.no_grad():
        print('Evaluate')
        t_test = evaluate(model, dataset, args)
        print('\n')
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    with open(f'./../data_{args.dataset}/Results.txt', 'w') as f:
        sys.stdout = f
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
        sys.stdout = sys.__stdout__
