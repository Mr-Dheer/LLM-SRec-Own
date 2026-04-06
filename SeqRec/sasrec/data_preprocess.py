"""
Data Preprocessing for Amazon Review Datasets.

Downloads and preprocesses the Amazon Reviews 2023 dataset for use with
SASRec and LLM-SRec. Uses the 5-core version (users and items with at least
5 interactions) with the leave-last-out split provided by McAuley Lab.

The preprocessing pipeline:
1. Load the HuggingFace dataset (5core_last_out_{dataset_name}).
2. Load item metadata (titles, descriptions) from the raw_meta split.
3. Build user/item ID mappings (original string IDs → integer IDs starting from 1).
4. Subsample users based on dataset-specific ratios (e.g., 5% for Movies_and_TV).
5. Filter users with >4 interactions and items with >4 interactions.
6. Write train/valid/test files in "user_id item_id" format.
7. Save item metadata (title, description, timestamps) as a pickled dictionary.

Output files (written to ../data_{dataset_name}/):
- {dataset_name}_train.txt: Training interactions
- {dataset_name}_valid.txt: Validation interactions (second-to-last item per user)
- {dataset_name}_test.txt: Test interactions (last item per user)
- text_name_dict.json.gz: Pickled dict with keys 'title', 'description', 'time'
  mapping item IDs to their metadata (used by LLM-SRec for prompt construction)
"""

import os
import os.path
import gzip
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import random
import pickle
from datasets import load_dataset


def preprocess_raw_5core(fname):
    """
    Download and preprocess the Amazon Reviews 2023 dataset.

    Args:
        fname: Amazon dataset category name (e.g., 'Movies_and_TV', 'Electronics',
               'Industrial_and_Scientific', 'CDs_and_Vinyl').

    Steps:
    1. Load interaction data and item metadata from HuggingFace.
    2. Map string user/item IDs to sequential integers.
    3. Subsample users (dataset-specific ratio to keep manageable size).
    4. Apply 5-core filtering (>4 interactions for both users and items).
    5. Re-map IDs to final sequential integers.
    6. Save train/valid/test splits and metadata.
    """

    random.seed(0)
    np.random.seed(0)

    # Load the pre-split dataset from HuggingFace (already split into train/valid/test
    # using leave-last-out protocol)
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"5core_last_out_{fname}", trust_remote_code=True)
    # Load item metadata (titles, descriptions) for prompt construction
    meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{fname}", trust_remote_code=True)

    # Build a lookup dictionary: ASIN → [title, description]
    print("Load Meta Data")
    meta_dict = {}
    for l in tqdm(meta_dataset['full']):
        meta_dict[l['parent_asin']] = [l['title'], l['description']]
    del meta_dataset

    # ===== Phase 1: Build initial user/item ID mappings =====
    usermap = dict()    # Original user_id string → integer ID
    usernum = 0
    itemmap = dict()    # Original ASIN string → integer ID
    itemnum = 0
    User = defaultdict(list)     # All interactions per user (across splits)
    User_s = {'train': defaultdict(list), 'valid': defaultdict(list), 'test': defaultdict(list)}  # Per-split interactions
    id2asin = dict()             # Integer item ID → original ASIN (for metadata lookup)
    time_dict = defaultdict(dict)  # item_id → {user_id: timestamp}

    for t in ['train', 'valid', 'test']:
        d = dataset[t]

        for l in tqdm(d):
            user_id = l['user_id']
            asin = l['parent_asin']

            # Map string user ID to integer
            if user_id in usermap:
                userid = usermap[user_id]
            else:
                usernum += 1
                userid = usernum
                usermap[user_id] = userid

            # Map ASIN to integer item ID
            if asin in itemmap:
                itemid = itemmap[asin]
            else:
                itemnum += 1
                itemid = itemnum
                itemmap[asin] = itemid

            User[userid].append(itemid)
            User_s[t][userid].append(itemid)
            id2asin[itemid] = asin
            time_dict[itemid][userid] = l['timestamp']

    # ===== Phase 2: Subsample users =====
    # Different datasets need different sampling ratios to maintain manageable size
    # while keeping enough data for meaningful evaluation
    sample_size = int(len(User.keys()))
    print('num users raw', sample_size)
    sample_rate = {
        'Movies_and_TV': 0.05,              # 5% of users
        'Electronics': 0.05,                  # 5% of users
        'Industrial_and_Scientific': 1.0,     # All users (small dataset)
        'CDs_and_Vinyl': 0.33,               # 33% of users
    }

    sample_ratio = sample_rate[fname]
    use_key = random.sample(list(User.keys()), int(sample_size * sample_ratio))

    print('num sample user', len(use_key))

    # ===== Phase 3: Count interactions for 5-core filtering =====
    # Count how many interactions each user and item has in the sampled subset
    CountU = defaultdict(int)  # User interaction counts
    CountI = defaultdict(int)  # Item interaction counts

    usermap_final = dict()     # Re-mapping to final sequential IDs
    itemmap_final = dict()
    usernum_final = 0
    itemnum_final = 0
    use_key_dict = defaultdict(int)   # Fast lookup for sampled users
    use_train_dict = defaultdict(int)  # Track which users have valid training data

    for key in use_key:
        use_key_dict[key] = 1
        for t in ['train', 'valid', 'test']:
            for i_ in User_s[t][key]:
                CountI[i_] += 1
                CountU[key] += 1

    # ===== Phase 4: Write filtered data and metadata =====
    # text_dict stores item metadata used by LLM-SRec for prompt construction:
    # - 'title': item_id → title string
    # - 'description': item_id → description string
    # - 'time': item_id → {user_id: timestamp} (for chronological ordering in prompts)
    text_dict = {'time': defaultdict(dict), 'description': {}, 'title': {}}

    for t in ['train', 'valid', 'test']:
        d = dataset[t]
        use_id = defaultdict(int)  # Track processed users to avoid duplicates
        f = open(f'./../data_{fname}/{fname}_{t}.txt', 'w')

        for l in tqdm(d):
            user_id = l['user_id']
            asin = l['parent_asin']
            user_id_ = usermap[user_id]

            # Skip duplicate user entries (only process first occurrence)
            if use_id[user_id_] == 0:
                use_id[user_id_] = 1
                pass
            else:
                continue

            # Apply filters: user must be sampled AND have >4 total interactions
            if use_key_dict[user_id_] == 1 and CountU[user_id_] > 4:

                # Filter items: only keep items with >4 interactions (5-core)
                use_items = []
                for it in User_s[t][user_id_]:
                    if CountI[it] > 4:
                        use_items.append(it)

                if t == 'train':
                    # Training split: only include users with >4 training items
                    if len(use_items) > 4:
                        use_train_dict[user_id_] = 1

                        # Re-map user to final sequential ID
                        if user_id_ in usermap_final:
                            userid = usermap_final[user_id_]
                        else:
                            usernum_final += 1
                            userid = usernum_final
                            usermap_final[user_id_] = userid

                        # Process each training item
                        for it in use_items:
                            # Re-map item to final sequential ID
                            if it in itemmap_final:
                                itemid = itemmap_final[it]
                            else:
                                itemnum_final += 1
                                itemid = itemnum_final
                                itemmap_final[it] = itemid

                            # Store item metadata for LLM prompt construction
                            d_ = meta_dict[id2asin[it]][1]  # Description
                            if d_ == None:
                                text_dict['description'][itemid] = 'Empty description'
                            elif len(d_) == 0:
                                text_dict['description'][itemid] = 'Empty description'
                            else:
                                text_dict['description'][itemid] = d_[0]

                            texts = meta_dict[id2asin[it]][0]  # Title
                            if texts == None:
                                text_dict['title'][itemid] = 'Empty title'
                            elif len(texts) == 0:
                                text_dict['title'][itemid] = 'Empty title'
                            else:
                                texts_ = texts
                                text_dict['title'][itemid] = texts_

                            # Store interaction timestamp for prompt chronological ordering
                            text_dict['time'][itemid][userid] = time_dict[it][user_id_]

                            # Write "user_id item_id" to the output file
                            f.write('%d %d\n' % (userid, itemid))
                else:
                    # Valid/test splits: only include users who have valid training data
                    if use_train_dict[user_id_] == 1:
                        for it in User_s[t][user_id_]:
                            if CountI[it] > 4:
                                if user_id_ in usermap_final:
                                    userid = usermap_final[user_id_]
                                else:
                                    usernum_final += 1
                                    userid = usernum_final
                                    usermap_final[user_id_] = userid
                                if it in itemmap_final:
                                    itemid = itemmap_final[it]
                                else:
                                    itemnum_final += 1
                                    itemid = itemnum_final
                                    itemmap_final[it] = itemid

                                d_ = meta_dict[id2asin[it]][1]
                                if d_ == None:
                                    text_dict['description'][itemid] = 'Empty description'
                                elif len(d_) == 0:
                                    text_dict['description'][itemid] = 'Empty description'
                                else:
                                    text_dict['description'][itemid] = d_[0]
                                texts = meta_dict[id2asin[it]][0]

                                if texts == None:
                                    text_dict['title'][itemid] = 'Empty title'
                                elif len(texts) == 0:
                                    text_dict['title'][itemid] = 'Empty title'
                                else:
                                    texts_ = texts
                                    text_dict['title'][itemid] = texts_
                                text_dict['time'][itemid][userid] = time_dict[it][user_id_]

                                f.write('%d %d\n' % (userid, itemid))
        f.close()

        # Save item metadata dictionary (used by LLM-SRec for prompt construction)
        with open(f'./../data_{fname}/text_name_dict.json.gz', 'wb') as tf:
            pickle.dump(text_dict, tf)

    del text_dict
    del meta_dict
    del dataset
