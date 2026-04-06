"""
SASRec Pre-training Script.

This script trains the SASRec (Self-Attentive Sequential Recommendation) model
on Amazon Review datasets. The trained SASRec model serves as the frozen CF-SRec
backbone in LLM-SRec.

Training procedure:
1. Download and preprocess the dataset if not already done.
2. Initialize SASRec with the specified hyperparameters.
3. Train using BPR (Bayesian Personalized Ranking) loss:
   - Positive items (actual next items) should score higher than
   - Negative items (randomly sampled non-interacted items)
4. Save the trained model checkpoint (.pth file) after the last epoch.
5. Evaluate on the test set using NDCG@10 and HR@10.

The saved checkpoint is later loaded by LLM-SRec (via models/recsys_model.py)
as a frozen model — its item embeddings and sequence encoder are used but
never updated during LLM-SRec training.

Usage:
    python main.py --dataset Movies_and_TV --device 0 --num_epochs 200
    python main.py --dataset Movies_and_TV --device 0 --inference_only --state_dict_path <path>

Hyperparameters (matching Appendix D in the paper):
    - hidden_units: 64 (item embedding dimension d)
    - maxlen: 128 (maximum sequence length)
    - batch_size: 128
    - num_blocks: 2 (number of self-attention layers)
    - num_heads: 1
    - dropout_rate: 0.1
    - lr: 0.001
"""

import os
import time
import torch
import argparse
import numpy as np
import sys

from model import SASRec
from data_preprocess import *
from utils import *

from tqdm import tqdm

# ===== Argument Parser =====
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)                               # Amazon dataset name (e.g., 'Movies_and_TV')
parser.add_argument('--batch_size', default=128, type=int)                    # Training batch size
parser.add_argument('--lr', default=0.001, type=float)                        # Learning rate
parser.add_argument('--maxlen', default=128, type=int)                        # Max interaction sequence length
parser.add_argument('--hidden_units', default=64, type=int)                   # Item embedding dimension (d=64)
parser.add_argument('--num_blocks', default=2, type=int)                      # Number of self-attention blocks
parser.add_argument('--num_epochs', default=200, type=int)                    # Training epochs
parser.add_argument('--num_heads', default=1, type=int)                       # Attention heads
parser.add_argument('--dropout_rate', default=0.1, type=float)                # Dropout rate
parser.add_argument('--l2_emb', default=0.0, type=float)                      # L2 regularization on item embeddings
parser.add_argument('--device', default='0', type=str, help='cpu, hpu, gpu -> num')
parser.add_argument('--inference_only', default=False, action='store_true')   # Skip training, run evaluation only
parser.add_argument('--nn_parameter', default=False, action='store_true')     # Use nn.Parameter (for Gaudi HPU)
parser.add_argument('--state_dict_path', default=None, type=str)              # Path to load pre-trained checkpoint

args = parser.parse_args()

if __name__ == '__main__':

    # Detect hardware platform
    if args.device == 'hpu':
        args.is_hpu = True
    else:
        args.is_hpu = False

    # ===== Step 1: Download and preprocess dataset if needed =====
    # Checks if train/valid/test split files exist; if not, downloads from HuggingFace
    # and runs the preprocessing pipeline (5-core filtering, ID mapping, metadata extraction)
    if (not os.path.isfile(f'./../data_{args.dataset}/{args.dataset}_train.txt')) or (not os.path.isfile(f'./../data_{args.dataset}/{args.dataset}_valid.txt') or (not os.path.isfile(f'./../data_{args.dataset}/{args.dataset}_test.txt'))):
        print("Download Dataset")
        if not os.path.exists(f'./../data_{args.dataset}'):
            os.makedirs(f'./../data_{args.dataset}')
        preprocess_raw_5core(args.dataset)

    # ===== Step 2: Load the partitioned dataset =====
    dataset = data_partition(args.dataset, args)
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size

    # Print dataset statistics
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    # ===== Setup device =====
    if args.device == 'hpu':
        # Intel Gaudi HPU setup
        import habana_frameworks.torch.core as htcore
        args.device = torch.device('hpu')
        # nn.Embedding has compatibility issues on Gaudi, use nn.Parameter instead
        args.nn_parameter = True
    elif args.device != 'hpu' and args.device != 'cpu':
        args.device = 'cuda:' + str(args.device)

    # ===== Step 3: Initialize model and data sampler =====
    # WarpSampler: multi-process sampler that generates (user, seq, pos, neg) batches
    # in background processes for efficient training
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    # Initialize SASRec model
    model = SASRec(usernum, itemnum, args).to(args.device)

    # Xavier initialization for all model parameters
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # Skip parameters that can't be Xavier-initialized (e.g., 1D biases)

    # ===== Optional: Resume from checkpoint =====
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path)
            kwargs['args'].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            # Extract epoch number from filename for resuming training
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()

    # ===== Inference-only mode: evaluate and exit =====
    if args.inference_only:
        print('Evaluate')

        # Evaluate on test set with NDCG@10 and HR@10
        t_test = evaluate(model, dataset, args, ranking=10)
        print('')
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

        # Also evaluate with NDCG@20 and HR@20
        t_test = evaluate(model, dataset, args, ranking=20)
        print('')
        print('test (NDCG@20: %.4f, HR@20: %.4f)' % (t_test[0], t_test[1]))

        sys.exit("Terminating Inference")

    # ===== Step 4: Training loop =====
    # Loss: Binary Cross-Entropy (BPR-style)
    # - Positive items should score close to 1
    # - Negative items should score close to 0
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    time_list = []
    loss_list = []
    T = 0.0
    t0 = time.time()
    start_time = time.time()

    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        model.train()
        epoch_s_time = time.time()
        total_loss, count = 0, 0
        if args.inference_only: break

        for step in range(num_batch):
            # Get a batch from the background sampler
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            # Forward pass: compute logits for positive and negative items
            pos_logits, neg_logits = model(u, seq, pos, neg)
            # Target: positive items → 1, negative items → 0
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()

            # Only compute loss at non-padding positions (where pos != 0)
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # Optional L2 regularization on item embeddings
            if args.nn_parameter:
                loss += args.l2_emb * torch.norm(model.item_emb)
            else:
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)

            # Backward pass
            loss.backward()
            if args.is_hpu:
                htcore.mark_step()  # Required sync for Gaudi HPU
            adam_optimizer.step()
            if args.is_hpu:
                htcore.mark_step()

            total_loss += loss.item()
            count += 1

            if step % 100 == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        epoch_e_time = time.time()
        time_list.append(epoch_e_time - epoch_s_time)
        loss_list.append(total_loss / count)

        # ===== Step 5: Save model checkpoint at the last epoch =====
        # Saves [model.kwargs, model.state_dict()] as a .pth file
        # This checkpoint will be loaded by LLM-SRec as the frozen CF-SRec backbone
        if epoch == args.num_epochs:
            folder = args.dataset
            fname = 'SASRec_saving.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            if not os.path.exists(os.path.join(folder, fname)):
                try:
                    os.makedirs(os.path.join(folder))
                except:
                    print()
            torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname))

    # Clean up the background sampler processes
    sampler.close()
    end_time = time.time()

    # ===== Step 6: Final evaluation =====
    save_eval(model, dataset, args)

    print("Done")
    print("Time:", end_time - start_time)
