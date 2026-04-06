"""
Training loop for LLM-SRec.

This module implements the main training procedure described in Section 3 of the paper.
It trains ONLY the lightweight MLP projection layers and special token embeddings while
keeping both the LLM backbone and the pre-trained SASRec (CF-SRec) model frozen.

The training objective (Equation 4 in the paper) is:
    L = L_Retrieval + L_Distill + L_Uniform

Where:
    - L_Retrieval (Eq. 1): Next Item Retrieval loss — trains the model to rank the
      correct next item higher than negative candidates using user/item representations.
    - L_Distill (Eq. 2): Distillation loss — MSE between the user representation from
      the frozen SASRec and the user representation from the LLM, transferring sequential
      knowledge from CF-SRec into the LLM.
    - L_Uniform (Eq. 3): Uniformity loss — prevents over-smoothing by encouraging user
      representations to be uniformly distributed on the hypersphere.

Evaluation uses the leave-last-out protocol:
    - Last item in sequence → test set
    - Second-to-last item → validation set
    - Remaining items → training set
Metrics: NDCG@10, HR@10, NDCG@20, HR@20
"""

import os
import torch
import random
import time
import os
import sys

from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LambdaLR

from models.seqllm_model import *
from SeqRec.sasrec.utils import data_partition, SeqDataset, SeqDataset_Inference, SeqDataset_Validation


def setup_ddp(rank, world_size, args):
    """
    Initialize Distributed Data Parallel (DDP) for multi-GPU training.
    Sets up the process group using NCCL (NVIDIA GPUs) or HCCL (Intel Gaudi HPUs).
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["ID"] = str(rank)
    if args.device.type == 'hpu':
        import habana_frameworks.torch.distributed.hccl
        init_process_group(backend="hccl", rank=rank, world_size=world_size)
    else:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def train_model(args):
    """Entry point: spawns multiple processes for DDP or runs single-GPU training."""
    print('LLMRec strat train\n')
    if args.multi_gpu:
        world_size = args.world_size
        mp.spawn(train_model_,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
    else:
        train_model_(0, 0, args)


def inference(args):
    """Entry point for inference: spawns multiple processes for DDP or runs single-GPU."""
    print('LLMRec start inference\n')
    if args.multi_gpu:
        world_size = args.world_size
        mp.spawn(inference_,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
    else:
        inference_(0, 0, args)


def train_model_(rank, world_size, args):
    """
    Core training function for LLM-SRec (runs on a single GPU / process).

    Training flow:
    1. Load the pre-trained SASRec model (frozen) and the LLM backbone (frozen).
    2. Initialize trainable MLP layers: f_I (item_emb_proj), f_user (pred_user),
       f_item (pred_item), f_CF-user (pred_user_CF2), and learnable token embeddings
       [UserOut] (CLS) and [ItemOut] (CLS_item).
    3. For each batch:
       a. Get SASRec's user representation O_u = CF-SRec(S_u) via log_only mode (frozen).
       b. Construct text prompts with interaction history (item titles + timestamps + [HistoryEmb] tokens).
       c. Construct candidate item prompts (1 positive + 3 negatives) with [ItemOut] tokens.
       d. Feed prompts through frozen LLM, extract hidden states at [UserOut] and [ItemOut] positions.
       e. Project through MLPs to get user/item representations in the shared 128-d space.
       f. Compute total loss = L_Retrieval + L_Distill + L_Uniform and backpropagate.
    4. Validate every ~10% of training steps using HR@10 on validation set.
    5. Early stopping with patience=5 based on validation HR@10.
    """
    if args.multi_gpu:
        setup_ddp(rank, world_size, args)
        if args.device == 'hpu':
            args.device = torch.device('hpu')
        else:
            args.device = 'cuda:' + str(rank)
    random.seed(0)

    # Initialize the full LLM-SRec model (frozen LLM + frozen SASRec + trainable MLPs)
    model = llmrec_model(args).to(args.device)

    # Load and partition the Amazon dataset into train/valid/test splits
    # Uses leave-last-out: last item → test, second-to-last → valid, rest → train
    dataset = data_partition(args.rec_pre_trained_data, args, path=f'./SeqRec/data_{args.rec_pre_trained_data}/{args.rec_pre_trained_data}')
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size

    # Print average interaction sequence length for the dataset
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    # Create training DataLoader — each sample contains (user_id, input_seq, positive_seq, negative_seq)
    train_data_set = SeqDataset(user_train, len(user_train.keys()), itemnum, args.maxlen)

    if args.multi_gpu:
        train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size, sampler=DistributedSampler(train_data_set, shuffle=True), pin_memory=True)
        valid_data_loader = DataLoader(valid_data_set, batch_size=args.batch_size_infer, sampler=DistributedSampler(valid_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, static_graph=True)
    else:
        train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size, pin_memory=True, shuffle=True)

    # Adam optimizer with lr=0.0001, only updates the trainable MLP parameters
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.stage2_lr, betas=(0.9, 0.98))
    # Learning rate decay: multiply by 0.95 each epoch
    scheduler = LambdaLR(adam_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    epoch_start_idx = 1

    T = 0.0
    best_perform = 0    # Best validation HR@10 seen so far
    early_stop = 0      # Counter for early stopping
    early_thres = 5     # Stop training if no improvement for 5 consecutive checks
    t0 = time.time()

    # Prepare test set users for evaluation (cap at 10,000 users for efficiency)
    eval_set_use = eval_set[1]  # eval_set[1] = users with test data
    if len(eval_set_use) > 10000:
        users = random.sample(list(eval_set_use), 10000)
    else:
        users = list(eval_set_use)

    # Filter out users without test data
    user_list = []
    for u in users:
        if len(user_test[u]) < 1: continue
        user_list.append(u)

    # Create test set DataLoader for evaluation during training
    # Test set: input = train + valid items, target = test item, with 99 random negative items
    inference_data_set = SeqDataset_Inference(user_train, user_valid, user_test, user_list, itemnum, args.maxlen)
    if args.multi_gpu:
        inference_data_loader = DataLoader(inference_data_set, batch_size=args.batch_size_infer, sampler=DistributedSampler(inference_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, static_graph=True)
    else:
        inference_data_loader = DataLoader(inference_data_set, batch_size=args.batch_size_infer, pin_memory=True)

    # ==================== Main Training Loop ====================
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        model.train()
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch)

        for step, data in enumerate(train_data_loader):
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()

            # Forward pass + backward pass for one batch (phase2 = LLM-SRec training)
            # Inside pre_train_phase2: constructs prompts, runs through LLM, computes
            # L_Retrieval + L_Distill + L_Uniform, and updates MLP weights
            model([u, seq, pos, neg], optimizer=adam_optimizer, batch_iter=[epoch, args.num_epochs + 1, step, num_batch], mode='phase2')

            # ---------- Mid-epoch validation (every ~10% of steps) ----------
            if step % (num_batch // 10) == 0 and step != 0:
                # Prepare validation set users (cap at 10,000)
                eval_set_use = eval_set[0]  # eval_set[0] = users with valid data
                if len(eval_set_use) > 10000:
                    users = random.sample(list(eval_set_use), 10000)
                else:
                    users = list(eval_set_use)

                user_list_valid = []
                for u in users:
                    if len(user_valid[u]) < 1: continue
                    user_list_valid.append(u)
                valid_data_set = SeqDataset_Validation(user_train, user_valid, user_list_valid, itemnum, args.maxlen)
                valid_data_loader = DataLoader(valid_data_set, batch_size=args.batch_size_infer, pin_memory=True, shuffle=True)

                # Run validation: compute NDCG@10, HR@10 on validation set
                model.eval()
                num_valid_batch = len(user_valid.keys()) // args.batch_size_infer
                model.users = 0.0
                model.NDCG = 0.0
                model.HT = 0.0
                model.NDCG_20 = 0.0
                model.HIT_20 = 0.0
                model.all_embs = None  # Reset cached item embeddings (forces recomputation)
                with torch.no_grad():
                    for _, data in enumerate(valid_data_loader):
                        print("Validation, early stop:", early_stop)
                        u, seq, pos, neg = data
                        u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
                        # generate_batch mode: computes all item embeddings, then ranks candidates
                        model([u, seq, pos, neg, rank, None, 'original'], mode='generate_batch')

                # Check if validation HR@10 improved
                perform = model.HT / model.users

                if perform >= best_perform:
                    best_perform = perform
                    # Save best model checkpoint
                    if rank == 0:
                        if args.multi_gpu: model.module.save_model(args, epoch2=epoch, best=True)
                        else: model.save_model(args, epoch2=epoch, best=True)

                    # Run test set evaluation on the new best model
                    model.users = 0.0
                    model.NDCG = 0.0
                    model.HT = 0.0
                    model.NDCG_20 = 0.0
                    model.HIT_20 = 0.0
                    with torch.no_grad():
                        for _, data in enumerate(inference_data_loader):
                            print("Testing")
                            u, seq, pos, neg = data
                            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
                            model([u, seq, pos, neg, rank, None, 'original'], mode='generate_batch')

                    # Write test results (NDCG@10, HR@10, NDCG@20, HR@20) to file
                    out_dir = f'./models/{args.save_dir}/'
                    out_dir = out_dir[:-1] + 'best/'
                    out_dir += f'{args.rec_pre_trained_data}_'
                    out_dir += f'{args.llm}_{epoch}_results.txt'

                    f = open(out_dir, 'a')
                    f.write(f'NDCG: {model.NDCG/model.users}, HR: {model.HT/model.users}\n')
                    f.write(f'NDCG20: {model.NDCG_20/model.users}, HR20: {model.HIT_20/model.users}\n')
                    f.close()

                    early_stop = 0
                else:
                    # No improvement — save regular checkpoint and increment early stop counter
                    model.save_model(args, epoch2=epoch)
                    early_stop += 1
                if early_stop == early_thres:
                    sys.exit("Terminating Train")
                model.train()
                scheduler.step()

        # ---------- End-of-epoch validation (same logic as mid-epoch) ----------
        if rank == 0:
            model.eval()
            num_valid_batch = len(user_valid.keys()) // args.batch_size_infer
            model.users = 0.0
            model.NDCG = 0.0
            model.HT = 0.0
            model.NDCG_20 = 0.0
            model.HIT_20 = 0.0
            model.all_embs = None

            eval_set_use = eval_set[0]
            if len(eval_set_use) > 10000:
                users = random.sample(list(eval_set_use), 10000)
            else:
                users = list(eval_set_use)

            user_list_valid = []
            for u in users:
                if len(user_valid[u]) < 1: continue
                user_list_valid.append(u)
            valid_data_set = SeqDataset_Validation(user_train, user_valid, user_list_valid, itemnum, args.maxlen)
            valid_data_loader = DataLoader(valid_data_set, batch_size=args.batch_size_infer, pin_memory=True, shuffle=True)

            with torch.no_grad():
                for _, data in enumerate(valid_data_loader):
                    print("Validation, early stop:", early_stop)
                    u, seq, pos, neg = data
                    u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
                    model([u, seq, pos, neg, rank, None, 'original'], mode='generate_batch')

            if perform >= best_perform:
                best_perform = perform
                if rank == 0:
                    if args.multi_gpu: model.module.save_model(args, epoch2=epoch, best=True)
                    else: model.save_model(args, epoch2=epoch, best=True)

                model.users = 0.0
                model.NDCG = 0.0
                model.HT = 0.0
                model.NDCG_20 = 0.0
                model.HIT_20 = 0.0
                with torch.no_grad():
                    for _, data in enumerate(inference_data_loader):
                        print("Testing")
                        u, seq, pos, neg = data
                        u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
                        model([u, seq, pos, neg, rank, None, 'original'], mode='generate_batch')
                out_dir = f'./models/{args.save_dir}/'
                out_dir = out_dir[:-1] + 'best/'
                out_dir += f'{args.rec_pre_trained_data}_'
                out_dir += f'{args.llm}_{epoch}_results.txt'

                f = open(out_dir, 'a')
                f.write(f'NDCG: {model.NDCG/model.users}, HR: {model.HT/model.users}\n')
                f.write(f'NDCG20: {model.NDCG_20/model.users}, HR20: {model.HIT_20/model.users}\n')
                f.close()

                early_stop = 0
            else:
                model.save_model(args, epoch2=epoch)
                early_stop += 1
            if early_stop == early_thres:
                sys.exit("Terminating Train")
            model.train()
            scheduler.step()

    print('train time :', time.time() - t0)
    if args.multi_gpu:
        destroy_process_group()
    return
