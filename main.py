"""
Main entry point for LLM-SRec (Lost in Sequence, KDD 2025).

LLM-SRec is a method that enhances LLM-based sequential recommendation by
distilling user representations from a pre-trained collaborative filtering
sequential recommender (CF-SRec, i.e., SASRec) into a frozen LLM. Only
lightweight MLP projection layers and two special token embeddings ([UserOut],
[ItemOut]) are trained — neither the LLM nor the CF-SRec model is fine-tuned.

Paper: "Lost in Sequence: Do Large Language Models Understand Sequential
        Recommendation?" (Sein Kim et al., KDD 2025)
Code based on: https://github.com/ghdtjr/A-LLMRec
"""

import os
import sys
import argparse

from utils import *
from train_model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- Hardware / distributed training settings ---
    parser.add_argument("--multi_gpu", action='store_true')       # Enable multi-GPU distributed training
    parser.add_argument('--device', type=str, default='0')        # GPU device id, or 'hpu' for Intel Gaudi
    parser.add_argument("--world_size", type=int, default=8)      # Number of GPUs for DDP (Distributed Data Parallel)

    # --- Model selection ---
    parser.add_argument("--llm", type=str, default='llama-3b', help='flan_t5, llama, vicuna')  # Backbone LLM to use
    parser.add_argument("--recsys", type=str, default='sasrec')   # CF-SRec backbone (pre-trained SASRec model)
    parser.add_argument("--rec_pre_trained_data", type=str, default='Movies_and_TV')  # Amazon dataset name for pre-trained CF-SRec

    # --- Mode flags ---
    parser.add_argument("--train", action='store_true')           # Run training (Phase 2: train MLPs with frozen LLM + CF-SRec)
    parser.add_argument("--extract", action='store_true')         # Extract user embeddings from trained model
    parser.add_argument("--token", action='store_true')           # If set, fine-tune LLM token embeddings instead of using learnable [CLS] embeddings

    parser.add_argument("--save_dir", type=str, default='seqllm')  # Directory to save model checkpoints

    # --- Training hyperparameters ---
    parser.add_argument('--batch_size', default=20, type=int)           # Training batch size
    parser.add_argument('--batch_size_infer', default=20, type=int)     # Inference/evaluation batch size
    parser.add_argument('--infer_epoch', default=1, type=int)           # Epoch checkpoint to use for inference
    parser.add_argument('--maxlen', default=128, type=int)              # Maximum interaction sequence length (paper default)
    parser.add_argument('--num_epochs', default=10, type=int)           # Maximum training epochs (early stopping with patience=5)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)      # Learning rate for Adam optimizer (paper: 0.0001)
    parser.add_argument('--nn_parameter', default=False, action='store_true')  # Use nn.Parameter instead of nn.Embedding (needed for Gaudi HPU)


    args = parser.parse_args()

    # Set the compute device: Intel Gaudi HPU or NVIDIA CUDA GPU
    if args.device =='hpu':
        args.device = torch.device('hpu')
    else:
        args.device = 'cuda:' + str(args.device)

    # Launch training: trains the MLP projection layers (f_I, f_user, f_item, f_CF-user)
    # and special token embeddings ([UserOut], [ItemOut]) while keeping the LLM and SASRec frozen
    if args.train:
        train_model(args)
