"""
RecSys Model Loader — Loads the pre-trained CF-SRec (SASRec) model.

This module loads a frozen pre-trained SASRec model that serves as the
collaborative filtering backbone in LLM-SRec. The SASRec model is trained
separately (in SeqRec/sasrec/main.py) on user interaction sequences.

In LLM-SRec, the SASRec model provides two things (both frozen, no gradients):
1. Item embeddings: SASRec's learned item embedding table, which captures
   collaborative filtering knowledge about item relationships. These are
   projected via f_I (item_emb_proj) into the LLM's input space.
2. User representations O_u: The last hidden state of SASRec given a user's
   interaction sequence. This is the distillation target — LLM-SRec trains
   the LLM to produce user representations that match O_u (via L_Distill).
"""

import contextlib
import logging
import os
import glob

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from utils import *
from SeqRec.sasrec.model import SASRec


def load_checkpoint(recsys, pre_trained):
    """
    Load a pre-trained SASRec model checkpoint from disk.

    Looks for a single .pth file in the directory:
        ./SeqRec/{recsys}/{pre_trained}/

    The .pth file contains [kwargs, state_dict] where kwargs stores the
    model configuration (user_num, item_num, hyperparameters).

    Args:
        recsys: Model type string (e.g., 'sasrec').
        pre_trained: Dataset name the model was pre-trained on (e.g., 'Movies_and_TV').

    Returns:
        kwargs: Dictionary of model constructor arguments.
        checkpoint: Model state dictionary (weights).
    """
    path = f'./SeqRec/{recsys}/{pre_trained}/'

    # Find the single .pth checkpoint file in the directory
    pth_file_path = find_filepath(path, '.pth')
    assert len(pth_file_path) == 1, 'There are more than two models in this dir. You need to remove other model files.\n'
    kwargs, checkpoint = torch.load(pth_file_path[0], map_location="cpu", weights_only=False)
    logging.info("load checkpoint from %s" % pth_file_path[0])

    return kwargs, checkpoint


class RecSys(nn.Module):
    """
    Wrapper for the frozen pre-trained SASRec model.

    Loads the pre-trained SASRec checkpoint, freezes all parameters, and
    exposes the model for:
    - Extracting item embeddings (model.item_emb)
    - Computing user representations O_u (model.forward with mode='log_only')

    Attributes:
        model: The frozen SASRec model instance.
        item_num: Total number of items in the dataset.
        user_num: Total number of users in the dataset.
        hidden_units: SASRec's embedding dimension (default: 64).
    """

    def __init__(self, recsys_model, pre_trained_data, device):
        super().__init__()
        # Load pre-trained SASRec weights
        kwargs, checkpoint = load_checkpoint(recsys_model, pre_trained_data)
        kwargs['args'].device = device
        model = SASRec(**kwargs)
        model.load_state_dict(checkpoint)

        # Freeze ALL SASRec parameters — LLM-SRec does NOT fine-tune CF-SRec
        for p in model.parameters():
            p.requires_grad = False

        self.item_num = model.item_num
        self.user_num = model.user_num

        self.model = model.to(device)
        self.hidden_units = kwargs['args'].hidden_units  # Default: 64

    def forward():
        print('forward')
