"""
SASRec — Self-Attentive Sequential Recommendation (CF-SRec backbone).

This is the collaborative filtering sequential recommender used as the
pre-trained backbone in LLM-SRec. It is trained separately on user interaction
sequences and then frozen during LLM-SRec training.

SASRec (Kang & McAuley, 2018) uses a Transformer-based architecture:
1. Item embeddings + positional embeddings → input representation
2. Multiple self-attention blocks (causal masking) → sequence encoding
3. Final hidden states used for next-item prediction

In LLM-SRec, two components from this model are used (both frozen):
- item_emb: The learned item embedding table — projected via f_I into the LLM's
  input space to replace [HistoryEmb] tokens.
- log2feats output (via mode='log_only'): The last hidden state O_u, which is
  the user representation used as the distillation target in L_Distill.
"""

import numpy as np
import torch
import torch.nn as nn


class PointWiseFeedForward(torch.nn.Module):
    """
    Position-wise feed-forward network used within each SASRec attention block.

    Applies two 1D convolutions (equivalent to two linear layers applied to each
    position independently) with ReLU activation and dropout, plus a residual connection.

    Architecture: input → Conv1D → ReLU → Dropout → Conv1D → Dropout → + input (residual)
    """

    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # Conv1D expects (batch, channels, seq_len), so we transpose
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # Transpose back to (batch, seq_len, hidden)
        outputs += inputs  # Residual connection
        return outputs


class SASRec(torch.nn.Module):
    """
    Self-Attentive Sequential Recommendation model.

    Architecture:
    - Item embedding layer: maps item IDs → dense vectors (d=64 by default)
    - Positional embedding layer: encodes position in the interaction sequence
    - Stack of self-attention blocks, each containing:
      - Multi-head self-attention (with causal mask to prevent future information leakage)
      - Point-wise feed-forward network
      - Layer normalization
    - Final layer normalization

    Args:
        user_num: Total number of users in the dataset.
        item_num: Total number of items in the dataset.
        args: Hyperparameters including:
            - hidden_units: Embedding dimension (d=64)
            - maxlen: Maximum sequence length (128)
            - num_blocks: Number of attention blocks (2)
            - num_heads: Number of attention heads (1)
            - dropout_rate: Dropout probability (0.1)
    """

    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        # Store constructor arguments for checkpoint saving/loading
        self.kwargs = {'user_num': user_num, 'item_num': item_num, 'args': args}
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.embedding_dim = args.hidden_units  # d = 64 (default)
        self.nn_parameter = args.nn_parameter

        # Item embedding table: maps item IDs to d-dimensional vectors
        # In LLM-SRec, these embeddings are frozen and projected via f_I into the LLM's space
        if self.nn_parameter:
            # nn.Parameter version (for Gaudi HPU where nn.Embedding has issues)
            self.item_emb = nn.Parameter(torch.normal(0, 1, size=(self.item_num + 1, args.hidden_units)))
            self.pos_emb = nn.Parameter(torch.normal(0, 1, size=(args.maxlen, args.hidden_units)))
        else:
            # nn.Embedding version (default for NVIDIA GPU)
            # padding_idx=0: item ID 0 is reserved for padding (no interaction)
            self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
            self.item_emb.weight.data.normal_(0.0, 1)
            self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # Stacked self-attention blocks
        self.attention_layernorms = torch.nn.ModuleList()  # Pre-attention layer norms
        self.attention_layers = torch.nn.ModuleList()       # Multi-head self-attention layers
        self.forward_layernorms = torch.nn.ModuleList()     # Pre-FFN layer norms
        self.forward_layers = torch.nn.ModuleList()         # Position-wise feed-forward layers

        # Final layer normalization applied to the output of the last attention block
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.args = args

        # Build the stack of attention blocks (default: 2 blocks)
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        """
        Convert interaction sequences (item ID sequences) to feature representations.

        This is the core encoder of SASRec:
        1. Look up item embeddings and scale by sqrt(d)
        2. Add positional embeddings
        3. Apply dropout and mask padding positions
        4. Pass through all self-attention blocks with causal masking
        5. Apply final layer normalization

        Args:
            log_seqs: Interaction sequences of shape (batch_size, maxlen).
                      Each entry is an item ID (0 = padding/no interaction).

        Returns:
            log_feats: Sequence features of shape (batch_size, maxlen, hidden_units).
                       In LLM-SRec, log_feats[:, -1, :] is used as the user representation O_u.
        """
        # Step 1: Look up item embeddings and scale
        if self.nn_parameter:
            seqs = self.item_emb[torch.LongTensor(log_seqs).to(self.dev)]
            seqs *= self.embedding_dim ** 0.5  # Scale by sqrt(d) for attention stability
        else:
            seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
            seqs *= self.item_emb.embedding_dim ** 0.5

        # Step 2: Add positional embeddings (learnable, not sinusoidal)
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        if self.nn_parameter:
            seqs += self.pos_emb[torch.LongTensor(positions).to(self.dev)]
        else:
            seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))

        # Step 3: Dropout and zero out padding positions
        seqs = self.emb_dropout(seqs)
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)  # True where padding
        seqs *= ~timeline_mask.unsqueeze(-1)  # Zero out padding positions

        # Causal attention mask: prevents attending to future positions
        # This is crucial for sequential recommendation — the model can only
        # use past interactions to predict the next item
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # Step 4: Pass through self-attention blocks
        for i in range(len(self.attention_layers)):
            # Transpose for PyTorch's MultiheadAttention: (seq_len, batch, hidden)
            seqs = torch.transpose(seqs, 0, 1)
            # Pre-norm architecture: LayerNorm before attention
            Q = self.attention_layernorms[i](seqs)
            # Self-attention with causal mask
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
            # Residual connection
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)  # Back to (batch, seq_len, hidden)

            # Feed-forward with pre-norm and residual (inside PointWiseFeedForward)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)  # Re-zero padding positions

        # Step 5: Final layer normalization
        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode='default'):
        """
        Forward pass for SASRec.

        Modes:
        - 'log_only': Return ONLY the user representation O_u (last position hidden state).
          This is used by LLM-SRec to get the distillation target.
        - 'default': Return positive and negative logits for BPR-style training.
        - 'item': Return sequence features, positive embeddings, and negative embeddings.

        Args:
            user_ids: User IDs (not used in SASRec, but kept for API consistency).
            log_seqs: Input sequences of shape (batch_size, maxlen).
            pos_seqs: Positive (next) items for each position.
            neg_seqs: Negative (random) items for each position.
            mode: Operating mode ('default', 'log_only', 'item').

        Returns:
            Depends on mode. For 'log_only': user representation O_u of shape (batch_size, d).
        """
        # Encode the interaction sequence
        log_feats = self.log2feats(log_seqs)

        if mode == 'log_only':
            # Return only the LAST position's hidden state as user representation
            # This is O_u = CF-SRec(S_u) — the distillation target in Equation 2
            log_feats = log_feats[:, -1, :]
            return log_feats

        # Look up positive and negative item embeddings
        if self.nn_parameter:
            pos_embs = self.item_emb[torch.LongTensor(pos_seqs).to(self.dev)]
            neg_embs = self.item_emb[torch.LongTensor(neg_seqs).to(self.dev)]
        else:
            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        # Compute dot product scores: how well does the sequence predict pos/neg items
        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # (batch, maxlen)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # (batch, maxlen)

        if mode == 'item':
            # Return raw embeddings (flattened across sequence positions)
            return log_feats.reshape(-1, log_feats.shape[2]), pos_embs.reshape(-1, log_feats.shape[2]), neg_embs.reshape(-1, log_feats.shape[2])
        else:
            # Return logits for BCE loss training
            return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        """
        Predict recommendation scores for candidate items.

        Used during evaluation: computes the dot product between the user's
        sequence representation (last position) and candidate item embeddings.

        Args:
            user_ids: User IDs.
            log_seqs: Input sequences (batch_size, maxlen).
            item_indices: Candidate item IDs to score.

        Returns:
            logits: Recommendation scores for each candidate item.
        """
        log_feats = self.log2feats(log_seqs)

        # Use only the last position's features as the user representation
        final_feat = log_feats[:, -1, :]

        # Look up candidate item embeddings
        if self.nn_parameter:
            item_embs = self.item_emb[torch.LongTensor(item_indices).to(self.dev)]
        else:
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        # Score = dot product between user representation and each candidate
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
