"""
LLM4Rec — LLM backbone wrapper for LLM-SRec (seqllm4rec.py)

This module wraps the frozen LLM (LLaMA) and defines:
1. The trainable MLP projection layers that map LLM hidden states to the shared
   recommendation space (d'=128).
2. Learnable special token embeddings ([UserOut], [ItemOut]) whose LLM hidden
   states serve as user/item representations.
3. The three loss functions from the paper:
   - L_Retrieval (Eq. 1): Next Item Retrieval loss
   - L_Distill (Eq. 2): Distillation loss (MSE between CF-SRec and LLM user reps)
   - L_Uniform (Eq. 3): Uniformity loss (prevents over-smoothing)

Key insight from the paper: The LLM itself is completely FROZEN. Only these lightweight
components are trained, making LLM-SRec much more efficient than baselines like TALLRec,
LLaRA, and CoLLM which fine-tune the LLM using LoRA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
from peft import (
    prepare_model_for_kbit_training,
)


class llm4rec(nn.Module):
    """
    LLM backbone wrapper for the Next Item Retrieval approach.

    Architecture:
    - Frozen LLM (LLaMA 3.2 3B-Instruct, loaded in 8-bit quantization)
    - Learnable [UserOut] embedding (CLS): placed in prompts to aggregate user info
    - Learnable [ItemOut] embedding (CLS_item): placed in prompts to aggregate item info
    - f_user (pred_user): MLP that projects [UserOut] hidden state → d'=128
    - f_item (pred_item): MLP that projects [ItemOut] hidden state → d'=128
    - f_CF-user (pred_user_CF2): MLP that projects SASRec's O_u → d'=128 (distillation target)

    The recommendation score is: s(u,i) = f_item(h^i_I) · f_user(h^u_U)^T
    """

    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
        args=None
    ):
        super().__init__()
        self.device = device
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.args = args

        # Select the LLM backbone model
        if llm_model == 'llama':
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"      # 8B parameter model
        elif llm_model == 'llama-3b':
            model_id = "meta-llama/Llama-3.2-3B-Instruct"          # 3B parameter model (paper default)
        else:
            raise Exception(f'{llm_model} is not supported')
        print()
        print("=========")

        # Load the LLM — either full precision (for Gaudi HPU) or 8-bit quantized (for GPU)
        # 8-bit quantization reduces memory usage, enabling training on a single GPU
        if self.args.nn_parameter:
            # Full precision for Gaudi HPU (nn.Embedding not supported on HPU)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, torch_dtype=torch.float16)
        else:
            # 8-bit quantized for NVIDIA GPU (saves ~50% memory)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, torch_dtype=torch.float16, load_in_8bit=True,)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

        # Add special tokens used by LLM-SRec:
        # - [PAD]: padding token for batch processing
        # - [UserRep]: placeholder for explicit user representations (not used in LLM-SRec)
        # - [HistoryEmb]: replaced at runtime with projected SASRec item embeddings (f_I output)
        # - [UserOut]: learnable token whose hidden state becomes the user representation h^u_U
        # - [ItemOut]: learnable token whose hidden state becomes the item representation h^i_I
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[UserOut]', '[ItemOut]']})
        self.llm_tokenizer.add_special_tokens({'cls_token': "[CLS]"})

        # Resize token embeddings to accommodate the new special tokens
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        # Prepare the quantized model for training (enables gradient computation for trainable layers)
        self.llm_model = prepare_model_for_kbit_training(self.llm_model)

        # Freeze ALL LLM parameters — LLM-SRec does NOT fine-tune the LLM
        # (Unlike TALLRec/LLaRA/CoLLM which use LoRA to fine-tune)
        for _, param in self.llm_model.named_parameters():
            if args.token:
                # Exception: if --token flag is set, fine-tune the token embedding layer
                if 'token' in _:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        # Initialize learnable special token embeddings
        # These are the [UserOut] and [ItemOut] tokens described in Section 2.1.2
        # Their hidden states in the last LLM layer are used as user/item representations
        if not args.token:
            if args.nn_parameter:
                # nn.Parameter version (for Gaudi HPU compatibility)
                self.CLS = nn.Parameter(torch.normal(0, 1, size=(1, self.llm_model.config.hidden_size))).to(device)
                self.CLS_item = nn.Parameter(torch.normal(0, 1, size=(1, self.llm_model.config.hidden_size))).to(device)
            else:
                # nn.Embedding version (default for NVIDIA GPU)
                # Initialized with same mean/std as the LLM's existing token embeddings
                self.CLS = nn.Embedding(1, self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(self.CLS.weight, mean=self.llm_model.model.embed_tokens.weight.mean(), std=self.llm_model.model.embed_tokens.weight.std())
                self.CLS_item = nn.Embedding(1, self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(self.CLS_item.weight, mean=self.llm_model.model.embed_tokens.weight.mean(), std=self.llm_model.model.embed_tokens.weight.std())

        # f_user: Projects [UserOut] hidden state from LLM hidden dim → d'=128
        # Used to compute recommendation scores: s(u,i) = f_item(h^i_I) · f_user(h^u_U)^T
        self.pred_user = nn.Sequential(
                nn.Linear(self.llm_model.config.hidden_size, 2048),
                nn.LayerNorm(2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 128)   # Output: d' = 128 (shared recommendation space)
            )
        nn.init.xavier_normal_(self.pred_user[0].weight)
        nn.init.xavier_normal_(self.pred_user[3].weight)

        # f_item: Projects [ItemOut] hidden state from LLM hidden dim → d'=128
        # Same architecture as f_user but for item representations
        self.pred_item = nn.Sequential(
                nn.Linear(self.llm_model.config.hidden_size, 2048),
                nn.LayerNorm(2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 128)   # Output: d' = 128
            )
        nn.init.xavier_normal_(self.pred_item[0].weight)
        nn.init.xavier_normal_(self.pred_item[3].weight)

        # f_CF-user: Projects SASRec user representation O_u (d=64) → d'=128
        # This is the distillation TARGET — the LLM's user representation should match this.
        # Implements the f_CF-user in Equation 2: L_Distill = MSE(f_CF-user(O_u), f_user(h^u_U))
        self.pred_user_CF2 = nn.Sequential(
                nn.Linear(64, 128),     # Input: SASRec hidden dim (d=64)
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 128)     # Output: d' = 128
            )
        nn.init.xavier_normal_(self.pred_user_CF2[0].weight)
        nn.init.xavier_normal_(self.pred_user_CF2[3].weight)

        # Additional CF projection layer (same architecture as pred_user_CF2)
        self.cf_to_latent2 = nn.Sequential(
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 128)
            )
        nn.init.xavier_normal_(self.cf_to_latent2[0].weight)
        nn.init.xavier_normal_(self.cf_to_latent2[3].weight)

        self.mse = nn.MSELoss()  # For L_Distill (Equation 2)

        self.max_output_txt_len = max_output_txt_len

    def info_nce_loss_batch(self, anchor, log_emb, temperature=0.07):
        """
        InfoNCE contrastive loss (not used in the final LLM-SRec, but available for
        the contrastive distillation variant discussed in Appendix E.2, Equation 10).

        For each anchor, the corresponding log_emb is the positive pair,
        and all other log_embs in the batch are negatives.
        """
        batch_size = anchor.shape[0]

        # L2 normalize for cosine similarity
        anchor = F.normalize(anchor, p=2, dim=1)
        log_emb = F.normalize(log_emb, p=2, dim=1)

        # Compute pairwise cosine similarities scaled by temperature
        similarity_matrix = torch.matmul(anchor, log_emb.T) / temperature

        # Diagonal entries are positive pairs
        mask = torch.eye(batch_size, device=anchor.device).bool()
        pos_sim = similarity_matrix[mask].view(batch_size, 1)
        neg_sim = similarity_matrix[~mask].view(batch_size, -1)

        # Cross-entropy with positive pair at index 0
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def rec_loss(self, anchor, items):
        """
        L_Retrieval: Next Item Retrieval loss (Equation 1 in the paper).

        Computes cross-entropy loss where the model must identify the correct
        next item from a set of candidates (1 positive + N-1 negatives).

        s(u, i) = f_item(h^i_I) · f_user(h^u_U)^T

        L_Retrieval = -E[log(exp(s(u, i_{n+1})) / sum_k(exp(s(u, k))))]

        Args:
            anchor: User representations from f_user, shape (batch_size, d'=128).
            items: Item representations from f_item, shape (batch_size * candidate_num, d'=128).
                   Reshaped to (batch_size, candidate_num, d') for batch matrix multiply.

        Returns:
            Cross-entropy loss (positive item is always at index 0).
        """
        # Compute dot product scores between each user and their candidates
        # items reshaped: (batch, num_candidates, d') @ (batch, d', 1) → (batch, num_candidates)
        logits = torch.bmm(items.view(anchor.shape[0], -1, anchor.shape[1]), anchor.unsqueeze(2)).squeeze(2)

        # Label 0 means the first candidate (positive item) should have the highest score
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def uniformity(self, x, p=2):
        """
        Uniformity loss component (part of L_Uniform, Equation 3 in the paper).

        Encourages representations to be uniformly distributed on the unit hypersphere.
        Computed as: E[exp(-2 * ||x_i - x_j||^2)] over all pairs (i,j).

        Lower uniformity loss = more spread out representations = less over-smoothing.

        This prevents the distillation loss from collapsing all user representations
        to similar vectors, which would lose discriminative information.
        """
        return torch.pdist(x, p=p).pow(2).mul(-p).exp().mean()

    def replace_out_token_all(self, llm_tokens, inputs_embeds, token=[], embs=None,):
        """
        Replace special token embeddings in the LLM input with their actual values.
        Used during TRAINING for batched user prompts.

        This is the key mechanism that injects collaborative filtering knowledge into the LLM:

        For each special token type:
        - [HistoryEmb]: Replaced with projected SASRec item embeddings (f_I output).
          Each [HistoryEmb] in the sequence corresponds to one interacted item.
        - [UserRep]: Replaced with explicit user representation (not used in LLM-SRec).
        - [UserOut]: Replaced with the learnable CLS embedding. The LLM's hidden state
          at this position becomes h^u_U (user representation).
        - [ItemOut]: Replaced with the learnable CLS_item embedding. The LLM's hidden state
          at this position becomes h^i_I (item representation).

        Args:
            llm_tokens: Tokenized input (contains token IDs for finding special token positions).
            inputs_embeds: The LLM's token embedding matrix for the input.
            token: List of special token strings to replace.
            embs: Dictionary mapping token strings to their replacement embeddings.

        Returns:
            Modified inputs_embeds with special tokens replaced by their actual embeddings.
        """
        for t in token:
            # Find the token ID for this special token
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []
            for inx in range(len(llm_tokens["input_ids"])):
                # Find all positions where this special token appears
                idx_tensor = (llm_tokens["input_ids"][inx] == token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]

                if 'Emb' in t:
                    # [HistoryEmb]: Replace each occurrence with the corresponding
                    # projected SASRec item embedding from the user's interaction history
                    ee = embs[t][inx]
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                elif 'Rep' in t:
                    # [UserRep]: Replace with explicit user representation
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                else:
                    # [UserOut] or [ItemOut]: Replace with learnable CLS/CLS_item embedding
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS[torch.tensor([0]).to(self.device)], user_vector[idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector[idx+1:]), dim=0)
                            elif 'ItemOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS_item[torch.tensor([0]).to(self.device)], user_vector[idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS_item(torch.tensor([0]).to(self.device)), user_vector[idx+1:]), dim=0)

                vectors.append(user_vector.unsqueeze(0))
            inputs_embeds = torch.cat(vectors)
        return inputs_embeds

    def replace_out_token_all_infer(self, llm_tokens, inputs_embeds, token=[], embs=None, user_act=False, item_act=False):
        """
        Replace special token embeddings during INFERENCE for item prompts.

        Same logic as replace_out_token_all, but handles the case where each item
        in the batch has only ONE [HistoryEmb] token (its own SASRec embedding),
        unlike user prompts which have multiple [HistoryEmb] tokens (one per interaction).

        The key difference is in the [HistoryEmb] handling:
        - Training (replace_out_token_all): embs[t][inx] is a sequence of embeddings
        - Inference (this function): embs[t][inx] is wrapped in a list [embs[t][inx]]
        """
        for t in token:
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []
            for inx in range(len(llm_tokens["input_ids"])):
                idx_tensor = (llm_tokens["input_ids"][inx] == token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]
                if 'Emb' in t:
                    # Single item embedding per item prompt (wrapped in list for zip)
                    ee = [embs[t][inx]]
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)

                elif 'Rep' in t:
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                else:
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS[torch.tensor([0]).to(self.device)], user_vector[idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector[idx+1:]), dim=0)
                            elif 'ItemOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS_item[torch.tensor([0]).to(self.device)], user_vector[idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx], self.CLS_item(torch.tensor([0]).to(self.device)), user_vector[idx+1:]), dim=0)

                vectors.append(user_vector.unsqueeze(0))
            inputs_embeds = torch.cat(vectors)
        return inputs_embeds

    def get_embeddings(self, llm_tokens, token):
        """
        Find the positions of a special token in each sequence of the batch.

        Used after the LLM forward pass to extract the hidden states at the
        [UserOut] or [ItemOut] positions. These hidden states (h^u_U and h^i_I)
        are then projected through f_user/f_item MLPs.

        Args:
            llm_tokens: Tokenized input batch.
            token: Special token string (e.g., '[UserOut]' or '[ItemOut]').

        Returns:
            List of index tensors, one per sequence in the batch.
        """
        token_idx = []
        token_id = self.llm_tokenizer(token, return_tensors="pt", add_special_tokens=False).input_ids.item()
        for inx in range(len(llm_tokens['input_ids'])):
            idx_tensor = (llm_tokens['input_ids'][inx] == token_id).nonzero().view(-1)
            token_idx.append(idx_tensor)
        return token_idx

    def forward(self, samples, mode=0):
        """Dispatch to training mode 0 (default LLM-SRec training)."""
        if mode == 0:
            return self.train_mode0(samples)
        elif mode == 1:
            return self.train_mode1(samples)

    def train_mode0(self, samples):
        """
        Main training forward pass implementing the LLM-SRec loss (Equation 4).

        L = L_Retrieval + L_Distill + L_Uniform

        Steps:
        1. Tokenize user prompts and candidate item prompts.
        2. Replace special tokens ([UserOut], [ItemOut], [HistoryEmb]) with actual embeddings.
        3. Run frozen LLM forward pass for both user and item prompts.
        4. Extract hidden states at [UserOut] and [ItemOut] positions.
        5. Project through f_user and f_item MLPs to d'=128 space.
        6. Compute L_Retrieval: cross-entropy over candidate items.
        7. Project SASRec O_u through f_CF-user to d'=128.
        8. Compute L_Distill: MSE between f_user(h^u_U) and f_CF-user(O_u).
        9. Compute L_Uniform: uniformity penalty on both LLM and CF user representations.
        10. Return total loss.

        Args:
            samples: Dictionary containing:
                - 'text_input': User prompt strings
                - 'log_emb': SASRec user representations O_u (batch_size, 64)
                - 'candidates_pos': Candidate item prompt strings
                - 'interact': Projected SASRec item embeddings for [HistoryEmb]
                - 'candidate_embs': Projected SASRec item embeddings for candidates

        Returns:
            (total_loss, retrieval_loss_value, distill+uniform_loss_value)
        """
        max_input_length = 1024
        log_emb = samples['log_emb']  # SASRec user representations O_u

        # ===== Step 1: Tokenize user prompts =====
        llm_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        # Get the LLM's default token embeddings for the input
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

        # ===== Step 2a: Replace special tokens in user prompts =====
        # [UserOut] → learnable CLS embedding
        # [HistoryEmb] → projected SASRec item embeddings (one per interacted item)
        inputs_embeds = self.replace_out_token_all(llm_tokens, inputs_embeds, token=['[UserOut]', '[HistoryEmb]'], embs={'[HistoryEmb]': samples['interact']})

        # ===== Step 1b: Tokenize candidate item prompts =====
        candi_tokens = self.llm_tokenizer(
                samples['candidates_pos'],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

        candi_embeds = self.llm_model.get_input_embeddings()(candi_tokens['input_ids'])

        # ===== Step 2b: Replace special tokens in item prompts =====
        # [ItemOut] → learnable CLS_item embedding
        # [HistoryEmb] → projected SASRec item embedding (one per candidate item)
        candi_embeds = self.replace_out_token_all_infer(candi_tokens, candi_embeds, token=['[ItemOut]', '[HistoryEmb]'], embs={'[HistoryEmb]': samples['candidate_embs']})

        with torch.amp.autocast('cuda'):
            # ===== Step 3a: Run frozen LLM forward pass for item prompts =====
            candi_outputs = self.llm_model.forward(
                inputs_embeds=candi_embeds,
                output_hidden_states=True  # Need hidden states from the last layer
            )

            # ===== Step 4a: Extract hidden states at [ItemOut] positions =====
            # For each item, get the hidden state at the [ItemOut] token position
            # from the LAST layer of the LLM → h^i_I
            indx = self.get_embeddings(candi_tokens, '[ItemOut]')
            item_outputs = torch.cat([candi_outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])

            # ===== Step 3b: Run frozen LLM forward pass for user prompts =====
            outputs = self.llm_model.forward(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True
            )

            # ===== Step 4b: Extract hidden states at [UserOut] positions =====
            # h^u_U: the LLM's representation of the user at the [UserOut] position
            indx = self.get_embeddings(llm_tokens, '[UserOut]')
            user_outputs = torch.cat([outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])

        # ===== Step 5: Project through MLPs to shared d'=128 space =====
        user_outputs = self.pred_user(user_outputs)  # f_user(h^u_U) → (batch_size, 128)
        item_outputs = self.pred_item(item_outputs)  # f_item(h^i_I) → (batch_size * candidates, 128)

        # ===== Step 6: L_Retrieval (Equation 1) =====
        # Cross-entropy loss: positive item (index 0) should score highest
        rec_loss = self.rec_loss(user_outputs, item_outputs)

        # ===== Step 7: Project SASRec O_u through f_CF-user =====
        log_emb = self.pred_user_CF2(log_emb)  # f_CF-user(O_u) → (batch_size, 128)

        # ===== Step 8: L_Distill (Equation 2) =====
        # MSE between LLM user representation and SASRec user representation
        # This is the core distillation: transferring sequential knowledge from CF-SRec to LLM
        user_outputs = F.normalize(user_outputs, p=2, dim=1)  # L2 normalize
        log_emb = F.normalize(log_emb, p=2, dim=1)            # L2 normalize
        match_loss = self.mse(user_outputs, log_emb)

        # ===== Step 9: L_Uniform (Equation 3) =====
        # Uniformity loss: prevents over-smoothing by encouraging diverse representations
        # Applied to BOTH the LLM user representations AND the CF-SRec user representations
        match_loss += (self.uniformity(user_outputs) + self.uniformity(log_emb))

        # ===== Step 10: Total loss (Equation 4) =====
        # L = L_Retrieval + L_Distill + L_Uniform
        loss = rec_loss + match_loss

        return loss, rec_loss.item(), match_loss.item()
