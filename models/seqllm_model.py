"""
LLM-SRec Model — Top-level module (seqllm_model.py)

This is the main orchestrator that connects:
  1. A frozen pre-trained CF-SRec model (SASRec) — provides sequential user representations O_u
  2. A frozen LLM backbone (LLaMA) — processes text prompts with item history
  3. Trainable MLP projection layers — bridges CF-SRec and LLM representation spaces

Architecture overview (see Figure 2 in the paper):
  ┌─────────────┐
  │   SASRec     │  Frozen CF-SRec: produces user representation O_u ∈ R^d
  │  (CF-SRec)   │  from the item interaction sequence S_u
  └──────┬───────┘
         │ O_u
         ▼
  ┌─────────────┐
  │ f_CF-user    │  MLP: projects O_u from SASRec space (d=64) to shared space (d'=128)
  │(pred_user_CF2)│  Used as the distillation target
  └──────┬───────┘
         │
  ┌──────┴───────┐
  │              │
  │    L_Distill  │  MSE loss between f_CF-user(O_u) and f_user(h^u_U)
  │              │
  └──────┬───────┘
         │
  ┌─────────────┐     ┌─────────────┐
  │  f_user      │     │  f_item      │
  │ (pred_user)  │     │ (pred_item)  │
  │ LLM hidden → │     │ LLM hidden → │
  │   d' = 128   │     │   d' = 128   │
  └──────┬───────┘     └──────┬───────┘
         │                     │
  ┌──────┴─────────────────────┴──────┐
  │           Frozen LLM              │
  │    (LLaMA 3.2 3B-Instruct)       │
  │                                   │
  │  Input: text prompt with          │
  │  [HistoryEmb] replaced by         │
  │  projected SASRec item embeddings │
  │  [UserOut] / [ItemOut] = learnable │
  │  tokens whose hidden states are   │
  │  extracted as representations      │
  └──────┬───────────────────┬────────┘
         │                   │
  ┌──────┴───────┐    ┌──────┴───────┐
  │ item_emb_proj │    │ item_emb_proj │
  │   (f_I)       │    │   (f_I)       │
  │ SASRec emb →  │    │ SASRec emb →  │
  │ LLM dim       │    │ LLM dim       │
  └───────────────┘    └───────────────┘

Prompt format for users (Table 2(b) in the paper):
  "This user has made a series of purchases in the following order:
   Item No.1, Time: YYYY-MM-DD, <Title>[HistoryEmb], ...
   Based on this sequence of purchases, generate user representation token:[UserOut]"

Prompt format for items:
  "The item title and item embedding are as follows: <Title>[HistoryEmb],
   then generate item representation token:[ItemOut]"
"""

import random
import pickle

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from models.seqllm4rec import *
from sentence_transformers import SentenceTransformer
from datetime import datetime

from tqdm import trange, tqdm

try:
    import habana_frameworks.torch.core as htcore
except:
    0


class llmrec_model(nn.Module):
    """
    Top-level LLM-SRec model that orchestrates training and inference.

    Components:
    - self.recsys: Frozen pre-trained SASRec model (CF-SRec backbone)
    - self.llm: LLM backbone wrapper (llm4rec) containing the frozen LLM and trainable MLPs
    - self.item_emb_proj: f_I projection layer — maps SASRec item embeddings (d=64) to
      LLM hidden dimension (e.g., 3072 for LLaMA 3.2 3B) so they can replace [HistoryEmb]
      tokens in the LLM's input embedding space

    The model only trains: item_emb_proj, pred_user, pred_item, pred_user_CF2,
    CLS ([UserOut] embedding), CLS_item ([ItemOut] embedding)
    """

    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device

        # Load item metadata dictionary: maps item IDs to their titles, descriptions, and timestamps
        # This is used to construct the text prompts for the LLM
        with open(f'./SeqRec/data_{args.rec_pre_trained_data}/text_name_dict.json.gz', 'rb') as ft:
            self.text_name_dict = pickle.load(ft)

        # Load the frozen pre-trained SASRec model (CF-SRec)
        # This model encodes user interaction sequences into dense representations
        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)

        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units  # SASRec hidden dimension (default: 64)
        self.sbert_dim = 768  # Sentence-BERT dimension (not used in current implementation)

        # Loss functions
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

        # Cached item embeddings for efficient inference (computed once, reused for all users)
        self.all_embs = None
        self.maxlen = args.maxlen

        # Evaluation metrics accumulators
        self.NDCG = 0
        self.HIT = 0
        self.NDCG_20 = 0
        self.HIT_20 = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG = 0
        self.lan_HIT = 0
        self.num_user = 0
        self.yes = 0

        self.extract_embs_list = []  # Stores extracted user embeddings (for extract mode)

        self.bce_criterion = torch.nn.BCEWithLogitsLoss()

        # Initialize the LLM wrapper (frozen LLM + trainable MLPs + special tokens)
        self.llm = llm4rec(device=self.device, llm_model=args.llm, args=self.args)

        # f_I: Item embedding projection layer (Eq. 2 context in paper)
        # Maps SASRec item embeddings (d=64) → LLM hidden size (e.g., 3072)
        # This allows SASRec's collaborative filtering knowledge to be injected into
        # the LLM's input space by replacing [HistoryEmb] tokens with projected embeddings
        self.item_emb_proj = nn.Sequential(
            nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
            nn.LayerNorm(self.llm.llm_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
        )
        nn.init.xavier_normal_(self.item_emb_proj[0].weight)
        nn.init.xavier_normal_(self.item_emb_proj[3].weight)

        # Additional evaluation metric accumulators
        self.users = 0.0
        self.NDCG = 0.0
        self.HT = 0.0

    def save_model(self, args, epoch2=None, best=False):
        """
        Save the trainable components of LLM-SRec to disk.
        Only saves the lightweight MLPs and special token embeddings — NOT the LLM or SASRec weights.

        Saved components:
        - item_proj.pt: f_I — item embedding projection (SASRec dim → LLM dim)
        - pred_user.pt: f_user — user representation MLP (LLM hidden → d'=128)
        - pred_item.pt: f_item — item representation MLP (LLM hidden → d'=128)
        - CLS.pt: [UserOut] learnable token embedding
        - CLS_item.pt: [ItemOut] learnable token embedding
        """
        out_dir = f'./models/{args.save_dir}/'
        if best:
            out_dir = out_dir[:-1] + 'best/'

        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_'
        out_dir += f'{args.llm}_{epoch2}_'

        if args.train:
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            torch.save(self.llm.pred_user.state_dict(), out_dir + 'pred_user.pt')
            torch.save(self.llm.pred_item.state_dict(), out_dir + 'pred_item.pt')
            if not args.token:
                # Save learnable [UserOut] and [ItemOut] embeddings
                if args.nn_parameter:
                    torch.save(self.llm.CLS.data, out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.data, out_dir + 'CLS_item.pt')
                else:
                    torch.save(self.llm.CLS.state_dict(), out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.state_dict(), out_dir + 'CLS_item.pt')
            if args.token:
                # Alternative: save the fine-tuned token embedding layer
                torch.save(self.llm.llm_model.model.embed_tokens.state_dict(), out_dir + 'token.pt')

    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        """Load previously saved trainable components from disk."""
        out_dir = f'./models/{args.save_dir}/{args.rec_pre_trained_data}_'
        out_dir += f'{args.llm}_{phase2_epoch}_'

        item_emb_proj = torch.load(out_dir + 'item_proj.pt', map_location=self.device)
        self.item_emb_proj.load_state_dict(item_emb_proj)
        del item_emb_proj

        pred_user = torch.load(out_dir + 'pred_user.pt', map_location=self.device)
        self.llm.pred_user.load_state_dict(pred_user)
        del pred_user

        pred_item = torch.load(out_dir + 'pred_item.pt', map_location=self.device)
        self.llm.pred_item.load_state_dict(pred_item)
        del pred_item

        if not args.token:
            CLS = torch.load(out_dir + 'CLS.pt', map_location=self.device)
            self.llm.CLS.load_state_dict(CLS)
            del CLS

            CLS_item = torch.load(out_dir + 'CLS_item.pt', map_location=self.device)
            self.llm.CLS_item.load_state_dict(CLS_item)
            del CLS_item

        if args.token:
            token = torch.load(out_dir + 'token.pt', map_location=self.device)
            self.llm.llm_model.model.embed_tokens.load_state_dict(token)
            del token

    def find_item_text(self, item, title_flag=True, description_flag=True):
        """
        Look up text metadata (title and/or description) for a list of item IDs.
        Used to construct the text portion of LLM prompts.

        Args:
            item: List of item IDs.
            title_flag: Include item title in output.
            description_flag: Include item description in output.
        Returns:
            List of formatted text strings like '"Title, Description"'.
        """
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]

    def find_item_time(self, item, user, title_flag=True, description_flag=True):
        """
        Look up the interaction timestamp for each item-user pair.
        Converts Unix timestamp (ms) to 'YYYY-MM-DD' format for the prompt.

        The paper's prompt design includes timestamps to help the LLM understand
        the chronological ordering of interactions (see "Discussion regarding prompt design").
        """
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        l = [datetime.utcfromtimestamp(int(self.text_name_dict['time'][i][user]) / 1000) for i in item]
        return [l_.strftime('%Y-%m-%d') for l_ in l]

    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        """Look up text metadata for a single item ID (not a list)."""
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'

    def get_item_emb(self, item_ids):
        """
        Retrieve the frozen SASRec item embeddings for given item IDs.
        These embeddings capture collaborative filtering knowledge learned during
        SASRec pre-training and will be projected via f_I (item_emb_proj) into
        the LLM's input embedding space.

        Returns:
            Item embeddings of shape (num_items, d) where d=64 (SASRec hidden dim).
        """
        with torch.no_grad():
            if self.args.nn_parameter:
                item_embs = self.recsys.model.item_emb[torch.LongTensor(item_ids).to(self.device)]
            else:
                item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))

        return item_embs

    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        """
        Dispatch to the appropriate mode:
        - 'phase2': Training mode — compute loss and update weights
        - 'generate_batch': Inference/evaluation mode — rank candidate items
        - 'extract': Extract user embeddings for analysis
        """
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode == 'generate_batch':
            self.generate_batch(data)
            print(self.args.save_dir, self.args.rec_pre_trained_data)
            print('test (NDCG@10: %.4f, HR@10: %.4f), Num User: %.4f'
                    % (self.NDCG / self.users, self.HT / self.users, self.users))
            print('test (NDCG@20: %.4f, HR@20: %.4f), Num User: %.4f'
                    % (self.NDCG_20 / self.users, self.HIT_20 / self.users, self.users))
        if mode == 'extract':
            self.extract_emb(data)

    def make_interact_text(self, interact_ids, interact_max_num, user):
        """
        Construct the interaction history portion of the user prompt.

        For each interacted item, creates a string like:
            "Item No.1, Time: 2023-05-14, "Item Title"[HistoryEmb]"

        The [HistoryEmb] token will later be replaced with the projected SASRec
        item embedding (via item_emb_proj / f_I), injecting collaborative filtering
        knowledge into the LLM's representation.

        Args:
            interact_ids: Array of item IDs the user has interacted with.
            interact_max_num: Max number of recent interactions to include (10 during training).
            user: User ID (needed to look up per-user timestamps).

        Returns:
            interact_text: Comma-joined string of all interaction entries.
            interact_ids: The (possibly truncated) list of item IDs used.
        """
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        times = self.find_item_time(interact_ids, user)
        interact_text = []
        count = 1

        if interact_max_num == 'all':
            times = self.find_item_time(interact_ids, user)
        else:
            # Only use the most recent `interact_max_num` interactions
            times = self.find_item_time(interact_ids[-interact_max_num:], user)

        if interact_max_num == 'all':
            for title in interact_item_titles_:
                interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
                count += 1
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
                count += 1
            interact_ids = interact_ids[-interact_max_num:]

        interact_text = ','.join(interact_text)
        return interact_text, interact_ids

    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set=None, task='ItemTask'):
        """
        Construct candidate item prompts for training (Next Item Retrieval approach).

        Creates text prompts for `candidate_num` items:
        - 1 positive item (the actual next item the user interacted with)
        - (candidate_num - 1) negative items (randomly sampled non-interacted items)

        Each candidate prompt follows Table 2(b) format:
            "The item title and item embedding are as follows: "Title"[HistoryEmb],
             then generate item representation token:[ItemOut]"

        The [ItemOut] token's hidden state will be extracted as the item's representation h^i_I.

        Args:
            interact_ids: User's interaction history (to avoid sampling as negatives).
            candidate_num: Total number of candidates (1 positive + N-1 negatives).
            target_item_id: The positive (ground truth) next item ID.
            target_item_title: Title text of the positive item.
            candi_set: Optional pre-defined candidate set.

        Returns:
            candidate_text: List of prompt strings for each candidate.
            candidate_ids: List of item IDs (positive first, then negatives).
        """
        neg_item_id = []
        if candi_set == None:
            # Sample 99 random negative items not in the user's history
            neg_item_id = []
            while len(neg_item_id) < 99:
                t = np.random.randint(1, self.item_num + 1)
                if not (t in interact_ids or t in neg_item_id):
                    neg_item_id.append(t)
        else:
            # Use the provided candidate set, excluding user's history
            his = set(interact_ids)
            items = list(candi_set.difference(his))
            if len(items) > 99:
                neg_item_id = random.sample(items, 99)
            else:
                while len(neg_item_id) < 49:
                    t = np.random.randint(1, self.item_num + 1)
                    if not (t in interact_ids or t in neg_item_id):
                        neg_item_id.append(t)
        random.shuffle(neg_item_id)

        # Positive item is always first in the list
        candidate_ids = [target_item_id]
        candidate_text = [f'The item title and item embedding are as follows: ' + target_item_title + "[HistoryEmb], then generate item representation token:[ItemOut]"]

        # Add negative candidate prompts
        for neg_candidate in neg_item_id[:candidate_num - 1]:
            candidate_text.append(f'The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut]")
            candidate_ids.append(neg_candidate)

        return candidate_text, candidate_ids

    def make_candidate(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set=None, task='ItemTask'):
        """
        Generate candidate item IDs for inference/evaluation (no text prompts needed).

        During inference, item embeddings are pre-computed for ALL items once, so we only
        need the candidate IDs to index into the cached embeddings (self.all_embs).

        Returns:
            candidate_ids: List of [positive_id] + [99 negative_ids].
        """
        neg_item_id = []
        neg_item_id = []
        while len(neg_item_id) < 99:
            t = np.random.randint(1, self.item_num + 1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)

        random.shuffle(neg_item_id)

        candidate_ids = [target_item_id]
        candidate_ids = candidate_ids + neg_item_id[:candidate_num - 1]

        return candidate_ids

    def pre_train_phase2(self, data, optimizer, batch_iter):
        """
        Training step for LLM-SRec (Phase 2).

        This implements the core training procedure described in Section 3 of the paper.
        For each batch of users:

        1. Get SASRec user representation O_u (frozen, via log_only mode):
           O_u = CF-SRec(S_u) — the last hidden state from SASRec given the interaction sequence.

        2. Construct user prompts with interaction history:
           "This user has made a series of purchases... [HistoryEmb]... generate user
            representation token:[UserOut]"

        3. Construct candidate item prompts (1 pos + 3 neg per user):
           "The item title and item embedding are as follows: ... [HistoryEmb]...
            generate item representation token:[ItemOut]"

        4. Project SASRec item embeddings via f_I (item_emb_proj) to replace [HistoryEmb] tokens.

        5. Feed everything into the LLM (llm4rec.train_mode0) which:
           a. Runs the LLM forward pass (frozen weights)
           b. Extracts hidden states at [UserOut] → h^u_U and [ItemOut] → h^i_I positions
           c. Projects through f_user (pred_user) → user representation in d'=128
           d. Projects through f_item (pred_item) → item representation in d'=128
           e. Projects O_u through f_CF-user (pred_user_CF2) → CF user rep in d'=128
           f. Computes total loss = L_Retrieval + L_Distill + L_Uniform

        6. Backpropagate and update only the trainable MLP parameters.

        Args:
            data: Tuple of (user_ids, sequences, positive_items, negative_items).
            optimizer: Adam optimizer for the trainable parameters.
            batch_iter: Tuple of (epoch, total_epochs, step, total_steps) for logging.
        """
        epoch, total_epoch, step, total_step = batch_iter
        print(self.args.save_dir, self.args.rec_pre_trained_data, self.args.llm)
        optimizer.zero_grad()
        u, seq, pos, neg = data

        original_seq = seq.copy()

        mean_loss = 0

        text_input = []         # User prompt strings
        candidates_pos = []     # Candidate item prompt strings (pos + neg)
        candidates_neg = []
        interact_embs = []      # Projected SASRec embeddings for user's interaction history
        candidate_embs_pos = [] # Projected SASRec embeddings for candidate items
        candidate_embs_neg = []
        candidate_embs = []

        loss_rm_mode1 = 0
        loss_rm_mode2 = 0

        # Step 1: Get frozen SASRec user representations O_u for the batch
        # log_only mode returns the last hidden state of SASRec: O_u = CF-SRec(S_u)
        # Shape: (batch_size, d) where d = SASRec hidden dim (64)
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')

        # Step 2-4: For each user in the batch, construct prompts and get embeddings
        for i in range(len(u)):
            target_item_id = pos[i][-1]  # The ground truth next item (last position)
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)

            # Construct the user's interaction history text (last 10 items)
            # Each item entry: "Item No.X, Time: YYYY-MM-DD, "Title"[HistoryEmb]"
            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10, u[i])

            # 4 candidates: 1 positive + 3 negatives (for training efficiency)
            candidate_num = 4
            candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i] > 0], candidate_num, target_item_id, target_item_title, task='RecTask')

            # Construct the full user prompt (no explicit user representation in prompt —
            # this is LLM-SRec's design choice, see Appendix E.1)
            input_text = ''
            input_text += 'This user has made a series of purchases in the following order: '
            input_text += interact_text
            input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"

            text_input.append(input_text)
            candidates_pos += candidate_text

            # Project SASRec item embeddings via f_I to LLM hidden dimension
            # These will replace [HistoryEmb] tokens in the LLM input
            interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
            candidate_embs_pos.append(self.item_emb_proj((self.get_item_emb([candidate_ids]))).squeeze(0))

        candidate_embs = torch.cat(candidate_embs_pos)

        # Bundle everything into a samples dict for the LLM forward pass
        samples = {
            'text_input': text_input,       # User prompt strings
            'log_emb': log_emb,             # SASRec user representations O_u (distillation target)
            'candidates_pos': candidates_pos,  # Candidate item prompt strings
            'interact': interact_embs,      # Projected SASRec item embeddings for [HistoryEmb] replacement
            'candidate_embs': candidate_embs,  # Projected SASRec item embeddings for candidates
        }

        # Step 5: Forward through LLM and compute total loss
        # Returns: total_loss, retrieval_loss, distill+uniform_loss
        loss, rec_loss, match_loss = self.llm(samples, mode=0)

        print("LLMRec model loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, rec_loss))
        print("LLMRec model Matching loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, match_loss))

        # Step 6: Backpropagate — gradients only flow through trainable MLPs
        loss.backward()
        if self.args.nn_parameter:
            htcore.mark_step()  # Required sync step for Intel Gaudi HPU
        optimizer.step()
        if self.args.nn_parameter:
            htcore.mark_step()

    def split_into_batches(self, itemnum, m):
        """Split item IDs 1..itemnum into batches of size m for efficient embedding computation."""
        numbers = list(range(1, itemnum + 1))
        batches = [numbers[i:i + m] for i in range(0, itemnum, m)]
        return batches

    def generate_batch(self, data):
        """
        Inference / evaluation function using the Next Item Retrieval approach.

        Two phases:
        Phase 1 (one-time): Compute item embeddings for ALL items in the catalog.
            - For each item, construct its prompt and pass through the frozen LLM.
            - Extract the hidden state at [ItemOut] position → project via f_item → d'=128.
            - Cache all item embeddings in self.all_embs (shape: [itemnum, 128]).

        Phase 2 (per-batch): For each user in the batch:
            - Construct user prompt with interaction history.
            - Pass through frozen LLM, extract [UserOut] hidden state → f_user → d'=128.
            - Compute recommendation score: s(u,i) = f_item(h^i_I) · f_user(h^u_U)^T
            - Rank the target item among 100 candidates (1 pos + 99 neg).
            - Compute NDCG@10, HR@10, NDCG@20, HR@20.
        """
        # Phase 1: Pre-compute ALL item embeddings (only done once per evaluation)
        if self.all_embs == None:
            # Determine batch size based on model/dataset size (memory constraints)
            batch_ = 128
            if self.args.llm == 'llama':
                batch_ = 64
            if self.args.rec_pre_trained_data == 'Electronics' or self.args.rec_pre_trained_data == 'Books':
                batch_ = 64
                if self.args.llm == 'llama':
                    batch_ = 32
            batches = self.split_into_batches(self.item_num, batch_)
            self.all_embs = []
            max_input_length = 1024

            for bat in tqdm(batches):
                candidate_text = []
                candidate_ids = []
                candidate_embs = []
                for neg_candidate in bat:
                    # Construct item prompt: title + [HistoryEmb] + [ItemOut]
                    candidate_text.append('The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut]")
                    candidate_ids.append(neg_candidate)

                with torch.no_grad():
                    # Tokenize item prompts
                    candi_tokens = self.llm.llm_tokenizer(
                        candidate_text,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=max_input_length,
                    ).to(self.device)

                    # Get projected SASRec item embeddings for [HistoryEmb] replacement
                    candidate_embs.append(self.item_emb_proj((self.get_item_emb(candidate_ids))))

                    # Get LLM token embeddings and replace special tokens
                    candi_embeds = self.llm.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
                    candi_embeds = self.llm.replace_out_token_all_infer(candi_tokens, candi_embeds, token=['[ItemOut]', '[HistoryEmb]'], embs={'[HistoryEmb]': candidate_embs[0]})

                    # Forward through frozen LLM
                    with torch.amp.autocast('cuda'):
                        candi_outputs = self.llm.llm_model.forward(
                            inputs_embeds=candi_embeds,
                            output_hidden_states=True
                        )

                        # Extract hidden states at [ItemOut] positions from the last layer
                        indx = self.llm.get_embeddings(candi_tokens, '[ItemOut]')
                        item_outputs = torch.cat([candi_outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])

                        # Project to shared recommendation space (d'=128) via f_item
                        item_outputs = self.llm.pred_item(item_outputs)

                    self.all_embs.append(item_outputs)
                    del candi_outputs
                    del item_outputs
            # Concatenate all item embeddings: shape (item_num, 128)
            self.all_embs = torch.cat(self.all_embs)

        # Phase 2: Compute user representations and rank candidates
        u, seq, pos, neg, rank, candi_set, files = data
        original_seq = seq.copy()

        text_input = []
        interact_embs = []
        candidate = []
        with torch.no_grad():
            for i in range(len(u)):
                candidate_embs = []
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)

                # Construct user's interaction history text
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10, u[i])

                # Sample 100 candidate items (1 positive + 99 negatives)
                candidate_num = 100
                candidate_ids = self.make_candidate(seq[i][seq[i] > 0], candidate_num, target_item_id, target_item_title, candi_set)
                candidate.append(candidate_ids)

                # Construct user prompt (same format as training)
                input_text = ''
                input_text += 'This user has made a series of purchases in the following order: '
                input_text += interact_text
                input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"
                text_input.append(input_text)

                # Get projected SASRec item embeddings for [HistoryEmb] replacement
                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))

            max_input_length = 1024

            # Tokenize all user prompts in the batch
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

            # Get LLM token embeddings and replace special tokens
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token=['[UserOut]', '[HistoryEmb]'], embs={'[HistoryEmb]': interact_embs})

            with torch.cuda.amp.autocast():
                # Forward through frozen LLM to get user representations
                outputs = self.llm.llm_model.forward(
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True
                )

                # Extract hidden states at [UserOut] positions → h^u_U
                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                user_outputs = torch.cat([outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                # Project to shared space via f_user → (batch_size, 128)
                user_outputs = self.llm.pred_user(user_outputs)

                # Rank candidates for each user
                for i in range(len(candidate)):
                    # Look up pre-computed item embeddings for this user's candidates
                    # candidate IDs are 1-indexed, all_embs is 0-indexed → subtract 1
                    item_outputs = self.all_embs[np.array(candidate[i]) - 1]

                    # Compute recommendation scores: s(u,i) = item_emb · user_emb^T
                    logits = torch.mm(item_outputs, user_outputs[i].unsqueeze(0).T).squeeze(-1)

                    # Negate scores so higher similarity → lower rank value
                    logits = -1 * logits

                    # Get the rank of the positive item (index 0 in candidates)
                    rank = logits.argsort().argsort()[0].item()

                    # Compute NDCG@10 and HR@10
                    if rank < 10:
                        self.NDCG += 1 / np.log2(rank + 2)
                        self.HT += 1
                    # Compute NDCG@20 and HR@20
                    if rank < 20:
                        self.NDCG_20 += 1 / np.log2(rank + 2)
                        self.HIT_20 += 1
                    self.users += 1
        return self.NDCG

    def extract_emb(self, data):
        """
        Extract user embeddings from the trained model (for analysis/visualization).
        Same as generate_batch but only computes user representations without ranking.
        Stores results in self.extract_embs_list.
        """
        u, seq, pos, neg, original_seq, rank, files = data

        text_input = []
        interact_embs = []
        candidate = []
        with torch.no_grad():
            for i in range(len(u)):
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10, u[i])

                input_text = ''
                input_text += 'This user has made a series of purchases in the following order: '
                input_text += interact_text
                input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"
                text_input.append(input_text)

                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))

            max_input_length = 1024

            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token=['[UserOut]', '[HistoryEmb]'], embs={'[HistoryEmb]': interact_embs})

            with torch.cuda.amp.autocast():
                outputs = self.llm.llm_model.forward(
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True
                )

                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                user_outputs = torch.cat([outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)

                self.extract_embs_list.append(user_outputs.detach().cpu())

        return 0
