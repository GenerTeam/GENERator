from transformers import Trainer
import torch
import torch.nn.functional as F


class BPTrainer(Trainer):
    """Base-pair level trainer for k-mer tokenized DNA sequences."""
    def __init__(self, processing_class=None, bp_loss_only=False, **kwargs):
        kwargs.pop("tokenizer", None)          # Avoid deprecation warning
        super().__init__(**kwargs)
        self.dna_tokenizer = processing_class
        self.bp_loss_only = bp_loss_only
        # Class-level cache: build once
        self._special_ids = None
        self._nucleotide_indices = None        # [V, k]  long
        self._nucleotide_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    # ------------------ Entry point ------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")                      # [B, S]
        logits = model(**inputs).logits                    # [B, S, V]
        shift_logits = logits[..., :-1, :].contiguous()    # [B, S, V]
        shift_labels = labels[..., 1:].contiguous()        # [B, S]
        device = shift_logits.device
        k = self.dna_tokenizer.k

        # 1. Build special ids & nucleotide indices cache once
        if self._special_ids is None:
            self._build_static_cache(model, k)

        ignore_ids = torch.tensor([self.dna_tokenizer.unk_token_id,
                                   self.dna_tokenizer.pad_token_id,
                                   -100], device=device)
        # ignore_ids = torch.tensor([-100], device=device)
        ignore_mask = torch.isin(shift_labels, ignore_ids)
        shift_labels = shift_labels.masked_fill(ignore_mask, -100)

        # 2. Masks
        valid_mask = shift_labels != -100
        special_mask = torch.isin(shift_labels, self._special_ids) & valid_mask
        regular_mask = valid_mask & (~special_mask)

        # 3. Loss computation
        if regular_mask.any():
            bp_loss = self._marginal_bp_loss(shift_logits, shift_labels, regular_mask, k, device)
        else:
            bp_loss = torch.tensor(0.0, device=device)

        if special_mask.any():
            token_loss = F.cross_entropy(
                shift_logits[special_mask],
                shift_labels[special_mask],
                ignore_index=-100,
                reduction='mean'
            )
            token_loss = token_loss / k
        else:
            token_loss = torch.tensor(0.0, device=device)

        # 4. Weighted combine
        bp_count = regular_mask.sum()
        special_count = special_mask.sum()
        total = bp_count + special_count
        if total == 0:
            total_loss = torch.tensor(0.0, device=device)
        else:
            total_loss = (bp_loss * bp_count + token_loss * special_count) / total
        
        total_loss = total_loss / self.args.gradient_accumulation_steps

        if self.bp_loss_only:
            return (bp_loss, logits) if return_outputs else bp_loss

        return (total_loss, logits) if return_outputs else total_loss

    # ------------------ Static cache (built once) ------------------
    def _build_static_cache(self, model, k):
        # If model is wrapped by DDP, get the underlying model
        if hasattr(model, 'module'):
            model = model.module

        vocab_size = model.config.vocab_size
        device = model.device
        self._special_ids = torch.tensor(
            [self.dna_tokenizer.vocab[e] for e in self.dna_tokenizer.special_tokens], dtype=torch.long, device=device
        )

        # 2. Nucleotide indices [V, k], tensorized once
        indices = torch.zeros(vocab_size, k, dtype=torch.long, device=device)
        for tid in range(vocab_size):
            tok = self.dna_tokenizer.ids_to_tokens[tid]
            if tok in self.dna_tokenizer.special_tokens:
                indices[tid] = 0
            else:
                seq = tok[:k]
                idx = [self._nucleotide_map.get(c, 0) for c in seq]
                indices[tid] = torch.tensor(idx, dtype=torch.long, device=device)
        self._nucleotide_indices = indices

    # ------------------ Marginal BP loss (no Python loops) ------------------
    def _marginal_bp_loss(self, shift_logits, shift_labels, regular_mask, k, device):
        token_probs = F.softmax(shift_logits, dim=-1)                       # [B, S, V]
        bp_loss = torch.tensor(0.0, device=device)

        for pos in range(k):
            # 1. True nucleotide indices at the current position [B, S]
            target_nt = self._nucleotide_indices[shift_labels, pos].masked_fill(~regular_mask, -100)
            # 2. Build 4-class probabilities [B, S, 4]
            marginal_probs = torch.zeros(*shift_logits.shape[:2], 4, device=device)
            # 3. Scatter-add once: sum token_probs by mask
            src_indices = self._nucleotide_indices[:, pos]          # [V]  0~3
            for nt_idx in range(4):
                mask = src_indices == nt_idx                        # [V]
                marginal_probs[..., nt_idx] = token_probs[..., mask].sum(dim=-1)

            marginal_probs = marginal_probs.clamp(min=1e-8)
            log_marginal_probs = marginal_probs.log()
            # 4. NLL loss in one call
            pos_loss = F.nll_loss(
                log_marginal_probs.view(-1, 4),
                target_nt.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            bp_loss += pos_loss

        return bp_loss / k
