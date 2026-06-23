"""
model.py — OneBar generator: a BART encoder-decoder wrapper.

Implements the "Low-Latency Unified End-to-End Generation with Abstention"
(paper Sec. 3.3). One model M_theta consumes the compressed [SEP] evidence
schema s_x and decodes either a search query or the abstention token
[REJECT], producing K=8 ordered candidates via beam search.

    p_theta(y | s_x) = prod_t p_theta(y_t | y_<t, Enc_theta(s_x))

Offline-fallback: if `facebook/bart-base` cannot be downloaded, we build a
small randomly-initialised BartForConditionalGeneration so all code stays
runnable on CPU without network. This is documented in README.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import (
    BartForConditionalGeneration,
    BartConfig,
    BartTokenizerFast,
    AutoTokenizer,
)

REJECT_TOKEN = "[REJECT]"
DEFAULT_MODEL = "facebook/bart-base"


def _tiny_config(vocab_size: int) -> BartConfig:
    """A tiny BART config for fast CPU smoke tests / offline fallback."""
    return BartConfig(
        vocab_size=vocab_size,
        d_model=128,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=256,
        decoder_ffn_dim=256,
        max_position_embeddings=256,
        forced_bos_token_id=0,
    )


def load_tokenizer(model_name: str = DEFAULT_MODEL, offline: bool = False):
    """Load tokenizer; add [REJECT] special token."""
    tok = None
    if not offline:
        try:
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception as e:  # network / hub failure
            print(f"[OneBar] tokenizer download failed ({e}); using offline GPT2 BBPE.")
    if tok is None:
        # Fallback: byte-level BPE identical family to BART; build from gpt2 if
        # cached, else a minimal whitespace tokenizer is impractical for BART,
        # so we rely on a locally-buildable BartTokenizer vocab.
        from transformers import BartTokenizerFast
        try:
            tok = BartTokenizerFast.from_pretrained(model_name)
        except Exception:
            raise RuntimeError(
                "No tokenizer available offline. Run once online to cache "
                "'facebook/bart-base', or set HF proxy env vars.")
    if REJECT_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [REJECT_TOKEN]})
    return tok


class OneBarGenerator(nn.Module):
    """BART-based OneBar generator with K=8 beam decoding."""

    def __init__(self, model_name: str = DEFAULT_MODEL, tokenizer=None,
                 tiny: bool = False, offline: bool = False, K: int = 8):
        super().__init__()
        self.K = K
        self.tokenizer = tokenizer or load_tokenizer(model_name, offline=offline)
        self.model = self._build_backbone(model_name, tiny, offline)
        # resize for the added [REJECT] token
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _build_backbone(self, model_name, tiny, offline):
        if tiny:
            print("[OneBar] building TINY randomly-initialised BART (smoke test).")
            return BartForConditionalGeneration(_tiny_config(self.tokenizer.vocab_size))
        if not offline:
            try:
                return BartForConditionalGeneration.from_pretrained(model_name)
            except Exception as e:
                print(f"[OneBar] '{model_name}' download failed ({e}); "
                      f"falling back to tiny random BART config (OFFLINE).")
        return BartForConditionalGeneration(_tiny_config(self.tokenizer.vocab_size))

    # --------------------------------------------------------------------- #
    # forward helpers
    # --------------------------------------------------------------------- #
    def forward(self, input_ids, attention_mask, labels=None,
                decoder_input_ids=None):
        """Standard BART forward. If `labels` given, HF computes CE loss."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )

    def logits_for_sequence(self, input_ids, attention_mask, decoder_input_ids):
        """Return decoder logits [B, T, V] for given (forced) decoder inputs.

        Used by losses.py to compute teacher/student logits over a fixed
        target/rollout sequence (PIOPD) or candidate sequences (list-wise DPO).
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        return out.logits

    def sequence_logprob(self, input_ids, attention_mask, target_ids):
        """Sum log p(target | s_x) over non-pad target tokens.

        Implements l_theta(y|x) = sum_t log pi_theta(y_t | x, y_<t) — the
        sequence log-prob used by the list-wise DPO objective (Eq. 6).
        target_ids may contain -100 for pad positions (ignored).
        """
        pad_id = self.tokenizer.pad_token_id
        tgt = target_ids.clone()
        tgt_in = tgt.clone()
        tgt_in[tgt_in == -100] = pad_id
        decoder_input_ids = self._shift_right(tgt_in)
        logits = self.logits_for_sequence(input_ids, attention_mask, decoder_input_ids)
        logp = torch.log_softmax(logits, dim=-1)
        mask = (tgt != -100).float()
        gather_tgt = tgt.clone()
        gather_tgt[gather_tgt == -100] = pad_id
        tok_logp = logp.gather(-1, gather_tgt.unsqueeze(-1)).squeeze(-1)
        return (tok_logp * mask).sum(dim=-1)  # [B]

    def _shift_right(self, input_ids):
        """Shift labels right to form decoder_input_ids (BART convention)."""
        cfg = self.model.config
        decoder_start = cfg.decoder_start_token_id
        if decoder_start is None:
            decoder_start = cfg.bos_token_id or self.tokenizer.bos_token_id or 0
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = decoder_start
        pad_id = self.tokenizer.pad_token_id
        shifted.masked_fill_(shifted == -100, pad_id)
        return shifted

    # --------------------------------------------------------------------- #
    # generation
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def generate_queries(self, input_ids, attention_mask, K: Optional[int] = None,
                         max_len: int = 24) -> List[List[str]]:
        """Beam search with width K -> K ordered, de-duplicated candidates.

        Mirrors Sec. 3.3 inference: beam search (width 8), then lightweight
        de-duplication. (Safety / business-rule checks are stubbed.)
        """
        K = K or self.K
        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=K,
            num_return_sequences=K,
            max_length=max_len,
            early_stopping=True,
        )
        B = input_ids.size(0)
        gen = gen.view(B, K, -1)
        results = []
        for b in range(B):
            texts = self.tokenizer.batch_decode(gen[b], skip_special_tokens=False)
            cleaned = []
            for t in texts:
                t = (t.replace("<s>", "").replace("</s>", "")
                       .replace("<pad>", "").strip())
                if t and t not in cleaned:        # lightweight de-dup
                    cleaned.append(t)
            results.append(cleaned)
        return results

    def sample_rollout(self, input_ids, attention_mask, max_len: int = 24):
        """On-policy sampling for PIOPD (Algorithm 1, line 7).

        Sample one trajectory y_roll ~ pi_theta(.|x^(S)). Returns token ids.
        """
        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            top_p=0.95,
            num_beams=1,
            max_length=max_len,
        )
        return gen  # [B, T]
