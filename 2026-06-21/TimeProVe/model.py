"""TimeProVe: Propose, then Verify for efficient long-video temporal reasoning.

Faithful (toy-scale, runnable) reproduction of:
  Sinha, Reilly, Krishnan, Le, Das. "TimeProVe: Propose, then Verify for Efficient
  Long Video Temporal Reasoning in Activities of Daily Living". UNC Charlotte.
  arXiv: https://arxiv.org/abs/2606.20561

Pipeline (paper Sec 3):

  ACE (Action-based Candidate Evidence) -- the cheap component, one pass over the
  full video:
    (1) Action Detector  f_act(v) -> P in [0,1]^{T x |C|}  (Eq. 1), thresholded and
        decoded into a sparse event timeline A = {(c_i, s_i, e_i)} (Eq. 2).
    (2) Query-conditioned Proposal Generator: turns A into query-conditioned
        candidate (answer, evidence-window) hypotheses at two granularities --
        atomic windows (per event) and merged windows (Eq. 4) -- then RERANKS them
        with a local score R(w|q) = R_tmp + R_sem + R_cov - R_len (Eqs. 5-8).

  Temporal Verifier -- the expensive VLM, invoked ONLY on the few top clips:
    f_vlm(clip, q, a) -> (c in {0,1}, verified answer, evidence)  (Eq. 10).
    The pipeline walks the ranked hypotheses, calls the verifier per candidate, and
    stops on the first acceptance or when the verification budget is exhausted.
    => reasoning cost is decoupled from raw video length L.

What is a faithful neural module vs a stand-in:
  * Action Detector  -> real, trainable (toy temporal-conv stand-in for MS-Temba).
  * Reranking score  -> real, exactly the four terms of Eqs. 5-8.
  * Temporal Verifier-> small trainable scorer standing in for a large cloud VLM
    (GPT-4o / VideoLLaMA3). Real component:  f_vlm(clip, q, a) reads RGB pixels of
    the short clip and reasons; here we pool the clip's segment features and score
    them against the query + candidate-answer embeddings. Interface is identical.
  * Proposal "answer generator" (edge LLM, Gemma4-2B/Qwen2-7B) -> stand-in heuristic
    that reads the detected action label inside the window (the LLM's job is exactly
    to turn the localized action timeline into an answer string).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import (ACT2ID, ACTIONS, D, NUM_CLASSES, content_tokens, tok_id)

Hypothesis = Tuple[str, Tuple[int, int], float]  # (answer, window, score)


# ============================ ACE: Action Detector ===========================
class ActionDetector(nn.Module):
    """f_act: per-segment multi-label action probabilities (Eq. 1).

    Real paper: MS-Temba (17M params) over frozen I3D/CLIP features. Here a small
    temporal CNN over the toy segment features -- same input/output contract
    (v in R^{T x D} -> P in [0,1]^{T x |C|}).
    """

    def __init__(self, d=D, n_classes=NUM_CLASSES, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Conv1d(hidden, n_classes, kernel_size=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:   # feats [T, D]
        x = feats.transpose(0, 1).unsqueeze(0)                # [1, D, T]
        h = self.net(x)
        logits = self.head(h).squeeze(0).transpose(0, 1)      # [T, |C|]
        return logits

    def probs(self, feats):
        return torch.sigmoid(self.forward(feats))


@torch.no_grad()
def decode_timeline(P: torch.Tensor, theta: float = 0.5
                    ) -> List[Tuple[int, int, int]]:
    """Threshold P and decode maximal contiguous activations per class (Eq. 2)."""
    T = P.size(0)
    events: List[Tuple[int, int, int]] = []
    active = (P >= theta)
    for c in range(P.size(1)):
        t = 0
        while t < T:
            if active[t, c]:
                s = t
                while t < T and active[t, c]:
                    t += 1
                events.append((c, s, t - 1))
            else:
                t += 1
    events.sort(key=lambda x: x[1])
    return events


# ============== ACE: Query-conditioned Proposal Generator ====================
def _events_in_window(events, w) -> List[Tuple[int, int, int]]:
    s, e = w
    return [ev for ev in events if not (ev[2] < s or ev[1] > e)]   # overlap


def _window_tokens(events, w) -> set:
    """T(w) = union of content tokens of events overlapping window w."""
    toks: set = set()
    for c, _, _ in _events_in_window(events, w):
        toks.update(content_tokens(ACTIONS[c]))
    return toks


def r_tmp(w, L: int, tau: str) -> float:
    """Temporal compatibility (Eq. 5): soft prior matching the query's intent."""
    s, e = w[0] / L, w[1] / L
    if tau == "BEFORE":
        return 1.0 - e
    if tau == "AFTER":
        return s
    if tau == "FIRST":
        return 1.0 - s
    if tau == "LAST":
        return e
    if tau == "BETWEEN":
        return 1.0 - abs((s + e) / 2 - 0.5)
    return 1.0                                  # LAST/STATE-style: no positional bias


def r_sem(events, w, q_tokens) -> float:
    """Semantic relevance (Eq. 6): best single-action match within the window."""
    Q = set(q_tokens)
    if not Q:
        return 0.0
    best = 0
    for c, _, _ in _events_in_window(events, w):
        best = max(best, len(Q & set(content_tokens(ACTIONS[c]))))
    return best / max(len(Q), 1)


def r_cov(events, w, q_tokens) -> float:
    """Coverage (Eq. 7): how much of the query content the whole window covers."""
    Q = set(q_tokens)
    if not Q:
        return 0.0
    return len(Q & _window_tokens(events, w)) / max(len(Q), 1)


def r_len(w, L: int) -> float:
    """Length penalty (Eq. 8 term): discourage unnecessarily long clips."""
    return (w[1] - w[0] + 1) / L


def rank_score(events, w, q_tokens, tau, L) -> float:
    """R(w|q) = R_tmp + R_sem + R_cov - R_len  (Eq. 8)."""
    return (r_tmp(w, L, tau) + r_sem(events, w, q_tokens)
            + r_cov(events, w, q_tokens) - r_len(w, L))


def _dominant_action(events, w) -> Optional[int]:
    """Edge-LLM stand-in: the action label that occupies most of the window."""
    overlap = {}
    s, e = w
    for c, es, ee in _events_in_window(events, w):
        overlap[c] = overlap.get(c, 0) + (min(e, ee) - max(s, es) + 1)
    if not overlap:
        return None
    return max(overlap, key=overlap.get)


def proposal_generator(events, q_tokens, tau, L, merge_gap: int = 4
                       ) -> List[Hypothesis]:
    """Build query-conditioned (answer, window) hypotheses and rerank them.

    Atomic windows (one per detected event) + merged windows (Eq. 4): adjacent
    events whose temporal gap <= merge_gap are merged so broader context can be
    verified jointly. Each window is paired with a candidate answer (edge-LLM
    stand-in) and scored by R(w|q); the list is sorted by descending score (Eq. 9).
    """
    windows: List[Tuple[int, int]] = []
    # atomic
    for _, s, e in events:
        windows.append((s, e))
    # merged: chain events that are temporally close
    if len(events) >= 2:
        cur_s, cur_e = events[0][1], events[0][2]
        for _, s, e in events[1:]:
            if s - cur_e <= merge_gap:
                cur_e = max(cur_e, e)
            else:
                if (cur_s, cur_e) not in windows:
                    windows.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        if (cur_s, cur_e) not in windows:
            windows.append((cur_s, cur_e))

    existence = (tau == "STATE" and len(q_tokens) > 0)
    hyps: List[Hypothesis] = []
    for w in windows:
        dom = _dominant_action(events, w)
        if dom is None:
            continue
        if existence:
            # candidate "yes" only when the queried action is actually inside w
            if set(q_tokens) & _window_tokens(events, w):
                ans = "yes"
            else:
                continue
        else:
            ans = ACTIONS[dom]
        hyps.append((ans, w, rank_score(events, w, q_tokens, tau, L)))

    hyps.sort(key=lambda h: h[2], reverse=True)
    return hyps


# ========================= Temporal Verifier (VLM) ===========================
class TemporalVerifier(nn.Module):
    """f_vlm stand-in: decides whether a short clip supports a candidate answer.

    Real component (Eq. 10): an expensive cloud VLM reads the RGB clip
    V[s:e] together with (q, a) and returns (c in {0,1}, verified-answer, evidence).
    Pseudocode of the real call:
        c, a_hat, d = VLM(rgb_frames(s, e), query=q, candidate_answer=a)
    Here we pool the clip's segment features and score them jointly with the query
    and candidate-answer embeddings -- identical interface, tiny trainable model.
    """

    def __init__(self, d=D, emb_dim=32):
        super().__init__()
        self.clip_proj = nn.Linear(d, emb_dim)
        self.tok_emb = nn.Embedding(len(ACTIONS) + 3, emb_dim)  # answers + yes/no/unk
        self.scorer = nn.Sequential(
            nn.Linear(3 * emb_dim, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def _q_emb(self, q_tokens):
        if not q_tokens:
            return self.tok_emb.weight.new_zeros(self.tok_emb.embedding_dim)
        ids = torch.tensor([tok_id(t) for t in q_tokens])
        return self.tok_emb(ids).mean(0)

    def forward(self, feats, window, q_tokens, answer) -> torch.Tensor:
        s, e = window
        clip = feats[s:e + 1].mean(0)                 # pool RGB evidence clip
        cv = self.clip_proj(clip)
        qv = self._q_emb(q_tokens)
        av = self.tok_emb(torch.tensor(tok_id(answer)))
        logit = self.scorer(torch.cat([cv, qv, av], -1))
        return logit.squeeze(-1)                       # raw verification logit


# =============================== TimeProVe ===================================
class TimeProVe(nn.Module):
    """Full propose-then-verify pipeline."""

    def __init__(self, detector: ActionDetector = None,
                 verifier: TemporalVerifier = None, theta: float = 0.5):
        super().__init__()
        self.detector = detector or ActionDetector()
        self.verifier = verifier or TemporalVerifier()
        self.theta = theta

    @torch.no_grad()
    def answer(self, sample, budget: int = 3, accept: float = 0.5):
        """Run ACE then targeted verification.

        Returns dict with the final answer, the evidence window, the number of
        VLM (verifier) calls actually made, and the verified clip duration --
        the quantities the paper reports for the cost/efficiency analysis.
        """
        P = self.detector.probs(sample.feats)
        events = decode_timeline(P, self.theta)
        L = sample.L
        hyps = proposal_generator(events, sample.q_tokens, sample.tau, L)

        # existence with no positive hypothesis -> answer "no" with ZERO VLM calls
        if not hyps:
            default = "no" if sample.tau == "STATE" else (
                ACTIONS[events[0][0]] if events else "no")
            return {"answer": default, "window": None, "vlm_calls": 0,
                    "verified_dur": 0, "n_candidates": 0, "events": events}

        n_calls, verified_dur = 0, 0
        for ans, w, _ in hyps[:budget]:
            n_calls += 1
            verified_dur += (w[1] - w[0] + 1)
            prob = torch.sigmoid(self.verifier(sample.feats, w, sample.q_tokens, ans))
            if prob.item() >= accept:
                return {"answer": ans, "window": w, "vlm_calls": n_calls,
                        "verified_dur": verified_dur, "n_candidates": len(hyps),
                        "events": events}
        # budget exhausted -> fall back to the top-ranked hypothesis
        top = hyps[0]
        return {"answer": top[0], "window": top[1], "vlm_calls": n_calls,
                "verified_dur": verified_dur, "n_candidates": len(hyps),
                "events": events}
