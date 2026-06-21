"""Evaluate TimeProVe on the toy test set (CPU).

Reports the quantities the paper emphasises:
  * open-ended answer accuracy,
  * temporal grounding (IoU of the returned evidence window vs gold),
  * efficiency: average VLM (verifier) calls and processed clip duration, and the
    REDUCTIONS vs (a) an agentic baseline that verifies every candidate window and
    (b) a dense full-video baseline that exposes the VLM to the whole video --
    mirroring the paper's "-75% VLM calls, -93% cost" headline (Table 3).
"""
from __future__ import annotations

import torch

from data import ACTIONS, split
from model import ActionDetector, TemporalVerifier, TimeProVe
from train import iou


def load():
    model = TimeProVe(ActionDetector(), TemporalVerifier())
    model.load_state_dict(torch.load("timeprove.pt"))
    model.eval()
    return model


def main():
    torch.manual_seed(1)
    _, test = split()
    model = load()

    correct = 0
    ious, tp_calls, tp_dur, agent_calls, dense_dur = [], [], [], [], []
    show = []
    for s in test:
        out = model.answer(s, budget=3)
        if out["answer"] == s.gold_answer:
            correct += 1
        if s.gold_window is not None and out["window"] is not None:
            ious.append(iou(out["window"], s.gold_window))
        tp_calls.append(out["vlm_calls"])
        tp_dur.append(out["verified_dur"])
        agent_calls.append(max(out["n_candidates"], 1))   # agentic: verify all cands
        dense_dur.append(s.L)                              # dense: full video to VLM
        if len(show) < 6:
            show.append((s, out))

    n = len(test)
    acc = correct / n
    mean_iou = sum(ious) / max(len(ious), 1)
    call_red = 1 - (sum(tp_calls) / max(sum(agent_calls), 1))
    cost_red = 1 - (sum(tp_dur) / max(sum(dense_dur), 1))

    print("=== TimeProVe evaluation (toy OpenTSUBench-style) ===")
    print(f"test QA pairs              : {n}")
    print(f"answer accuracy            : {acc*100:.1f}%")
    print(f"mean grounding IoU         : {mean_iou:.3f}")
    print(f"avg VLM calls / query      : {sum(tp_calls)/n:.2f}  "
          f"(agentic baseline {sum(agent_calls)/n:.2f})")
    print(f"avg processed clip dur     : {sum(tp_dur)/n:.2f} segs  "
          f"(dense full-video {sum(dense_dur)/n:.2f} segs)")
    print(f"=> VLM-call reduction      : {call_red*100:.1f}%")
    print(f"=> inference-cost reduction: {cost_red*100:.1f}%")

    print("\n--- sample predictions ---")
    for s, out in show:
        ok = "OK" if out["answer"] == s.gold_answer else "X "
        print(f"  [{ok}] {s.question:38s} tau={s.tau:6s} "
              f"pred={out['answer']:12s} gold={s.gold_answer:12s} "
              f"win={out['window']} calls={out['vlm_calls']}")


if __name__ == "__main__":
    main()
