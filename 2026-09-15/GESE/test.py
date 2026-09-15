from __future__ import annotations

import json
from pathlib import Path

import torch

from data import build_splits, build_world, reward_batch, static_headline
from model import Generator, Selector, pick_best_by_logprob, pick_best_by_selector


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    world,
    gen: Generator,
    selector: Selector,
    pairs: list[tuple[int, int]],
    num_candidates: int = 12,
) -> dict:
    gen.eval()
    selector.eval()

    ctr_static = []
    ctr_only_gen = []
    ctr_gese = []
    ctr_oracle = []

    batch_size = 128
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        user_id = torch.tensor([u for u, _ in chunk], dtype=torch.long, device=DEVICE)
        item_id = torch.tensor([it for _, it in chunk], dtype=torch.long, device=DEVICE)
        item_topic = world.item_topic[item_id]

        # static baseline (no personalization)
        static_titles = torch.stack([static_headline(int(t.item())) for t in item_topic]).to(DEVICE)
        static_ctr = reward_batch(world, user_id, item_id, static_titles)

        # exploration candidates
        sample_out = gen.sample(user_id, item_topic, num_candidates=num_candidates, temperature=1.0)

        # only-generator: choose most likely title under generator
        best_gen_title, _ = pick_best_by_logprob(sample_out)
        gen_ctr = reward_batch(world, user_id, item_id, best_gen_title)

        # GESE: selector exploits best predicted CTR
        best_sel_title, _ = pick_best_by_selector(selector, sample_out, user_id, item_topic)
        sel_ctr = reward_batch(world, user_id, item_id, best_sel_title)

        # oracle (upper bound inside candidate set)
        bsz, k, l = sample_out.titles.shape
        flat_titles = sample_out.titles.view(bsz * k, l)
        flat_user = user_id.repeat_interleave(k)
        flat_item = item_id.repeat_interleave(k)
        all_ctr = reward_batch(world, flat_user, flat_item, flat_titles).view(bsz, k)
        oracle_ctr = all_ctr.max(dim=1).values

        ctr_static.append(static_ctr.cpu())
        ctr_only_gen.append(gen_ctr.cpu())
        ctr_gese.append(sel_ctr.cpu())
        ctr_oracle.append(oracle_ctr.cpu())

    ctr_static = torch.cat(ctr_static).mean().item()
    ctr_only_gen = torch.cat(ctr_only_gen).mean().item()
    ctr_gese = torch.cat(ctr_gese).mean().item()
    ctr_oracle = torch.cat(ctr_oracle).mean().item()

    return {
        "avg_ctr_static": round(ctr_static, 4),
        "avg_ctr_only_generator": round(ctr_only_gen, 4),
        "avg_ctr_gese_selector": round(ctr_gese, 4),
        "avg_ctr_oracle_in_candidates": round(ctr_oracle, 4),
        "regret_only_generator": round(ctr_oracle - ctr_only_gen, 4),
        "regret_gese": round(ctr_oracle - ctr_gese, 4),
    }


def main() -> None:
    gen_ckpt = torch.load(ROOT / "generator.pt", map_location=DEVICE)
    sel_ckpt = torch.load(ROOT / "selector.pt", map_location=DEVICE)
    cfg = gen_ckpt["cfg"]

    world = build_world(n_users=cfg["n_users"], n_items=cfg["n_items"], seed=cfg["seed"])  # CPU
    world.user_topic_pref = world.user_topic_pref.to(DEVICE)
    world.user_style_pref = world.user_style_pref.to(DEVICE)
    world.item_topic = world.item_topic.to(DEVICE)

    splits = build_splits(
        seed=cfg["seed"],
        n_users=cfg["n_users"],
        n_items=cfg["n_items"],
        n_train=cfg["n_train"],
        n_test=cfg["n_test"],
    )

    gen = Generator(n_users=cfg["n_users"], n_topics=world.n_topics, hidden_dim=24, emb_dim=24).to(DEVICE)
    gen.load_state_dict(gen_ckpt["state_dict"])

    selector = Selector(n_users=cfg["n_users"], n_topics=world.n_topics, hidden_dim=48, emb_dim=24).to(DEVICE)
    selector.load_state_dict(sel_ckpt["state_dict"])

    metrics = evaluate(world, gen, selector, splits.test_pairs, num_candidates=12)
    metrics["device"] = str(DEVICE)

    out_path = ROOT / "test_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
