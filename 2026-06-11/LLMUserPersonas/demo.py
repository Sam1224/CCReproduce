import asyncio
from collections import defaultdict

import numpy as np

from data import ToyPersonaConfig, build_catalog, build_users
from kmeans import kmeans
from persona import RuleBasedPersonaGenerator, retrieve_candidates_by_persona
from service import AsyncPersonaService, PersonaRequest


def cluster_history(videos, history_ids, k: int = 6, seed: int = 7):
    embs = np.stack([videos[i]["emb"] for i in history_ids], axis=0)
    labels, _ = kmeans(embs, k=k, seed=seed)

    groups = defaultdict(list)
    for vid, lab in zip(history_ids, labels.tolist()):
        groups[int(lab)].append(int(vid))

    # sort clusters by size
    clusters = sorted(groups.values(), key=len, reverse=True)
    return clusters


async def main() -> None:
    cfg = ToyPersonaConfig()
    catalog = build_catalog(cfg, seed=7)
    users = build_users(cfg, catalog, seed=7)

    videos = catalog["videos"]
    topic_vec = catalog["topic_vec"]

    user = users[0]
    history = user["history"]

    clusters = cluster_history(videos, history, k=6, seed=7)

    gen = RuleBasedPersonaGenerator(topic_vec)
    svc = AsyncPersonaService(generator=gen, videos=videos)
    await svc.start()

    history_set = set(history)

    print(f"User={user['user_id']} main_topics={user['main_topics']}")
    print("-")

    for idx, cluster_ids in enumerate(clusters[:4], start=1):
        req = PersonaRequest(user_id=user["user_id"], cluster_video_ids=cluster_ids[:25], history_video_ids=history_set)
        resp = await svc.submit(req)

        print(f"[Cluster {idx}] size={len(cluster_ids)} examples={[videos[i]['title'] for i in cluster_ids[:3]]}")
        for p in resp.personas:
            cands = retrieve_candidates_by_persona(p.text, videos, topic_vec, top_k=8)
            print(f"  - {p.kind:7s} | {p.text:20s} | conf={p.confidence:.2f} | candidates={[videos[i]['title'] for i in cands[:3]]}")
        print("-")

    await svc.stop()


if __name__ == "__main__":
    asyncio.run(main())
