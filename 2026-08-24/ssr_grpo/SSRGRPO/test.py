import torch

from dataset import ITEMS, QUERIES, VOCAB, encode
from model import SSRGRPOModel


def main() -> None:
    model = SSRGRPOModel(len(VOCAB))
    try:
        model.load_state_dict(torch.load("ssr_grpo_toy.pt", map_location="cpu"))
    except FileNotFoundError:
        print("checkpoint not found; evaluating randomly initialized model")
    model.eval()
    item_tokens = torch.stack([encode(item) for item in ITEMS])
    with torch.no_grad():
        item_embeddings, _ = model.encode(item_tokens)
        for query in QUERIES:
            query_embedding, _ = model.encode(encode(query).unsqueeze(0))
            scores = (query_embedding @ item_embeddings.t()).squeeze(0)
            top_index = int(scores.argmax().item())
            print({"query": query, "top_item": ITEMS[top_index], "score": round(float(scores[top_index]), 4)})


if __name__ == "__main__":
    main()
