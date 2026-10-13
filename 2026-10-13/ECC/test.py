import json
from pathlib import Path

import torch

from model import Tokenizer, SuggestionSeq2Seq, build_sft_batch, exact_match_accuracy


OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"


def evaluate():
    checkpoint = torch.load(OUTPUT_DIR / "ecc_toy.pt", map_location="cpu")
    tokenizer = Tokenizer(["placeholder"])
    tokenizer.token_to_id = checkpoint["token_to_id"]
    tokenizer.id_to_token = {value: key for key, value in tokenizer.token_to_id.items()}

    model = SuggestionSeq2Seq(vocab_size=len(tokenizer.token_to_id))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    valid_examples = checkpoint["valid_examples"]
    src_ids, _ = build_sft_batch(valid_examples, tokenizer)
    predictions = model.greedy_decode(
        src_ids,
        bos_id=tokenizer.token_to_id["<bos>"],
        eos_id=tokenizer.token_to_id["<eos>"],
    )
    pred_texts = [tokenizer.decode(row.tolist()) for row in predictions]
    target_texts = [target for _, target in valid_examples]

    metrics = {
        "valid_size": len(valid_examples),
        "exact_match": exact_match_accuracy(pred_texts, target_texts),
        "examples": [
            {
                "prompt": valid_examples[index][0],
                "prediction": pred_texts[index],
                "target": target_texts[index],
            }
            for index in range(min(5, len(valid_examples)))
        ],
    }
    with open(OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    evaluate()
