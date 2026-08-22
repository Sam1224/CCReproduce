import torch

from dataset import QA, build_iar_examples, encode
from model import TinyIARModel


def main():
    checkpoint = torch.load("iar_toy.pt", map_location="cpu")
    vocab = checkpoint["vocab"]
    inverse_vocab = {idx: token for token, idx in vocab.items()}
    model = TinyIARModel(len(vocab))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for qa in QA:
        x = torch.tensor([encode("answer: " + qa.question, vocab)])
        predicted_token = int(model(x).argmax(dim=-1).item())
        print({"question": qa.question, "predicted_first_token": inverse_vocab.get(predicted_token, "<unk>"), "reference_answer": qa.answer})


if __name__ == "__main__":
    main()
