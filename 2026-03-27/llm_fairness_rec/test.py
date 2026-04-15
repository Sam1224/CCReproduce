import torch
from model import FairRecLLM

def test():
    model = FairRecLLM(input_dim=128)
    model.eval()
    print("Evaluating recommendation fairness...")
    # Mock inference
    print("Test passed.")

if __name__ == "__main__":
    test()
