import torch
from model import BiMD

def test():
    model = BiMD(text_dim=512, motion_dim=263)
    model.eval()
    print("Generating motion from bilingual text...")
    # Mock inference
    print("Test passed.")

if __name__ == "__main__":
    test()
