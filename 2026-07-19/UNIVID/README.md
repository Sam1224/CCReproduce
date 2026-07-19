# UNIVID Toy Reproduction

A compact PyTorch reproduction of the UNIVID idea: policy-aware video captions become an interpretable intermediate representation for moderation decisions, leakage recall, and trend governance.

This implementation uses toy video frame features and OCR/title tokens so it can run on CPU. The interfaces mirror a larger pipeline: `ToyVideoModerationDataset` emits multimodal signals, `UNIVIDLite` predicts policy-aware caption tokens plus violation decisions, and `UNIVIDRAG` retrieves prior violative events before final moderation.

## Run

```bash
pip install -r requirements.txt
python train.py --epochs 3
python test.py --checkpoint checkpoints/univid_lite.pt
```
