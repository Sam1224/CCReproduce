# SynthAVE Toy Reproduction

A compact reproduction of SynthAVE's scalable synthetic labeling and LLM-arena validation loop for e-commerce attribute value extraction.

The code simulates multilingual product records, trains a small PyTorch attribute extractor, then validates noisy synthetic labels with a 21-judge arena analogue. Agreement levels are used as confidence signals for data cleaning and human-triage routing.

## Run

```bash
pip install -r requirements.txt
python train.py --epochs 3
python test.py --checkpoint checkpoints/synthave_extractor.pt
```
