# EcomVLMAdapt Toy Reproduction

This folder reproduces the key recipe from "Adapting Vision-Language Models for E-commerce Understanding at Scale": visual verification, e-commerce instruction tuning, and attribute-centric extraction.

Run:

```bash
python train.py
python test.py
```

The original work trains large VLM backbones on millions of verified instructions and proprietary e-commerce benchmarks. This toy version uses numerical image features plus tokenized listing context to keep the data, model, training, and evaluation pipeline runnable on CPU.
