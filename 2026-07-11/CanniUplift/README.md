# CanniUplift Toy Reproduction

This folder implements a runnable PyTorch reproduction of the main CanniUplift ideas: treatment attention, platform-level global alignment (PGA), and redemption-based decomposition denoising (RDD).

Run:

```bash
python train.py --cpu
python test.py --cpu
```

The dataset is synthetic because the paper uses proprietary marketplace logs. The sample schema mirrors the paper interface: user features, seller candidates, treatment/coupon values, redemption labels, observed GMV, seller uplift, and platform-level delta GMV.
