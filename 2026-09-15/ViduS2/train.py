import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ToyViduS2Dataset
from model import StreamingViduS2Model


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(
    *,
    name: str,
    model: StreamingViduS2Model,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in loader:
            avatar_code = batch["avatar_code"].to(device)
            ref_attr_ids = batch["ref_attr_ids"].to(device)
            ref_update_attr_ids = batch["ref_update_attr_ids"].to(device)
            ref_update_t = batch["ref_update_t"].to(device)
            stream_attr_ids = batch["stream_attr_ids"].to(device)
            edit_flags = batch["edit_flags"].to(device)
            target_frames = batch["target_frames"].to(device)

            pred = model.rollout(
                avatar_code=avatar_code,
                ref_attr_ids=ref_attr_ids,
                ref_update_attr_ids=ref_update_attr_ids,
                ref_update_t=ref_update_t,
                stream_attr_ids=stream_attr_ids,
                edit_flags=edit_flags,
            )
            loss = F.mse_loss(pred, target_frames)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        mean_loss = running / max(1, len(loader))
        print(f"[{name}] epoch={epoch + 1} mse={mean_loss:.6f}")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # baseline：不包含编辑，且不允许 reference（用于对比）
    base_ds = ToyViduS2Dataset(
        length=args.samples,
        t=args.timesteps,
        enable_edits=False,
        enable_ref_update=False,
        seed=args.seed,
    )
    base_loader = DataLoader(base_ds, batch_size=args.batch_size, shuffle=True)
    base_model = StreamingViduS2Model(
        frame_shape=(3, base_ds.codec.h, base_ds.codec.w),
        allow_reference=False,
    ).to(device)

    # editing：包含 editing + reference update
    edit_ds = ToyViduS2Dataset(
        length=args.samples,
        t=args.timesteps,
        enable_edits=True,
        enable_ref_update=True,
        seed=args.seed + 123,
    )
    edit_loader = DataLoader(edit_ds, batch_size=args.batch_size, shuffle=True)
    edit_model = StreamingViduS2Model(
        frame_shape=(3, edit_ds.codec.h, edit_ds.codec.w),
        allow_reference=True,
    ).to(device)

    train_one(
        name="baseline",
        model=base_model,
        loader=base_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )
    train_one(
        name="editing",
        model=edit_model,
        loader=edit_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )

    base_ckpt = output_dir / "vidus2_baseline.pt"
    edit_ckpt = output_dir / "vidus2_editing.pt"
    torch.save({"model": base_model.state_dict()}, base_ckpt)
    torch.save({"model": edit_model.state_dict()}, edit_ckpt)
    print(f"saved: {base_ckpt}")
    print(f"saved: {edit_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
