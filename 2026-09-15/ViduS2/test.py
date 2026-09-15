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


@torch.no_grad()
def evaluate_model(model: StreamingViduS2Model, loader: DataLoader, device: torch.device) -> dict:
    codec = loader.dataset.codec
    model.eval()

    mse_list = []
    mse_no_face_list = []
    acc_bg = []
    acc_clothes = []
    acc_style = []
    acc_edited = []

    # 评测时额外给一个不包含 face identity 区域的 MSE，更贴近“编辑属性是否正确”
    h, w = codec.h, codec.w
    mask_face = torch.zeros(1, 1, h, w, dtype=torch.float32, device=device)
    mask_face[:, :, 2:6, 6:10] = 1.0
    mask_keep = 1.0 - mask_face
    denom_keep = mask_keep.sum().clamp_min(1.0)

    for batch in loader:
        avatar_code = batch["avatar_code"].to(device)
        ref_attr_ids = batch["ref_attr_ids"].to(device)
        ref_update_attr_ids = batch["ref_update_attr_ids"].to(device)
        ref_update_t = batch["ref_update_t"].to(device)
        stream_attr_ids = batch["stream_attr_ids"].to(device)
        edit_flags = batch["edit_flags"].to(device)
        target_frames = batch["target_frames"].to(device)
        target_attr_ids = batch["target_attr_ids"].to(device)

        pred = model.rollout(
            avatar_code=avatar_code,
            ref_attr_ids=ref_attr_ids,
            ref_update_attr_ids=ref_update_attr_ids,
            ref_update_t=ref_update_t,
            stream_attr_ids=stream_attr_ids,
            edit_flags=edit_flags,
        )

        mse_list.append(float(F.mse_loss(pred, target_frames).item()))
        mse_no_face = ((pred - target_frames) ** 2 * mask_keep).sum() / denom_keep / float(pred.shape[0] * pred.shape[1] * pred.shape[2])
        mse_no_face_list.append(float(mse_no_face.item()))

        pred_attr = codec.estimate_attr_ids(pred.cpu()).to(device)
        eq = (pred_attr == target_attr_ids).float()  # (B,T,3)

        acc_bg.append(float(eq[:, :, 0].mean().item()))
        acc_clothes.append(float(eq[:, :, 1].mean().item()))
        acc_style.append(float(eq[:, :, 2].mean().item()))

        edited_mask = (edit_flags > 0.5).float()
        denom = edited_mask.sum().clamp_min(1.0)
        acc_edited.append(float((eq * edited_mask).sum().item() / float(denom.item())))

    n = max(1, len(mse_list))
    return {
        "mse": sum(mse_list) / n,
        "mse_no_face": sum(mse_no_face_list) / n,
        "acc_bg": sum(acc_bg) / n,
        "acc_clothes": sum(acc_clothes) / n,
        "acc_style": sum(acc_style) / n,
        "acc_edited_only": sum(acc_edited) / n,
    }


@torch.no_grad()
def demo_single_sequence(
    base_model: StreamingViduS2Model,
    edit_model: StreamingViduS2Model,
    dataset: ToyViduS2Dataset,
    device: torch.device,
) -> None:
    codec = dataset.codec
    sample = dataset[0]
    avatar_code = sample["avatar_code"].unsqueeze(0).to(device)
    ref_attr_ids = sample["ref_attr_ids"].unsqueeze(0).to(device)
    ref_update_attr_ids = sample["ref_update_attr_ids"].unsqueeze(0).to(device)
    ref_update_t = sample["ref_update_t"].unsqueeze(0).to(device)
    stream_attr_ids = sample["stream_attr_ids"].unsqueeze(0).to(device)
    edit_flags = sample["edit_flags"].unsqueeze(0).to(device)
    target_attr_ids = sample["target_attr_ids"].unsqueeze(0).to(device)

    pred_base = base_model.rollout(
        avatar_code,
        ref_attr_ids,
        ref_update_attr_ids,
        ref_update_t,
        stream_attr_ids,
        edit_flags,
    )
    pred_edit = edit_model.rollout(
        avatar_code,
        ref_attr_ids,
        ref_update_attr_ids,
        ref_update_t,
        stream_attr_ids,
        edit_flags,
    )

    base_ids = codec.estimate_attr_ids(pred_base.cpu())[0]
    edit_ids = codec.estimate_attr_ids(pred_edit.cpu())[0]
    tgt_ids = target_attr_ids[0].cpu()

    print("--- demo: 单条序列属性随时间变化（t, target, baseline, editing） ---")
    print(f"ref_update_t={int(ref_update_t.item())}")
    for t in range(dataset.t):
        tgt = tuple(int(x) for x in tgt_ids[t].tolist())
        b = tuple(int(x) for x in base_ids[t].tolist())
        e = tuple(int(x) for x in edit_ids[t].tolist())
        mark = "<-- ref updated" if t == int(ref_update_t.item()) else ""
        print(f"t={t:02d} target={tgt} base={b} edit={e} {mark}")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    # test 数据：强制编辑 clothes/style + 允许 reference update
    ds = ToyViduS2Dataset(
        length=args.samples,
        t=args.timesteps,
        enable_edits=True,
        enable_ref_update=True,
        force_edit_clothes_style=True,
        seed=args.seed + 999,
    )
    loader = DataLoader(ds, batch_size=args.batch_size)

    base_model = StreamingViduS2Model(frame_shape=(3, ds.codec.h, ds.codec.w), allow_reference=False).to(device)
    edit_model = StreamingViduS2Model(frame_shape=(3, ds.codec.h, ds.codec.w), allow_reference=True).to(device)

    base_ckpt = Path(args.baseline)
    if base_ckpt.exists():
        base_model.load_state_dict(torch.load(base_ckpt, map_location=device)["model"])
    edit_ckpt = Path(args.editing)
    if edit_ckpt.exists():
        edit_model.load_state_dict(torch.load(edit_ckpt, map_location=device)["model"])

    base_metrics = evaluate_model(base_model, loader, device)
    edit_metrics = evaluate_model(edit_model, loader, device)

    print({"baseline": base_metrics, "editing": edit_metrics})

    if args.demo:
        demo_single_sequence(base_model, edit_model, ds, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="outputs/vidus2_baseline.pt")
    parser.add_argument("--editing", default="outputs/vidus2_editing.pt")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
