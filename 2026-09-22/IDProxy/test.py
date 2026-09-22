import torch
from torch.utils.data import DataLoader
from dataset import ToyIDProxyDataset
from model import IDProxyCTR, idproxy_loss


def main() -> None:
    batch = next(iter(DataLoader(ToyIDProxyDataset(size=8), batch_size=8)))
    model = IDProxyCTR()
    output = model(batch)
    loss = idproxy_loss(output, batch["label"], batch["is_cold"])
    assert output["logits"].shape == (8,)
    assert output["proxy"].shape == output["id_item"].shape
    assert torch.isfinite(loss)
    print({"logits_shape": list(output["logits"].shape), "loss": round(float(loss.detach()), 4)})


if __name__ == "__main__":
    main()
