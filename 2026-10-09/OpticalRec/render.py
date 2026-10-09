from __future__ import annotations

import argparse
import os

from data import build_prepared_data, render_product_card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="samples")
    parser.add_argument("--num_samples", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prepared = build_prepared_data()
    for product in prepared.products[: args.num_samples]:
        img = render_product_card(product)
        path = os.path.join(args.out_dir, f"item_{product.item_id}_{product.category}.png")
        img.save(path)
        print(path)


if __name__ == "__main__":
    main()
