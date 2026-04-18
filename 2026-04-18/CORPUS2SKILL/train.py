import argparse
from pathlib import Path

from corpus2skill.build import build_skill_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, help="Path to JSONL corpus")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--branching_factor", type=int, default=3)
    parser.add_argument("--max_leaf_size", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_skill_tree(
        corpus_path=Path(args.corpus),
        out_dir=out_dir,
        max_depth=args.max_depth,
        branching_factor=args.branching_factor,
        max_leaf_size=args.max_leaf_size,
    )


if __name__ == "__main__":
    main()
