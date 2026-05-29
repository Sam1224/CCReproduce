import argparse

from src.toy_data import generate_toy_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument("--num_docs", type=int, default=2000)
    parser.add_argument("--doc_len", type=int, default=48)
    parser.add_argument("--num_queries", type=int, default=500)
    parser.add_argument("--query_len", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    generate_toy_dataset(
        args.out_dir,
        vocab_size=args.vocab_size,
        num_docs=args.num_docs,
        doc_len=args.doc_len,
        num_queries=args.num_queries,
        query_len=args.query_len,
        overlap=args.overlap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
