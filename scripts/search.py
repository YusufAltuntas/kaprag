"""Run one query against a chunk index and print the top hits.

Run:
  uv run python scripts/search.py "2023 yılı net kârı ne kadar?" --variant fixed --mode dense
  uv run python scripts/search.py "net kâr" --variant structure --mode fts --k 10

Dense mode embeds the query with the index's model (needs the `ml` group and a GPU,
or `--device cpu`).
"""

import argparse
import sys

from dotenv import load_dotenv

from kaprag.corpus import REPO_ROOT
from kaprag.index.store import VARIANTS, connect
from kaprag.retrieval import dense_search, fts_search, query_lexemes


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(REPO_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--mode", required=True, choices=["dense", "fts"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chars", type=int, default=200, help="text preview length")
    parser.add_argument("--model", default=None, help="embedding model (default: EMBED_MODEL)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with connect() as conn:
        if args.mode == "dense":
            from kaprag.index.embed import Embedder, embed_model_name

            embedder = Embedder(args.model or embed_model_name(), device=args.device)
            hits = dense_search(conn, args.variant, embedder.embed_query(args.query), k=args.k)
        else:
            print(f"lexemes: {' | '.join(query_lexemes(conn, args.query))}")
            hits = fts_search(conn, args.variant, args.query, k=args.k)

    print(f"{args.variant}/{args.mode}: {args.query}")
    if not hits:
        print("no hits")
    for rank, hit in enumerate(hits, start=1):
        pages = (
            f"p{hit.page_start}"
            if hit.page_start == hit.page_end
            else f"p{hit.page_start}-{hit.page_end}"
        )
        heading = hit.section_path[-1] if hit.section_path else ""
        preview = " ".join(hit.text.split())[: args.chars]
        print(f"{rank:>2}. {hit.score:.4f} {hit.chunk_id} {pages} [{hit.chunk_type}] {heading}")
        print(f"    {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
