"""Download NCBI gene tables and build unified gene embeddings."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .ncbi_gene_embeddings import build_embeddings, build_records


NCBI_GENE_INFO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz"
NCBI_GENE_SUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_summary.gz"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        default="/tmp/ncbi-gene-embeddings",
        help="Directory for downloaded NCBI source files",
    )
    parser.add_argument("--tax-id", type=int, default=9606, help="NCBI taxonomy ID")
    parser.add_argument(
        "--gene-info",
        help=(
            "Optional local path to NCBI gene_info.gz; if omitted, download from "
            "--gene-info-url"
        ),
    )
    parser.add_argument(
        "--gene-summary",
        help=(
            "Optional local path to NCBI gene_summary.gz; if omitted and "
            "not skipped, download from --gene-summary-url"
        ),
    )
    parser.add_argument(
        "--skip-gene-summary",
        action="store_true",
        help="Do not use gene_summary.gz; only gene_info is required",
    )
    parser.add_argument(
        "--gene-info-url",
        default=NCBI_GENE_INFO_URL,
        help="Source URL for gene_info.gz",
    )
    parser.add_argument(
        "--gene-summary-url",
        default=NCBI_GENE_SUMMARY_URL,
        help="Source URL for gene_summary.gz",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download source files even if they already exist",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace embedding model name",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output pickle path for the embedding dictionary",
    )
    parser.add_argument("--metadata", help="Optional metadata JSON output path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--no-aliases", action="store_true")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable l2-normalization of sentence vectors",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )
    return parser.parse_args()


def download_gz(url, destination, overwrite=False, retries=3):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if destination.stat().st_size == 0:
            raise RuntimeError(
                f"cached file is empty and overwrite is disabled: {destination}"
            )
        with destination.open("rb") as handle:
            magic = handle.read(2)
        if magic != b"\x1f\x8b":
            raise RuntimeError(
                f"cached file is not gzip-encoded and overwrite is disabled: {destination}"
            )
        return
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, str(destination))
            if destination.stat().st_size == 0:
                raise RuntimeError(f"downloaded empty file: {destination}")
            with destination.open("rb") as handle:
                magic = handle.read(2)
            if magic != b"\x1f\x8b":
                raise RuntimeError(
                    f"downloaded file is not gzip-encoded: {destination}"
                )
            print(f"saved {url} -> {destination}")
            return
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as error:
            last_error = error
            if attempt == retries:
                break
            wait_seconds = 2 ** (attempt - 1)
            print(f"download retry {attempt}/{retries} for {url}: {error}")
            print(f"waiting {wait_seconds}s before retry...")
            time.sleep(wait_seconds)
    raise RuntimeError(f"failed to download {url} -> {destination}") from last_error


def main():
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.gene_info:
        gene_info = Path(args.gene_info)
        if not gene_info.exists():
            raise FileNotFoundError(f"gene_info file not found: {gene_info}")
    else:
        gene_info = workdir / "gene_info.gz"
        download_gz(args.gene_info_url, gene_info, overwrite=args.force_download)

    gene_summary = None
    if not args.skip_gene_summary:
        if args.gene_summary:
            gene_summary = Path(args.gene_summary)
            if not gene_summary.exists():
                raise FileNotFoundError(f"gene_summary file not found: {gene_summary}")
        else:
            gene_summary = workdir / "gene_summary.gz"
            download_gz(args.gene_summary_url, gene_summary, overwrite=args.force_download)
    summary_path = None
    if not args.skip_gene_summary:
        summary_path = gene_summary

    records = build_records(gene_info, summary_path, args.tax_id)
    if not records:
        raise RuntimeError("NCBI records are empty; check tax_id and source files")

    embeddings, metadata = build_embeddings(
        records,
        include_aliases=not args.no_aliases,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        normalize=args.normalize,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        import pickle

        with output.open("wb") as handle:
            pickle.dump(embeddings, handle)
    if args.metadata:
        payload = {
            "pipeline": "ncbi_gene_embedding_workflow",
            "workdir": str(workdir.resolve()),
            "tax_id": args.tax_id,
            "input_gene_info": str(gene_info.resolve()),
            "input_gene_summary": str(summary_path.resolve()) if summary_path else None,
            "use_aliases": not args.no_aliases,
            "normalize": args.normalize,
            "model": args.model,
            "model_device": metadata["model_device"],
            "dimension": metadata["dimension"],
            **metadata,
        }
        Path(args.metadata).write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "vector_count": len(embeddings),
                "dimension": metadata["dimension"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
