"""Build unified gene embeddings from NCBI descriptions and a HuggingFace model."""

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np


def split_field(value):
    """Split alias-like fields (gene symbols, synonyms, etc.)."""
    if not value or value == "-":
        return set()
    return {
        piece.strip().upper()
        for piece in re.split(r"[|;,]", value)
        if piece.strip() and piece != "-"
    }


def parse_gene_info(path, tax_id):
    """Parse `gene_info.gz` for a fixed tax id and return per-GeneID records."""
    records = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n")
        if not header.startswith("#tax_id"):
            raise ValueError(f"{path} missing gene_info header line")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 16:
                continue
            if parts[0] != str(tax_id):
                continue
            gene_id = parts[1]
            symbol = (parts[2] or "").strip()
            if not symbol or symbol == "-":
                continue
            entry = records.setdefault(
                gene_id,
                {
                    "gene_id": gene_id,
                    "symbol": symbol.upper(),
                    "synonyms": set(),
                    "description": "",
                    "type_of_gene": "",
                    "full_name": "",
                    "nomenclature_status": "",
                    "other_designations": "",
                },
            )
            for index, target_key, allow_empty in (
                (8, "description", False),
                (9, "type_of_gene", True),
                (10, "full_name", True),
                (12, "nomenclature_status", True),
                (13, "other_designations", False),
            ):
                value = (parts[index] or "").strip()
                if value == "-":
                    value = ""
                if not entry[target_key] or (allow_empty and not allow_empty):
                    entry[target_key] = value
            entry["synonyms"].update(split_field(parts[4]))
            if not entry["description"] and parts[8] not in ("", "-"):
                entry["description"] = parts[8].strip()
            if not entry["full_name"] and parts[10] not in ("", "-"):
                entry["full_name"] = parts[10].strip()
            if not entry["other_designations"] and parts[13] not in ("", "-"):
                entry["other_designations"] = parts[13].strip()
    return records


def parse_gene_summary(path, tax_id, records):
    """Attach RefSeq/other summaries to existing records keyed by GeneID."""
    if path is None:
        return
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n")
        if not header.startswith("#tax_id"):
            raise ValueError(f"{path} missing gene_summary header line")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            if parts[0] != str(tax_id):
                continue
            gene_id = parts[1]
            record = records.get(gene_id)
            if record is None:
                continue
            summary = (parts[3] or "").strip()
            if summary and summary != "-":
                record.setdefault("summary", []).append(summary)


def build_embedding_text(record):
    """Compose one semantic description line for each gene record."""
    parts = [f"Gene symbol: {record['symbol']}"]
    if record.get("full_name"):
        parts.append(f"Full name: {record['full_name']}")
    if record.get("description"):
        parts.append(f"Functional description: {record['description']}")
    aliases = sorted(
        alias for alias in record.get("synonyms", set()) if alias != record["symbol"]
    )
    if aliases:
        parts.append("Aliases: " + ", ".join(aliases[:64]))
    if record.get("type_of_gene"):
        parts.append(f"Type: {record['type_of_gene']}")
    if record.get("nomenclature_status"):
        parts.append(f"Nomenclature status: {record['nomenclature_status']}")
    if record.get("other_designations"):
        other = ", ".join(sorted(split_field(record["other_designations"])))
        if other:
            parts.append(f"Other designations: {other[:512]}")
    for summary in record.get("summary", []):
        if summary:
            parts.append(f"Summary: {summary}")
    return " ".join(parts).strip()


def build_records(gene_info, gene_summary, tax_id):
    records = parse_gene_info(gene_info, tax_id)
    parse_gene_summary(gene_summary, tax_id, records)
    return records


def select_embeddings(records, include_aliases=True):
    """Create symbol->text map and alias->canonical map."""
    texts = {}
    aliases = {}
    for record in records.values():
        symbol = record["symbol"]
        if symbol not in texts:
            texts[symbol] = build_embedding_text(record)
        if include_aliases:
            for alias in record.get("synonyms", set()):
                if alias in {"", "-"}:
                    continue
                aliases.setdefault(alias, symbol)
    return texts, aliases


def _get_torch():
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError(
            "transformers embedding requires torch; install torch/transformers first"
        ) from exc
    return torch, F


def load_embedding_model(name, trust_remote_code=True, device="auto"):
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for embedding generation; "
            "install it in the active environment"
        ) from exc
    torch, _ = _get_torch()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(name, trust_remote_code=trust_remote_code).to(device)
    return tokenizer, model, device


def mean_pool(last_hidden, attention_mask):
    """Mean-pool transformer token embeddings with attention masking."""
    torch, _ = _get_torch()
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(1)
    lengths = mask.sum(1).clamp(min=1.0)
    return summed / lengths


def encode_texts(tokenizer, model, device, texts, batch_size, max_length, normalize=True):
    torch, F = _get_torch()
    vectors = []
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.inference_mode():
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            if normalize:
                pooled = F.normalize(pooled, dim=-1)
        vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
        if device != "cpu":
            torch.cuda.empty_cache()
    return np.vstack(vectors)


def build_embeddings(
    records,
    include_aliases=True,
    model_name="Qwen/Qwen3-Embedding-0.6B",
    batch_size=32,
    max_length=256,
    trust_remote_code=True,
    device="auto",
    normalize=True,
):
    """Encode all records and return a symbol->vector dictionary."""
    texts, aliases = select_embeddings(records, include_aliases=include_aliases)
    symbols = sorted(texts)
    sentences = [texts[symbol] for symbol in symbols]
    tokenizer, model, resolved_device = load_embedding_model(
        model_name,
        trust_remote_code=trust_remote_code,
        device=device,
    )
    encoded = encode_texts(
        tokenizer,
        model,
        resolved_device,
        sentences,
        batch_size=batch_size,
        max_length=max_length,
        normalize=normalize,
    )
    embeddings = {
        symbol: vector for symbol, vector in zip(symbols, encoded)
    }
    alias_collisions = []
    for alias, canonical in aliases.items():
        if not alias:
            continue
        if alias in embeddings:
            if alias != canonical:
                alias_collisions.append((alias, canonical))
            continue
        if canonical in embeddings:
            embeddings[alias] = embeddings[canonical]
    dimension = int(encoded.shape[1]) if encoded.size else 0
    metadata = {
        "source": "NCBI",
        "model": model_name,
        "model_device": resolved_device,
        "normalization": "l2" if normalize else "none",
        "alias_projection": include_aliases,
        "records": len(records),
        "symbol_vectors": len(texts),
        "alias_vectors": len(embeddings) - len(texts),
        "alias_collisions": len(alias_collisions),
        "dimension": dimension,
    }
    if alias_collisions:
        metadata["alias_collision_examples"] = alias_collisions[:20]
    return embeddings, metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-info", required=True, help="Path to NCBI gene_info.gz")
    parser.add_argument(
        "--gene-summary",
        help="Optional NCBI gene_summary.gz for richer semantic descriptions",
    )
    parser.add_argument("--tax-id", type=int, default=9606, help="NCBI tax ID")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace embedding model name",
    )
    parser.add_argument(
        "--output", required=True, help="Output pickle path for the embedding dict"
    )
    parser.add_argument("--metadata", help="Optional metadata JSON output path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--no-aliases", action="store_true")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument(
        "--no-normalize", dest="normalize", action="store_false",
        help="Disable l2-normalization of sentence vectors",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    records = build_records(args.gene_info, args.gene_summary, args.tax_id)
    if not records:
        raise RuntimeError("NCBI records are empty; check input files and tax_id")
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        import pickle

        pickle.dump(embeddings, handle)
    if args.metadata:
        Path(args.metadata).write_text(json.dumps(metadata, indent=2))
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "vector_count": len(embeddings),
                "dimension": metadata["dimension"],
                **metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
