"""GenePT-w spot embeddings without constructing dense gene vectors."""

import hashlib
import pickle
from pathlib import Path

import numpy as np


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_genept_embeddings(path):
    """Load and validate the official GenePT gene-symbol embedding dictionary."""
    path = Path(path)
    with path.open("rb") as handle:
        source = pickle.load(handle)
    if not isinstance(source, dict) or not source:
        raise ValueError("GenePT asset must contain a nonempty dictionary")
    result = {}
    dimension = None
    for symbol, value in source.items():
        if not isinstance(symbol, str):
            raise TypeError("GenePT dictionary keys must be gene symbols")
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        dimension = len(vector) if dimension is None else dimension
        if len(vector) != dimension or not np.isfinite(vector).all():
            raise ValueError(f"invalid GenePT vector for {symbol}")
        result[symbol.upper()] = vector
    return result, dimension


def source_gene_symbols(root, schema):
    """Return every source gene symbol, independent of the old HVG schema."""
    import h5py

    with h5py.File(Path(schema["source"]) / "spatial_index.h5", "r") as handle:
        genes = handle["genes"][:]
    symbols = []
    for gene in genes:
        name = bytes(gene["geneName"]).decode().rstrip("\0")
        identifier = bytes(gene["geneID"]).decode().rstrip("\0")
        symbols.append((name or identifier).upper())
    return symbols


def make_gene_lookup(symbols, embeddings):
    """Map all source columns onto unique native GenePT gene vectors.

    Multiple source identifiers can share one gene symbol. They map to the same
    compact column so their counts are summed and that biological gene is only
    counted once in a spot's pooling divisor.
    """
    source_to_gene = np.full(len(symbols), -1, dtype=np.int32)
    matched_symbols = []
    compact_indices = {}
    for source_column, symbol in enumerate(symbols):
        symbol = symbol.upper()
        vector = embeddings.get(symbol)
        if vector is None:
            continue
        compact_column = compact_indices.get(symbol)
        if compact_column is None:
            compact_column = len(matched_symbols)
            compact_indices[symbol] = compact_column
            matched_symbols.append(symbol)
        source_to_gene[source_column] = compact_column
    if not matched_symbols:
        raise ValueError("none of the source genes map to the GenePT dictionary")
    matrix = np.stack([embeddings[symbol] for symbol in matched_symbols]).astype(
        np.float32, copy=False
    )
    source_dim = matrix.shape[1]
    metadata = {
        "source_dimension": source_dim,
        "gene_embedding_dimension": source_dim,
        "source_genes": len(symbols),
        "source_unique_symbols": len(set(symbols)),
        "mapped_source_columns": int(np.sum(source_to_gene >= 0)),
        "mapped_unique_genes": len(matched_symbols),
        "mapped_source_fraction": float(np.mean(source_to_gene >= 0)),
        "dimension_reduction": "none; learned linear projection runs in the model",
    }
    return matrix, source_to_gene, np.asarray(matched_symbols), metadata


def pool_spot_embeddings(expression, gene_lookup, mapped_genes):
    """Compute sum(expression_i * GenePT_i) / count(nonzero mapped genes)."""
    columns = np.flatnonzero(mapped_genes)
    mapped_expression = expression[:, columns].tocsr()
    nonzero_genes = np.diff(mapped_expression.indptr).astype(np.int32, copy=False)
    pooled = np.asarray(mapped_expression @ gene_lookup[columns], dtype=np.float32)
    usable = nonzero_genes > 0
    pooled[usable] /= nonzero_genes[usable, None]
    pooled[~usable] = 0
    return pooled, nonzero_genes
