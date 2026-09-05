"""PyTorch minibatch loading over compact whole-slide SCS tiles."""

import numpy as np
from torch.utils.data import DataLoader, Dataset

from .shared_data import SharedBatches, TileCache


class PlannedBatchDataset(Dataset):
    """One deterministic epoch plan, with one lazy tile cache per worker."""

    def __init__(self, batches, epoch=0, batch_size=None, merge_tiles=True):
        self.batches = batches
        source = list(batches.plan(epoch))
        self.plan = (
            self._merge(source, batch_size or batches.batch_size)
            if merge_tiles
            else [[item] for item in source]
        )
        self.cache = None

    @staticmethod
    def _merge(source, batch_size):
        """Pack consecutive tile-local chunks into full physical batches."""
        result, current, count = [], [], 0
        for tile_id, source_rows in source:
            offset = 0
            while offset < len(source_rows):
                take = min(batch_size - count, len(source_rows) - offset)
                current.append((tile_id, source_rows[offset : offset + take]))
                count += take
                offset += take
                if count == batch_size:
                    result.append(current)
                    current, count = [], 0
        if current:
            result.append(current)
        return result

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        parts = self.plan[index]
        if self.cache is None:
            self.cache = TileCache(
                self.batches.root, self.batches.schema, capacity=max(2, len(parts))
            )
        arrays = [self.cache.get(tile_id).batch(rows) for tile_id, rows in parts]
        expression = np.concatenate([part[0][0] for part in arrays], axis=0)
        positions = np.concatenate([part[0][1] for part in arrays], axis=0)
        directions = np.concatenate([part[1][0] for part in arrays], axis=0)
        foreground = np.concatenate([part[1][1] for part in arrays], axis=0)
        # Direction rows for background examples are all zero in the compact
        # format. Argmax is harmless because the loss masks those examples.
        return expression, positions, directions.argmax(axis=-1), foreground


def whole_slide_batches(
    root,
    split,
    batch_size,
    per_class_cap,
    seed,
    epoch=0,
    workers=4,
    prefetch=2,
    merge_tiles=True,
):
    batches = SharedBatches(root, split, batch_size, per_class_cap, seed)
    dataset = PlannedBatchDataset(batches, epoch, batch_size, merge_tiles)
    kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers:
        kwargs.update(prefetch_factor=prefetch, persistent_workers=False)
    return batches, DataLoader(**kwargs)
