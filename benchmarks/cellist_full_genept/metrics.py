"""Segmentation metrics on a common RNA support; no model output used as truth."""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def aggregate(x, labels, size=None):
    labels = np.asarray(labels, dtype=np.int64)
    size = int(labels.max(initial=0))+1 if size is None else size
    mapper = sparse.csr_matrix((np.ones(len(labels)), (labels, np.arange(len(labels)))), shape=(size, len(labels)))
    return (mapper @ x).tocsr()


def compact(labels):
    ids = np.unique(np.r_[0, labels])
    return ids, np.searchsorted(ids, labels)


def row_pearson(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    n = a.shape[1]
    sa, sb = np.asarray(a.sum(1)).ravel(), np.asarray(b.sum(1)).ravel()
    aa = np.asarray(a.multiply(a).sum(1)).ravel()-sa*sa/n
    bb = np.asarray(b.multiply(b).sum(1)).ravel()-sb*sb/n
    ab = np.asarray(a.multiply(b).sum(1)).ravel()-sa*sb/n
    denom = np.sqrt(np.maximum(aa, 0)*np.maximum(bb, 0))
    return np.clip(np.divide(ab, denom, out=np.full(len(sa), np.nan), where=denom>1e-12), -1, 1)


def random_correlations(x, coords, labels, repeats=10, seed=20260207):
    """Primary published implementation: random SPOTS, plus random cell-centroid axes."""
    ids, code = compact(labels); n = len(ids)
    count = np.bincount(code, minlength=n)
    cx = np.bincount(code, weights=coords[:,0], minlength=n)/np.maximum(count, 1)
    cy = np.bincount(code, weights=coords[:,1], minlength=n)/np.maximum(count, 1)
    total = aggregate(x, code, n)
    rng = np.random.default_rng(seed)
    random_values, direction_values = [], []
    for _ in range(repeats):
        mask = rng.choice([False, True], size=len(code))
        left = aggregate(x.multiply(mask[:,None]), code, n)
        random_values.append(row_pearson(left, total-left))
        angle = rng.uniform(0, np.pi)
        mask = (coords[:,0]-cx[code])*np.cos(angle)+(coords[:,1]-cy[code])*np.sin(angle) >= 0
        left = aggregate(x.multiply(mask[:,None]), code, n)
        direction_values.append(row_pearson(left, total-left))
    def average(values):
        values = np.asarray(values)
        valid = np.isfinite(values).sum(0)
        return np.divide(np.nansum(values,axis=0), valid, out=np.full(n,np.nan), where=valid>0), valid
    random_mean, random_n = average(random_values)
    direction_mean, direction_n = average(direction_values)
    return dict(cell_id=ids[1:], n_spots=count[1:],
        n_umis=np.asarray(total.sum(1)).ravel()[1:], n_genes=np.diff(total.indptr)[1:],
        random_correlation=random_mean[1:], random_valid_repeats=random_n[1:],
        directional_correlation=direction_mean[1:], directional_valid_repeats=direction_n[1:])


def best_overlap(source, target):
    valid = (source>0)&(target>0)
    if not valid.any(): return {}
    pairs, counts = np.unique(np.stack([source[valid],target[valid]],1), axis=0, return_counts=True)
    result, scores = {}, {}
    for (s,t), count in zip(pairs, counts):
        if count > scores.get(int(s), -1):
            result[int(s)] = int(t); scores[int(s)] = int(count)
    return result


def cross_correlation(x, source, target):
    """Author-style source-to-target relabeling, including many-to-one unions.

    Both scores use exactly the same finite, nonempty three-region population.
    Call again with swapped methods; never rank across the two populations.
    """
    mapping = best_overlap(source, target)
    max_source = int(np.max(source, initial=0))
    remap = np.zeros(max_source+1, np.int64)
    for s,t in mapping.items(): remap[s] = t
    mapped = remap[source]
    size = int(np.max(target, initial=0))+1
    equal = (mapped == target)&(target>0)
    shared = aggregate(x.multiply(equal[:,None]), target, size)
    a = aggregate(x, mapped, size)-shared
    b = aggregate(x, target, size)-shared
    a.eliminate_zeros(); b.eliminate_zeros()
    if np.any(a.data < -1e-8) or np.any(b.data < -1e-8):
        raise AssertionError('Invalid overlap expression subtraction')
    ra, rb = row_pearson(shared,a), row_pearson(shared,b)
    umis = np.stack([np.asarray(v.sum(1)).ravel() for v in (shared,a,b)],1)
    valid = np.isfinite(ra)&np.isfinite(rb)&(umis.min(1)>0)
    valid[0] = False
    rows = np.flatnonzero(valid)
    return dict(target_cell=rows, source_correlation=ra[rows], target_correlation=rb[rows],
        overlap_umis=umis[rows,0], source_unique_umis=umis[rows,1], target_unique_umis=umis[rows,2],
        sensitivity_100umi=np.min(umis[rows],1)>=100,
        source_cells_matched=len(mapping), target_cells_matched=len(set(mapping.values())),
        source_cells_total=len(np.unique(source[source>0])),
        many_to_one_excess=len(mapping)-len(set(mapping.values())))


def transcript_iou(manual, predicted):
    """Exact mutual-best intersection/union on molecule rows; also penalize missed GT."""
    manual, predicted = np.asarray(manual), np.asarray(predicted)
    if len(manual) != len(predicted) or np.any(manual<=0):
        raise ValueError('Expected one positive manual ID and one prediction per molecule')
    m2p, p2m = best_overlap(manual,predicted), best_overlap(predicted,manual)
    gt = np.unique(manual)
    scores, mutual = [], []
    for g in gt:
        p = m2p.get(int(g))
        matched = p is not None and p2m.get(p) == int(g)
        value = float(np.sum((manual==g)&(predicted==p))/np.sum((manual==g)|(predicted==p))) if matched else 0.
        scores.append(value); mutual.append(matched)
    scores, mutual = np.asarray(scores), np.asarray(mutual)
    return dict(manual_cells=len(gt), mutual_matches=int(sum(mutual)),
        matched_only_mean=float(scores[mutual].mean()) if mutual.any() else None,
        matched_only_median=float(np.median(scores[mutual])) if mutual.any() else None,
        all_gt_mean=float(scores.mean()), all_gt_median=float(np.median(scores)),
        all_gt_recall_iou50=float(np.mean(scores>=.5)),
        matched_fraction=float(mutual.mean()), manual_ids=gt.tolist(), all_gt_ious=scores.tolist(),
        definition='transcript-row IoU; mutual best overlap; unmatched GT=0 in all_gt metrics; not polygon-mask IoU')


def fano_median(x):
    if x.shape[0]<2: return np.nan
    mean = np.asarray(x.mean(0)).ravel()
    second = np.asarray(x.multiply(x).mean(0)).ravel()
    var = np.maximum(second-mean*mean, 0)*x.shape[0]/(x.shape[0]-1)
    valid = mean>0
    return float(np.median(var[valid]/mean[valid])) if valid.any() else np.nan


def paired_purity(x, coords, nucleus_ids, nucleus_centers, first, second, resolution_um,
                  neigh_dist_um=2.5, half_width_units=20):
    """Shared HVG expression enhanced with the official inverse-square spatial kernel.

    Common nucleus-centered windows, >100 observed spots for BOTH methods.
    Only common finite pairs are summarized; all coverage exclusions are returned.
    """
    first_map, second_map = best_overlap(nucleus_ids, first), best_overlap(nucleus_ids, second)
    common = sorted(set(first_map)&set(second_map))
    if not common:
        return [], dict(common_nuclei=0,eligible_nuclei=0,unique_cell_pairs=0,
            neighborhood_side_um=2*half_width_units*resolution_um,
            enhancement_radius_coordinate_units=int(neigh_dist_um/resolution_um),
            minimum_spots_exclusive=100,kernel='1/distance^2 for nonself; self=1; no normalization')
    tree = cKDTree(coords)
    distances = tree.sparse_distance_matrix(tree, int(neigh_dist_um/resolution_um), output_type='coo_matrix')
    use = distances.data>0
    weight = sparse.csr_matrix((1/distances.data[use]**2, (distances.row[use], distances.col[use])),shape=(len(coords),len(coords)))
    weight.setdiag(1)
    enhanced = (weight @ x.astype(np.float64)).tocsr()
    half_width = half_width_units
    rows=[]
    def groups(labels):
        order=np.argsort(labels,kind='stable'); ids,start,count=np.unique(labels[order],return_index=True,return_counts=True)
        return {int(i):order[s:s+n] for i,s,n in zip(ids,start,count)}
    groups_a,groups_b=groups(first),groups(second)
    for nucleus in common:
        a,b=first_map[nucleus],second_map[nucleus]
        ai,bi=groups_a[a],groups_b[b]
        center=nucleus_centers[nucleus]
        neighborhood=np.array(tree.query_ball_point(center,half_width,p=np.inf),dtype=np.int64)
        if min(len(ai),len(bi),len(neighborhood))<=100: continue
        denom=fano_median(enhanced[neighborhood])
        fa,fb=fano_median(enhanced[ai]),fano_median(enhanced[bi])
        if not np.isfinite([fa,fb,denom]).all() or denom<=0: continue
        rows.append(dict(nucleus=nucleus,cell_first=a,cell_second=b,n_spots_first=len(ai),n_spots_second=len(bi),
            purity_first=fa/denom,purity_second=fb/denom,neighborhood_spots=len(neighborhood)))
    return rows, dict(common_nuclei=len(common),eligible_nuclei=len(rows),
        unique_cell_pairs=len({(r['cell_first'],r['cell_second']) for r in rows}),
        neighborhood_side_um=2*half_width*resolution_um,enhancement_radius_coordinate_units=int(neigh_dist_um/resolution_um),
        minimum_spots_exclusive=100,kernel='1/distance^2 for nonself; self=1; no normalization')
