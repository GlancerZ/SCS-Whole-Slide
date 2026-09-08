# Upstream source dependencies for the benchmark snapshot

The main repository does not vendor complete upstream checkouts. The benchmark
uses these exact versions plus the source patches in this directory:

| Dependency | Repository | Commit | Patch license |
|---|---|---|---|
| SCS | https://github.com/chenhcs/SCS | `44bc89b2fccdc51cbd59ecfd50d1ff5a2134f378` | MIT; see `scs/LICENSE` |
| Cellist | https://github.com/wanglabtongji/Cellist | `7f6781786e8745fbc3569e1fc1f644c3e705be5e` | GPL-3.0; see `cellist/LICENSE` |

From a fresh main-repository clone, obtain and patch the dependencies:

```bash
git clone https://github.com/chenhcs/SCS.git SCS
git -C SCS checkout --detach 44bc89b2fccdc51cbd59ecfd50d1ff5a2134f378
git -C SCS apply --check ../upstream_patches/scs/local.patch
git -C SCS apply ../upstream_patches/scs/local.patch

mkdir -p tools
git clone https://github.com/wanglabtongji/Cellist.git tools/Cellist
git -C tools/Cellist checkout --detach 7f6781786e8745fbc3569e1fc1f644c3e705be5e
git -C tools/Cellist apply --check ../../upstream_patches/cellist/local.patch
git -C tools/Cellist apply ../../upstream_patches/cellist/local.patch
```

Do not apply these commands over existing locally modified checkouts. The SCS
patch also adds `src/spateo_compat.py`, used directly by preparation scripts.
It includes the current preprocessing, postprocessing, and TensorFlow runtime
adaptations. The Cellist patch controls CPU thread counts and optional imports;
the benchmark-specific degenerate-HVG guard lives in `cellist.py`.

Upstream licenses and copyright notices remain applicable to their respective
code and patches. In particular, the Cellist patch is not relicensed under the
main repository's MIT license. These are source patches, not packaged binaries.

The runtime remains split into Python 3.9 preparation/evaluation, Python 3.10
Cellist, and Python 3.11 PyTorch environments. The exact observed package
versions are recorded in `benchmarks/cellist_full_genept/results/environments/`.
`requirements-scs.txt` describes the older SCS reference environment and is not a
single combined installation recipe for all three environments. Obtain datasets
and the GenePT embedding dictionary separately as described in the benchmark
protocol. This is a source snapshot of the cluster workflow, not a data-inclusive
one-command reproduction package.
