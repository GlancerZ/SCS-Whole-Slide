# GenePT-informed spatial transformer

**a, Whole-slide sampling.** One model is shared across 267 tiles. Each sample consists of a centre spot and 49 neighbouring RNA spots. Spots are retained as separate tokens; neighbourhood selection does not aggregate expression across spatial positions. Sparse gene IDs, expression counts and centre-relative two-dimensional coordinates form the input. There is no fixed 6,000-gene feature cutoff. The schematic spot positions and gene values are illustrative, not measured data.

**b, Per-spot expression encoding.** Frozen GenePT vectors (1,536 dimensions) are weighted by the corresponding expression counts and summed, then divided by the number of distinct nonzero expressed genes with a GenePT mapping. This denominator is not the sum of expression counts. Genes without a GenePT mapping are excluded from both the numerator and denominator. A trainable linear map projects the resulting representation to 256 dimensions. An independent trainable linear map projects the centre-relative coordinates from 2 to 256 dimensions, and the two representations are added. No PCA, neighbouring-spot expression pooling or full-slide dense spot-embedding cache is used.

The drawing presents the conceptual pooling-then-projection operation for clarity. The sparse implementation uses its algebraically equivalent form: project the shared frozen gene table with a bias-free trainable linear map, perform expression-weighted embedding-bag pooling with weights x/k, and add the spot bias once for nonempty spots. An empty mapped-gene set receives a zero expression representation. The native table is frozen, but its projected values depend on current trainable weights.

**c, Spatial transformer and readout.** Fifty 256-dimensional tokens enter 32 pre-layer-normalized residual Transformer blocks. Each block uses four-head scaled dot-product attention (64 dimensions per head) and a feed-forward network of width 256 → 512 → 256 with GELU and dropout. The plus signs inside the block denote residual additions. After final layer normalization, only the centre token is passed to a shared 256 → 4,096 → 1,024 MLP. Two linear heads predict 16 direction-class logits and one foreground logit, respectively. Direction cross-entropy is masked to foreground spots; foreground classification uses binary cross-entropy with logits. The model has 22,527,505 trainable parameters, excluding the frozen GenePT table. This is the 4× setting relative to the 64-wide, 8-layer baseline: both width and depth are scaled by four.

Training uses BF16 mixed precision. Muon updates the Transformer hidden matrices, while AdamW updates the remaining trainable parameters. This figure describes architecture, not evidence of segmentation accuracy.

Source: `optimizations/scs_streaming/torch_model.py`, scale=4, 50 neighbours/tokens, 16 direction classes, input_dim=1536.

## Regeneration

```bash
python figures/genept_spatial_transformer/draw_architecture.py
rsvg-convert -w 3600 -o figures/genept_spatial_transformer/architecture.png figures/genept_spatial_transformer/architecture.svg
rsvg-convert -f pdf -o figures/genept_spatial_transformer/architecture.pdf figures/genept_spatial_transformer/architecture.svg
```

SVG retains editable text and vector shapes; PDF is vector; PNG is a 2× raster preview. The layout is publication-inspired and is not an official Nature template.
