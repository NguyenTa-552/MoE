# MVTec metadata split example

This is a complete, deterministic split example generated from the local upstream
`DAPO/data/mvtec/meta.json` at commit
`290411d7374cc2182ad3491b826131af1ff73224`. The exact source URL and SHA256 are in
`provenance.json`. No images, masks, checkpoint, or training results are included.

## What the existing metadata supplies

An actual record is:

```json
{
  "img_path": "bottle/test/broken_small/000.png",
  "mask_path": "bottle/ground_truth/broken_small/000_mask.png",
  "cls_name": "bottle",
  "specie_name": "broken_small",
  "anomaly": 1
}
```

`cls_name` supplies the product label and `specie_name` the defect label. The
splitter uses both for stratification, and `img_path` identifies each image.
You do not enter these labels again or manually assign images to roles.

## Generated files and concrete assignments

| File | Image count | One actual assigned image |
| --- | ---: | --- |
| `splits/expert_train.json` | 1,036 | `bottle/test/broken_large/014.png` |
| `splits/router_train.json` | 364 | `bottle/test/broken_large/015.png` |
| `splits/source_val.json` | 325 | `bottle/test/broken_large/012.png` |

For the 20 `bottle/broken_large` images, the distribution is exactly **12 / 4 / 4**.
Across the full metadata, all **88 product/species combinations** occur in every
role. Counts approximate 60/20/20 after rounding within each stratum. No image path
appears in multiple roles. `splits/summary.json` records counts, label coverage,
manifest hashes, seed 0, and the independent-image assumption.

The example uses MVTec's metadata **test partition as supervised source data**,
following the referenced DAPO training protocol. Target evaluation remains separate.
Product/species labels are not physical-instance IDs; this image split assumes
independence. An optional verified group map can preserve known related views.

## Use this example on the server

From the repository root:

```bash
cp moe/examples/mvtec/experiment.example.json moe/configs/local.json
```

Edit the two placeholder paths in your new local config:

```json
{
  "baseline_checkpoint": "/actual/server/path/to/epoch_4.pth",
  "source_root": "/actual/server/path/to/mvtec",
  "splits": "moe/examples/mvtec/splits"
}
```

The source root must contain the referenced `bottle/`, `cable/`, and other product
directories with images and masks. Check the remaining model/vocabulary settings
against your reproduced checkpoint. The example leaves
`single_type_mask_verified=false`; enable it only after verifying that each binary
mask's positive pixels carry the sample's image-type label, as required by this
dense training mode. Combined categories retain their existing annotation meaning.

These pre-generated manifests reference the source records captured here. If your
server metadata differs, generate fresh manifests directly from that metadata:

```bash
python -m moe.cli split \
  --metadata /actual/server/path/to/mvtec/meta.json \
  --output manifests/mvtec-source --seed 0
```

Then set `"splits": "manifests/mvtec-source"` in the local config. No `--groups`
argument is required. Existing output directories cannot be overwritten.

## Validate and submit

```bash
bash moe/scripts/run_cluster.sh preflight --config moe/configs/local.json
# After preflight succeeds; use a fresh output root and your available budget.
bash moe/scripts/run_cluster.sh source moe/configs/local.json runs/mvtec-example 0
```

The launcher uses `student9696`, QoS `qos96`, and Conda environment `DAPOenv`.
Preflight checks configuration and split integrity; actual asset loading and
baseline-equivalence checks occur at the beginning of the source study. This
example has only been validated as metadata, not executed as a research experiment.
