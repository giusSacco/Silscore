# Local data sources

Acquire the source data separately; they are not bundled. Methods and
Supplementary Information are described in the [manuscript](https://arxiv.org/abs/2611.15812).
The repository [LICENSE](../LICENSE) is provisionally all-rights-reserved.
Check the applicable terms for source software, data and derived results.

| Local path | Source | SHA-256 |
| --- | --- | --- |
| Source supplement | Huesken et al. (2005), Supplementary Table 3, `41587_2005_BFnbt1118_MOESM4_ESM.pdf`; [primary record](https://doi.org/10.1038/nbt1118). | `4d6cbaca93d98fa738659dfaf68875af3039b2061a94aa550b0c5a5b10e83d63` |
| `data/processed/huesken_canonical_split.csv` | Canonical Huesken split from that supplement: 2,182 training and 249 test rows. | `03508de5bb7a66208c7f3910e2d99300356c16fed88b0a93e72f2fff4ea6f074` |
| `data/raw/Mix.csv` | Mix evaluation table; [versioned source](https://raw.githubusercontent.com/lulab/OligoFormer/e2f53ad63387bbe166bf123949151e2bc9bf6ec3/data/Mix.csv). | `10c34deaa5a02e5a168e6951edbd7144fe9025ad40a679c3defb8fe726620c87` |
| `data/raw/Taka.csv` | Katoh–Suzuki evaluation table; [versioned source](https://raw.githubusercontent.com/lulab/OligoFormer/e2f53ad63387bbe166bf123949151e2bc9bf6ec3/data/Taka.csv). | `221672544ee0bd7e4c8029d9e8f5d7463c90aa9f5623976002b5e68c35300b2f` |

OligoFormer source version: `e2f53ad63387bbe166bf123949151e2bc9bf6ec3`;
see its [LICENSE](https://github.com/lulab/OligoFormer/blob/e2f53ad63387bbe166bf123949151e2bc9bf6ec3/LICENSE)
and [README](https://github.com/lulab/OligoFormer/blob/e2f53ad63387bbe166bf123949151e2bc9bf6ec3/README.md)
for the source terms. External code, weights and inputs are not included here.
Torch and RNA-FM are not Silscore runtime dependencies.

Comparison methods: Reynolds et al. ([2004](https://doi.org/10.1038/nbt936)),
Ui-Tei et al. ([2004](https://doi.org/10.1093/nar/gkh247)), Ichihara et al.
([2007](https://doi.org/10.1093/nar/gkm699), i-Score), and Vert et al.
([2006](https://doi.org/10.1186/1471-2105-7-520), DSIR).
Thermodynamic parameter sources are recorded in `src/silscore/thermo.py`.

## Prepare local inputs and analysis caches

Download Huesken Supplementary Table 3 from the source above. Extract it once
using [PyMuPDF](https://pymupdf.readthedocs.io/) in a temporary environment:

```bash
python -m venv /tmp/silscore-pdf
/tmp/silscore-pdf/bin/python -m pip install PyMuPDF==1.26.7
/tmp/silscore-pdf/bin/python scripts/extract_huesken.py --pdf /path/to/table3.pdf
```

The extractor verifies the PDF and resulting CSV against the checksums above.
PyMuPDF is used only for this preparation step. Place `Mix.csv` and `Taka.csv`
at the paths above, then run from the repository root in the `silscore` environment:

```bash
python scripts/prepare_data.py --strict-canonical-counts
python scripts/reproduce.py --overwrite
```

The runner rebuilds the six Silscore caches. Input preparation validates local
files and records checksums without converting or reconstructing them. The included model uses
only canonical Huesken training rows. To regenerate the model resource:

```bash
python scripts/silscore.py train \
  --input data/processed/huesken_canonical_split.csv \
  --output src/silscore/resources/silscore.json
```

## Recompute the OligoFormer comparison statistics

`data/oligoformer/` contains five tables with evaluation IDs, frozen Silscore
predictions and 25 OligoFormer predictions per record. Columns identify the
validation split (seeds 43–47) and training initialization (seeds 0–4).
Sequences and activity labels are read from the local input CSVs, not included
in these prediction tables. Human and rodent evaluations share the same test IDs.

With the three local CSVs prepared, recompute the 4000-replicate intervals on CPU:

```bash
python scripts/analysis/prepare_oligoformer.py --output output/oligoformer-check
cmp output/oligoformer-check/oligoformer.json data/cache/oligoformer.json
```

The point estimate is the mean of 25 run-level Pearson correlations. Bootstrap
samples jointly resample evaluation records and independently resample the five
validation splits and five initializations. Human and rodent contrasts use the
same resamples. No training, inference or checkpoint files are needed.

## GPU training protocol

To repeat training, use the upstream version linked above, its RNA-FM features,
and its unchanged preprocessing, training and prediction entry points. Target
contexts are 57 nt; we used the released contexts and recovered 70 missing Huesken
contexts from Supplementary Table 2, with boundary padding for nine records.
For each of the three training pools (Huesken, human-489, rodent-489), use five
stratified 80/20 fit/validation splits (seeds 43–47) and five PyTorch training
seeds (0–4): 75 fits in total. Python/NumPy seeds stay at 42; inference uses
PyTorch seed 0. Use the released training defaults and keep evaluation records
outside fitting and checkpoint selection. GPU training is outside this
repository's automated reproduction workflow.

## Update the figure notebooks

Edit the paired `.py` files in `notebooks/`, then regenerate the executed
`.ipynb` files in the `silscore` environment:

```bash
python scripts/export_notebooks.py
```

Each notebook can also be opened and run independently using the included caches.
