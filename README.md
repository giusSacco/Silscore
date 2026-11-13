# Silscore

*Modelling siRNA potency with positional thermodynamics and data-selected
nucleotide-identity terms*

[![Tests](https://github.com/giusSacco/Silscore/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/giusSacco/Silscore/actions/workflows/tests.yml)
[![Python ≥3.10](https://img.shields.io/badge/Python-%E2%89%A53.10-blue)](pyproject.toml)

Silscore ([Sacco et al., 2026](https://arxiv.org/abs/2611.15812)) is a physics-based model for predicting the silencing efficacy of siRNAs
from their sequence. It combines RNA duplex thermodynamics with data-selected
sequence features to describe how duplex stability and nucleotide position
contribute to predicted efficacy.

Provide an RNA target sequence to score all complementary 19-nucleotide siRNAs
along it using the included pretrained model. You can also score guides you
already have or retrain the model on your own experimental efficacy measurements.

For a 19-nucleotide guide $g$, the model predicts efficacy as:

```math
\hat{y}(g) = \beta_0
 \;-\; \underbrace{\kappa\bigl(\Delta G(g)-\Delta G^{*}\bigr)^{2}}_{\text{global stability}}
 \;+\; \underbrace{\sum_{i=1}^{18} w_i\,\Delta G_i}_{\text{local stability}}
 \;+\; \underbrace{\sum_{(p,b)\in\mathcal{S}} \theta_p(b)\,\delta_{g_p,b}}_{\text{nucleotide identity}} \; .
```

- **Global stability:** the quadratic term captures a preference for intermediate
  duplex stability, with fitted curvature $\kappa$ and optimum $\Delta G^{*}$.
- **Local stability:** the fitted weights $w_i$ describe where local duplex
  stability favours or disfavours silencing. Their sum is constrained to zero.
- **Nucleotide identity:** screening of training residuals selects a set $\mathcal{S}$
  of position–base pairs (six in the pretrained model). Each coefficient contributes
  only when the guide has that base at that position. These terms describe sequence
  effects beyond duplex stability; [recognition of the guide's 5′ nucleotide by
  Argonaute](https://doi.org/10.1038/nature09039) is one such mechanism.

All coefficients are fitted from experimental efficacy measurements.

See [the manuscript](https://arxiv.org/abs/2611.15812) for predictive performance, benchmarking, parameter
interpretation and further analyses.

## Installation

Install Silscore from the repository with pip:

```bash
python -m pip install .
```

This installs both the `silscore` Python package and the `silscore` command. For
development, use `python -m pip install -e .`. The Conda environment in
`environment.yml` remains available for reproducing the paper analyses and figures.

The model can also be used from Python:

```python
from silscore import load_default_model, predict_targets

model = load_default_model()
target = (
    "CCGAAUAUUUCAAGAUAAUUACGAUAGACGAUGCACAUCCACGCAGUCCGAUA"
    "CAGGAACUUUGUGUUAUCGCCAUCCAUAACUUGGGUCUACCUCCUAA"
)
predictions = predict_targets(model, [("target", target)])
print(len(predictions))  # 82 overlapping 19-nt windows
```

## Usage

The primary prediction workflow accepts an RNA target sequence and scans every
19-nucleotide window, moving one nucleotide at a time. Provide the target in
5′→3′ orientation, with at least 19 nucleotides. Any `T` is converted to `U`
with a warning.

This synthetic 100-nucleotide target produces 82 predictions, one for each
19-nucleotide window:

```bash
silscore predict \
  --target CCGAAUAUUUCAAGAUAAUUACGAUAGACGAUGCACAUCCACGCAGUCCGAUACAGGAACUUUGUGUUAUCGCCAUCCAUAACUUGGGUCUACCUCCUAA
```

This example takes under a second on an Apple M1 CPU.

The pretrained model is included in the installed package, and the default output is
`output/predictions.csv`. The output columns are `target_id`, `start`, `end`,
`target`, `guide`, and `score`. Each row reports the 19-nucleotide target window
and its reverse-complement guide.
Coordinates are 1-based and inclusive.
With the default model, higher scores indicate greater predicted silencing.

For multiple targets, provide a CSV with columns `id,target`:

```csv
id,target
target_001,CCGAAUAUUUCAAGAUAAUUACGAUAGACGAUGCACAUCCACGCAGUCCGAUACAGGAACUUUGUGUUAUCGCCAUCCAUAACUUGGGUCUACCUCCUAA
target_002,CUGCAGAGAGCCCAUUAUUGAGAGGUUGCGUAUCUUCAGGGAGCACGGCGUGCAUAAGGGGAAUGAUUUCUCUUCAACGAACAUGGGAGGACGCCGUGUU
```

Run the same prediction command with `--input`:

```bash
silscore predict \
  --input targets.csv --output output/target_predictions.csv
```

The batch output uses the same window-level columns, with the input `id`
reported as `target_id`. To score guides you already have, use columns `id,guide`
with uppercase 19-nucleotide RNA guide sequences. The same `predict --input`
command then produces columns `id`, `guide`, and `score`.

## Retraining

Prepare a CSV with columns `id,guide,efficacy`. Guides are 19-nucleotide RNA
sequences in 5′→3′ orientation, and larger efficacy values must represent
stronger silencing.

```bash
silscore train \
  --input training.csv --output output/my_model.json
```

For reference, training on the 2,182-guide Huesken training set takes under a second
on an Apple M1 CPU.

If the CSV contains a `split` column, label rows `train` or `test`; only
`train` rows are used for fitting. Without `split`, all rows are used.

To use a model fitted from another dataset, pass its JSON path explicitly:

```bash
silscore predict \
  --model output/my_model.json \
  --target CCGAAUAUUUCAAGAUAAUUACGAUAGACGAUGCACAUCCACGCAGUCCGAUACAGGAACUUUGUGUUAUCGCCAUCCAUAACUUGGGUCUACCUCCUAA
```

## Paper and figures

This repository also contains instructions and [scripts](scripts/analysis/) to reproduce
model fits, analyses and figures from the manuscript.

The [performance notebook](notebooks/nb1_performance.ipynb) and the other
[figure notebooks](notebooks/) (`.ipynb`) include saved figures for viewing
on GitHub. To run them locally, create the analysis environment:

```bash
conda env create -f environment.yml
conda activate silscore
```

Select the `silscore` environment as the notebook kernel and run all cells.
Only the included caches are needed. See [data/SOURCES.md](data/SOURCES.md)
for data preparation, cache regeneration and updating the notebooks.

## Citation

If you use Silscore in your research, please cite:

Giuseppe Sacco, Isabella Baldini, Riccardo Paone, Mattia Capulli, Guido Sanguinetti,
and Giovanni Bussi. *Silscore: modelling siRNA potency with positional
thermodynamics and data-selected nucleotide-identity terms*.

Preprint, 2026. [arXiv:2611.15812](https://arxiv.org/abs/2611.15812).
See the manuscript's Methods and Supplementary Information for analysis details.
