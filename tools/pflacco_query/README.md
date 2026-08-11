# Isolated pflacco query extractor

This tool is the only supported pflacco execution boundary. It uses Python 3.11 and
`pflacco==1.2.2`; it reads an existing landscape-query sample Parquet file and writes
one fixed feature Parquet file. It does not import or evaluate benchmark functions.

Create the environment from this directory with `uv sync`, then run:

```bash
uv run python extract.py \
  --query-id pflacco_standard \
  --samples ../../results/landscape_queries/samples/lhs_50d/bbob_train/samples.parquet \
  --output ../../results/landscape_queries/features/pflacco_standard/bbob_train/features.parquet
```

Use `pflacco_broad` with `lhs_100d`. The extractor fixes NBC tie breaking to
`dist_tie_breaker="first"` and derives the information-content seed from explicit
integer `SeedSequence` inputs. It never installs R packages, downloads dependencies at
runtime, evaluates the objective function, or falls back to custom formulas.
