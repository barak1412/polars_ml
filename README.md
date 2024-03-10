# polars-ml

## Machine Learning Polars Plugin


[![PyPI version](https://badge.fury.io/py/polars-ml.svg)](https://badge.fury.io/py/polars-ml)

# Getting Started
Install from Pypi:
```console
pip install polars-ml
```

# Examples
## Sparse Namespace

```python
import polars as pl
import polars_ml.sparse as ps


df = pl.DataFrame({
    'feature': [
        [0, 1, 0, 0, 5, 0],
        [2, 0, 0, 0, 3, 4],
        [0, 1],
        None
    ]
})

df_sparse = df.with_columns(
   ps.from_list(pl.col('feature')).alias('sparse_feature')
)

print(df_sparse)
```
```
shape: (4, 2)
┌─────────────┬─────────────────────────┐
│ feature     ┆ sparse_feature          │
│ ---         ┆ ---                     │
│ list[i64]   ┆ struct[3]               │
╞═════════════╪═════════════════════════╡
│ [0, 1, … 0] ┆ {6,[1, 4],[1, 5]}       │
│ [2, 0, … 4] ┆ {6,[0, 4, 5],[2, 3, 4]} │
│ [0, 1]      ┆ {2,[1],[1]}             │
│ null        ┆ {null,null,null}        │
└─────────────┴─────────────────────────┘
```

# Credits

1. Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)
2. [Marco Edward Gorelli](https://github.com/MarcoGorelli) - for using his [polars plugin tutorial](https://marcogorelli.github.io/polars-plugins-tutorial).