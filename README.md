# polars-ml

## Machine Learning Polars Plugin


[![PyPI version](https://badge.fury.io/py/polars-ml.svg)](https://badge.fury.io/py/polars-ml)

# Getting Started
Install from Pypi:
```console
pip install polars-ml
```

# Examples
## Graph Namespace
```python
import polars as pl
import polars_ml as plm

df = pl.DataFrame({
    'src_node': ['V1', 'V2', 'V3'],
    'neighbors': [['V2', 'V4'], ['V3'], ['V1']],
    'weights': [[1.0, 2.0], [0.5], [3.5]]
})

embedding_df = df.with_columns(
    plm.graph.node2vec(source_node=pl.col('src_node'),
                       neighbors=pl.col('neighbors'),
                       weights=pl.col('weights'),
                       is_directed=False,
                       p=1.0,
                       q=1.0,
                       max_neighbors=50,
                       embedding_size=64,
                       random_state=42,
                       verbose=True).alias('embedding')
).select('src_node', 'embedding')

print(embedding_df)
```
```
shape: (3, 2)
┌──────────┬───────────────────────────────────┐
│ src_node ┆ embedding                         │
│ ---      ┆ ---                               │
│ str      ┆ list[f32]                         │
╞══════════╪═══════════════════════════════════╡
│ V1       ┆ [0.521827, -0.314611, … -0.16515… │
│ V2       ┆ [0.335624, -0.041853, … 0.224424… │
│ V3       ┆ [0.274431, -0.210741, … -0.02325… │
└──────────┴───────────────────────────────────┘
```
## Nltk Namespace
```python
import polars as pl
import polars_ml as plm


df = pl.DataFrame({
    'words': ['the', 'bull', 'is', 'running', 'away']
})

df_stemmed = df.with_columns(
    plm.nltk.snowball_stem(pl.col('words'), language='english')
)

print(df_stemmed)
```
```
shape: (5, 1)
┌───────┐
│ words │
│ ---   │
│ str   │
╞═══════╡
│ the   │
│ bull  │
│ is    │
│ run   │
│ away  │
└───────┘
```
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
```python
df_sparse_norm = df_sparse.select('sparse_feature') \
    .with_columns(ps.normalize(pl.col('sparse_feature'), how='vertical', p=2.0).alias('sparse_feature_norm'))
print(df_sparse_norm)
```
```
shape: (4, 2)
┌─────────────────────────┬───────────────────────────────────┐
│ sparse_feature          ┆ sparse_feature_norm               │
│ ---                     ┆ ---                               │
│ struct[3]               ┆ struct[3]                         │
╞═════════════════════════╪═══════════════════════════════════╡
│ {6,[1, 4],[1, 5]}       ┆ {6,[1, 4],[0.707107, 0.857493]}   │
│ {6,[0, 4, 5],[2, 3, 4]} ┆ {6,[0, 4, 5],[1.0, 0.514496, 1.0… │
│ {2,[1],[1]}             ┆ {2,[1],[0.707107]}                │
│ {null,null,null}        ┆ {null,null,null}                  │
└─────────────────────────┴───────────────────────────────────┘
```
# Credits

1. GRAPE for fast and scalable graph processing and random-walk-based embedding. See article [here](https://www.nature.com/articles/s43588-023-00465-8) and library [here](https://github.com/AnacletoLAB/grape).
2. Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost).
3. [Marco Edward Gorelli](https://github.com/MarcoGorelli) - for using his [polars plugin tutorial](https://marcogorelli.github.io/polars-plugins-tutorial).