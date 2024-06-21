import polars as pl
from polars.plugins import register_plugin_function
from polars_ml import lib


def snowball_stem(expr: pl.Expr, *, language: str) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=lib,
        function_name='snowball_stem',
        is_elementwise=True,
        kwargs={'language': language}
    )