import os.path
import polars as pl
from polars.utils.udfs import _get_shared_lib_location
from polars.plugins import register_plugin_function


_lib = _get_shared_lib_location(os.path.dirname(__file__))


def snowball_stem(expr: pl.Expr, *, language: str) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=_lib,
        function_name='snowball_stem',
        is_elementwise=True,
        kwargs={'language': language}
    )