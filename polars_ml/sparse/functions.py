import os.path
import polars as pl
from polars.utils.udfs import _get_shared_lib_location


_lib = _get_shared_lib_location(os.path.dirname(__file__))


def from_list(expr: pl.Expr) -> pl.Expr:
    return expr.register_plugin(
        lib=_lib,
        symbol="from_list",
        is_elementwise=True,
    )


def normalize(expr: pl.Expr) -> pl.Expr:
    return expr.register_plugin(
        lib=_lib,
        symbol="normalize",
        is_elementwise=True,
    )