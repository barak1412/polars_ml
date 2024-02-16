import os.path
from typing import Union
import polars as pl
from polars.utils.udfs import _get_shared_lib_location

_lib = _get_shared_lib_location(os.path.dirname(__file__))


def snowball_stem(expr: Union[str, pl.Expr], *, language: str) -> pl.Expr:
    return expr.register_plugin(
        lib=_lib,
        symbol="snowball_stem",
        is_elementwise=True,
        kwargs={'language': language},
    )