from typing import Any, Callable
import polars as pl
from polars_ml.sparse import functions


@pl.api.register_expr_namespace("sparse")
class SparseNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def __getattr__(self, function_name: str) -> Callable[[Any], pl.Expr]:
        def func(*args: Any, **kwargs: Any) -> pl.Expr:
            return getattr(functions, function_name)(
                self._expr, *args, **kwargs
            )
        return func