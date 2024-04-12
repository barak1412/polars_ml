import os.path
import polars as pl
from polars.utils.udfs import _get_shared_lib_location
from polars.plugins import register_plugin_function
from polars_ml.sparse.constants import DIM, INDICES, VALUES


_lib = _get_shared_lib_location(os.path.dirname(__file__))


def from_list(expr: pl.Expr) -> pl.expr:
    output_name = expr.meta.output_name()
    return pl.struct(expr.list.len().alias(DIM),
                     expr.list.eval(pl.arg_where(pl.element() != 0)).alias(INDICES),
                     expr.list.eval(pl.element().filter(pl.element() != 0)).alias(VALUES)).alias(output_name)


def normalize(expr: pl.Expr,  *, how: str = 'vertical', p: float = 2.0) -> pl.Expr:
    # validate params
    if how not in ['vertical']:
        raise ValueError(f'Illegal how = {how}, only vertical is supported.')
    if p < 1.0:
        raise ValueError(f'p must be greater or equals to 1.0, {p} was given.')

    return register_plugin_function(
        args=[expr],
        plugin_path=_lib,
        function_name='normalize',
        is_elementwise=True,
        kwargs={'how': how, 'p': p}
    )
