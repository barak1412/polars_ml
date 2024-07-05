import polars as pl
from polars.plugins import register_plugin_function
from polars_ml import lib
from polars_ml.sparse.constants import DIM, INDICES, VALUES


def from_list(expr: pl.Expr) -> pl.expr:
    output_name = expr.meta.output_name()
    return pl.struct(pl.when(expr.is_null()).then(None).otherwise(expr.list.len()).alias(DIM),
                     expr.list.eval(pl.arg_where(pl.element() != 0)).alias(INDICES),
                     expr.list.eval(pl.element().filter(pl.element() != 0)).alias(VALUES)).alias(output_name)


def get(expr: pl.Expr, index: int) -> pl.expr:
    # validate index is non-negative
    if index < 0:
        raise Exception('index must be non-negative value')

    output_name = expr.meta.output_name()
    return pl.when(expr.is_null()).then(None) \
        .when(expr.struct.field(DIM) <= index).then(None) \
        .when(expr.struct.field(INDICES).list.contains(index)) \
        .then(
            expr.struct.field(VALUES).list.get(
                expr.struct.field(INDICES).list.eval(pl.arg_where(pl.element() == index)).list.get(0, null_on_oob=True)
        )
    ).otherwise(0).alias(output_name)


def normalize(expr: pl.Expr,  *, how: str = 'vertical', p: float = 2.0) -> pl.Expr:
    # validate params
    if how not in ['vertical']:
        raise ValueError(f'Illegal how = {how}, only vertical is supported.')
    if p < 1.0:
        raise ValueError(f'p must be greater or equals to 1.0, {p} was given.')

    return register_plugin_function(
        args=[expr],
        plugin_path=lib,
        function_name='normalize',
        is_elementwise=True,
        kwargs={'how': how, 'p': p}
    )
