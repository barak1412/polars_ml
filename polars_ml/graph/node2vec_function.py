import os.path
import polars as pl
from polars.utils.udfs import _get_shared_lib_location
from polars.plugins import register_plugin_function


_lib = _get_shared_lib_location(os.path.dirname(__file__))


def node2vec(source_node: pl.Expr, neighbors: pl.Expr, weights: pl.Expr = None,
            is_directed: bool = False,
            walk_length: int = 10,
            num_of_walks: int = 8,
            p: float = 1.0,
            q: float = 1.0,
            max_neighbors: int = None,
            normalize_by_degree: bool = False,
            model_type: str = 'skipgram',
            embedding_size: int = 64,
            window_size: int = 5,
            embedding_type: str = 'central',
            random_state: int = 42,
            verbose: bool = False) -> pl.Expr:

    func_kwargs = {
        'is_directed': is_directed,
        'walk_length': walk_length,
        'num_of_walks': num_of_walks,
        'p': p,
        'q': q,
        'max_neighbors': max_neighbors,
        'normalize_by_degree': normalize_by_degree,
        'model_type': model_type,
        'embedding_size': embedding_size,
        'window_size': window_size,
        'embedding_type': embedding_type,
        'random_state': random_state,
        'verbose': verbose
    }

    if weights is None:
        func_args = [source_node, neighbors]
        func_name = 'node2vec_without_weights'
    else:
        raise NotImplemented()

    return register_plugin_function(
        args=func_args,
        plugin_path=_lib,
        function_name=func_name,
        is_elementwise=True,
        kwargs=func_kwargs
    )