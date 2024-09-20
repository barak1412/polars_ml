import polars as pl
from polars.plugins import register_plugin_function
from polars_ml import lib


_SUPPORTED_MODEL_TYPES = ['skipgram', 'cbow']
_SUPPORTED_EMBEDDING_TYPES = ['central', 'contextual']


def _validate_node2vec_params(walk_length: int,
                              num_of_walks: int,
                              p: float,
                              q: float,
                              max_neighbors: int,
                              model_type: str,
                              embedding_size: int,
                              window_size: int,
                              embedding_type: str):
    if walk_length <= 0:
        raise Exception(f'walk_length must be greater than zero, {walk_length} was given.')
    if num_of_walks <= 0:
        raise Exception(f'num_of_walks must be greater than zero, {num_of_walks} was given.')
    if p <= 0.0:
        raise Exception(f'p must be greater than zero, {p} was given.')
    if q <= 0.0:
        raise Exception(f'p must be greater than zero, {q} was given.')
    if max_neighbors is not None and max_neighbors <= 0:
        raise Exception(f'max_neighbors can be None or greater than zero, {max_neighbors} was given.')
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise Exception(f'model_type must be inside {_SUPPORTED_MODEL_TYPES}, {model_type} was given.')
    if embedding_size <= 0:
        raise Exception(f'embedding_size must be greater than zero, {embedding_size} was given.')
    if window_size <= 0:
        raise Exception(f'window_size must be greater than zero, {window_size} was given.')
    if embedding_type not in _SUPPORTED_EMBEDDING_TYPES:
        raise Exception(f'embedding_type must be inside {_SUPPORTED_EMBEDDING_TYPES}, {embedding_type} was given.')


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

    # validate params in range
    _validate_node2vec_params(walk_length, num_of_walks, p, q, max_neighbors, model_type,
                              embedding_size, window_size, embedding_type)

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
        func_args = [source_node, neighbors, weights]
        func_name = 'node2vec_with_weights'

    return register_plugin_function(
        args=func_args,
        plugin_path=lib,
        function_name=func_name,
        is_elementwise=False,
        kwargs=func_kwargs
    )