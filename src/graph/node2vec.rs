#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use grape::graph::GraphBuilder;
use grape::graph::Graph;
use grape::graph::walks_parameters::{WalksParameters, SingleWalkParameters, WalkWeights};
use grape::cpu_models::{Node2Vec, Node2VecModels, IdentifyWalkTransformer, GraphEmbedder};


#[derive(Deserialize)]
pub struct Node2VecKwargs {
    // graph
    pub is_directed: bool,

    // walks
    pub walk_length: u64,
    pub num_of_walks: u32,
    pub p: f32,
    pub q: f32,

    // model
    pub model_type: String,
    pub embedding_size: u32,
    pub window_size: u32,
    pub embedding_type: String,

    // general
    pub random_state: u32,
    pub verbose: bool
}

//#[polars_expr(output_type=Int64)]
pub fn node2vec_without_weights(inputs: &[Series], node2vec_kwargs: Node2VecKwargs) -> PolarsResult<Series> {
    let source_nodes_ca = inputs[0].str()?;
    let neighbors_lst_ca = inputs[1].list()?;

    // build walks params
    let walks_params = build_walks_params(&node2vec_kwargs);

    // build model
    let node2vec = build_node2vec(&node2vec_kwargs, walks_params);

    // build the graph
    let graph = build_graph_without_weights(source_nodes_ca, neighbors_lst_ca, node2vec_kwargs.is_directed);

    // run node2vec walks and model
    let mut embedding_1d_central = vec![0.0f32; node2vec_kwargs.embedding_size as usize * graph.get_number_of_nodes() as usize];
    let mut embedding_1d_contextual = vec![0.0f32; node2vec_kwargs.embedding_size as usize * graph.get_number_of_nodes() as usize];
    let mut embedding_buffer = vec![embedding_1d_central.as_mut_slice(), embedding_1d_contextual.as_mut_slice()];
    node2vec.fit_transform(&graph, embedding_buffer.as_mut_slice());

    // transform to series
    let embedding = match node2vec_kwargs.embedding_type.as_str() {
        "central" => &embedding_buffer[0],
        "contextual" => &embedding_buffer[1],
        _ => !unreachable!()
    };
    let embedding_series = transform_embedding_to_series(&source_nodes_ca,
                        &graph, embedding, node2vec_kwargs.embedding_size);
    Ok(embedding_series)
}

fn transform_embedding_to_series(source_nodes_ca: &StringChunked,
                                 graph: &Graph, embedding: &[f32],
                                 embedding_size: u32) -> Series {
    let mut embedding_builder: ListPrimitiveChunkedBuilder<Float32Type> = ListPrimitiveChunkedBuilder::new("", source_nodes_ca.len(), source_nodes_ca.len(), DataType::Float32);
    source_nodes_ca.for_each(|source_node|{
        match source_node {
            Some(source_node) => {
                let node_index = graph.get_node_id_from_node_name(source_node).unwrap();
                let start_index = (node_index * embedding_size) as usize;
                let end_index = ((node_index + 1 ) * embedding_size) as usize;
                let node_embedding = &embedding[start_index..end_index];
                let node_embedding_series = Series::new("", node_embedding);
                embedding_builder.append_series(&node_embedding_series).unwrap();
            },
            None => embedding_builder.append_null()
        }
    });

    let result = embedding_builder.finish();
    result.into_series()
}
fn build_walks_params(node2vec_kwargs: &Node2VecKwargs) -> WalksParameters {
    let output_params = WalksParameters {
        single_walk_parameters: SingleWalkParameters {
            walk_length: node2vec_kwargs.walk_length,
            weights: WalkWeights {
                return_weight: 1.0 / node2vec_kwargs.p,
                explore_weight: 1.0 / node2vec_kwargs.q,
                change_edge_type_weight: 1.0,
                change_node_type_weight: 1.0
            },
            max_neighbours: None,
            normalize_by_degree: false
        },
        iterations: node2vec_kwargs.num_of_walks,
        random_state: node2vec_kwargs.random_state
    };

    output_params
}

fn build_node2vec(node2vec_kwargs: &Node2VecKwargs, walks_params: WalksParameters) -> Node2Vec<IdentifyWalkTransformer> {
    let model_type = match node2vec_kwargs.model_type.as_str() {
        "cbow" => Node2VecModels::CBOW,
        "skipgram" => Node2VecModels::SkipGram,
        _ => !unreachable!()
    };

    let node2vec = Node2Vec::new(
        model_type,
        IdentifyWalkTransformer::default(),
        Some(node2vec_kwargs.embedding_size as usize),
        Some(walks_params),
        Some(node2vec_kwargs.window_size as usize),
        None, None, None, None, None, None, None, None, None, None, None,
        Some(node2vec_kwargs.verbose)
    ).unwrap();

    node2vec
}
fn build_graph_without_weights(source_nodes_ca: &StringChunked, neighbors_lst_ca: &ListChunked,
                is_directed: bool) -> Graph {
    let mut graph_builder = GraphBuilder::new(None, Some(is_directed));
    graph_builder.set_default_weight(1.0);

    // add edges without duplicates
    unsafe {
        source_nodes_ca.iter()
            .zip(neighbors_lst_ca.amortized_iter())
            .for_each(|(source_node_option, neighbors_lst_series_option)| {
                if let (Some(source_node), Some(neighbors_lst_series)) = (source_node_option, neighbors_lst_series_option) {
                    let neighbors_lst_series = neighbors_lst_series.as_ref().str().unwrap();
                    neighbors_lst_series.for_each(|neighbor_node|{
                        if let Some(neighbor_node) = neighbor_node {
                            graph_builder.add_edge(source_node.to_string(), neighbor_node.to_string(), None, None);
                        }
                    })
                }
            });
    }

    let graph: Graph = graph_builder.build().unwrap();
    graph
}