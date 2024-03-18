#![allow(clippy::unused_unit)]
use polars::prelude::*;
use std::collections::HashMap;
use polars::chunked_array::builder::list::ListPrimitiveChunkedBuilder;
use pyo3_polars::derive::polars_expr;
use crate::sparse::{DIM, INDICES, VALUES};

fn float_sparse_vector(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(Field::new(&field.name().clone(),
                          DataType::Struct(vec![Field::new(DIM, DataType::UInt32),
                                                Field::new(INDICES, DataType::List(Box::new(DataType::UInt32))),
                                                Field::new(VALUES, DataType::List(Box::new(DataType::Float64)))]
                          )))

}

#[polars_expr(output_type_func=float_sparse_vector)]
fn normalize(inputs: &[Series]) -> PolarsResult<Series> {
    let struct_ = inputs[0].struct_()?;
    let fields = struct_.fields();
    let indices_ca = fields[1].list()?;
    let values_ca = fields[2].list()?;
    let indices_norms = sparse_summarize(indices_ca, values_ca);

    unsafe {
        let mut new_values_builder: ListPrimitiveChunkedBuilder<Float64Type> = ListPrimitiveChunkedBuilder::new(fields[2].name().clone(), values_ca.len(), values_ca.len(), DataType::Float64);
        indices_ca.amortized_iter()
            .zip(values_ca.amortized_iter())
            .for_each(|(indices_series, values_series)| {
                match (indices_series, values_series) {
                    (Some(indices_series), Some(values_series)) => {
                        let indices_ca = indices_series.as_ref().idx().unwrap();
                        let values_ca = values_series.as_ref().f64().unwrap();
                        let out :Float64Chunked = indices_ca.iter().zip(values_ca).map(|(idx, value)| {
                            let idx = idx.unwrap();
                            let value = value.unwrap();
                            let norm_value = indices_norms.get(&idx).unwrap();
                            Some(value / norm_value)
                        }).collect_ca("");
                        new_values_builder.append_series(&out.into_series());
                    },
                    _ => new_values_builder.append_null()
                };
            });

        // create result sparse vector column
        let result = StructChunked::new(struct_.name(),
                                        &[fields[0].clone(), fields[1].clone(), new_values_builder.finish().into_series()]).unwrap();
        Ok(result.into_series())
    }
}
fn sparse_summarize(indices_list_ca: &ListChunked, values_list_ca: &ListChunked) -> HashMap<IdxSize, f64> {
    let mut indices_norms:HashMap<IdxSize, f64> = HashMap::new();
    unsafe {
        indices_list_ca.amortized_iter()
            .zip(values_list_ca.amortized_iter())
            .for_each(|(indices_series, values_series)| {
                match (indices_series, values_series) {
                    (Some(indices_series), Some(values_series)) => {
                        indices_series.as_ref().idx().unwrap().iter().zip(values_series.as_ref().f64().unwrap().iter())
                            .for_each(|(idx, value)| {
                                let idx = idx.unwrap();
                                let value = value.unwrap();
                                let old_idx_value = indices_norms.get(&idx).unwrap_or(&0.0);
                                let new_idx_value = old_idx_value + value;
                                indices_norms.insert(idx, new_idx_value);
                            });
                    },
                    _ => ()
                };
            });
        indices_norms
    }
}