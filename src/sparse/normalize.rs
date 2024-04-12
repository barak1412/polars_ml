#![allow(clippy::unused_unit)]
use polars::prelude::*;
use std::collections::HashMap;
use num_traits::ToPrimitive;
use polars::chunked_array::builder::list::ListPrimitiveChunkedBuilder;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use crate::sparse::{DIM, INDICES, VALUES};


#[derive(Deserialize)]
struct NormalizeKwargs {
    how: String,
    p: f64,
}

fn float_sparse_vector(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(Field::new(&field.name().clone(),
                          DataType::Struct(vec![Field::new(DIM, DataType::UInt32),
                                                Field::new(INDICES, DataType::List(Box::new(DataType::UInt32))),
                                                Field::new(VALUES, DataType::List(Box::new(DataType::Float64)))]
                          )))

}

#[polars_expr(output_type_func=float_sparse_vector)]
fn normalize(inputs: &[Series], kwargs: NormalizeKwargs) -> PolarsResult<Series> {
    let how = kwargs.how.as_str();
    let p = kwargs.p;

    if p < 1.0 {
        return polars_bail!(ComputeError: "p must be greater or equals to 1.")
    }
    let struct_ = inputs[0].struct_()?;
    let fields = struct_.fields();
    let indices_ca = fields[1].list()?;
    let values_ca = fields[2].list()?;

    match values_ca.inner_dtype() {
        DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64 => {
            match how {
                "vertical" => normalize_vertical(struct_.name(), &fields, &indices_ca, &values_ca, p),
                "horizontal" => polars_bail!(ComputeError: "Horizontal normalization is currently not implemented."),
                how => polars_bail!(ComputeError: "'{}' is unsupported.", how)
            }
        },
        dtype => {
            polars_bail!(InvalidOperation:format!("dtype {dtype} not \
            supported for normalize, expected Int32, Int64, Float32, Float64."))
        }
    }
}
fn normalize_vertical(struct_name: &str, fields: &[Series], indices_ca: &ListChunked, values_ca: &ListChunked, p: f64) -> PolarsResult<Series> {
    let indices_norms = sparse_summarize(indices_ca, values_ca, p);

    unsafe {
        let mut new_values_builder: ListPrimitiveChunkedBuilder<Float64Type> = ListPrimitiveChunkedBuilder::new(fields[2].name(), values_ca.len(), values_ca.len(), DataType::Float64);
        indices_ca.amortized_iter()
            .zip(values_ca.amortized_iter())
            .for_each(|(indices_series, values_series)| {
                match (indices_series, values_series) {
                    (Some(indices_series), Some(values_series)) => {
                        let indices_ca = indices_series.as_ref().idx().unwrap();

                        let out = match values_ca.inner_dtype() {
                            DataType::Int32 => {
                                let values_ca = values_series.as_ref().i32().unwrap();
                                normalize_ca(&indices_ca, &values_ca, &indices_norms)
                            },
                            DataType::Int64 => {
                                let values_ca = values_series.as_ref().i64().unwrap();
                                normalize_ca(&indices_ca, &values_ca, &indices_norms)
                            },
                            DataType::Float32 => {
                                let values_ca = values_series.as_ref().f32().unwrap();
                                normalize_ca(&indices_ca, &values_ca, &indices_norms)
                            },
                            DataType::Float64 => {
                                let values_ca = values_series.as_ref().f64().unwrap();
                                normalize_ca(&indices_ca, &values_ca, &indices_norms)
                            }
                            _ => unreachable!()
                        };
                        new_values_builder.append_series(&out.into_series());
                    },
                    _ => new_values_builder.append_null()
                };
            });

        // create result sparse vector column
        let result = StructChunked::new(struct_name,
                                        &[fields[0].clone(), fields[1].clone(), new_values_builder.finish().into_series()]).unwrap();
        Ok(result.into_series())
    }
}

fn normalize_ca<T>(indices_ca: &IdxCa, values_ca: &ChunkedArray<T>, indices_norms: &HashMap<IdxSize, f64>) -> Float64Chunked
    where
        T: PolarsNumericType,
        T::Native: ToPrimitive,
{
    let out :Float64Chunked = indices_ca.iter().zip(values_ca).map(|(idx, value)| {
        let idx = idx.unwrap();
        let value = T::Native::to_f64(&value.unwrap()).unwrap();
        let norm_value = indices_norms.get(&idx).unwrap();
        Some(value / norm_value)
    }).collect_ca("");

    out
}

fn sparse_summarize(indices_list_ca: &ListChunked, values_list_ca: &ListChunked, p: f64) -> HashMap<IdxSize, f64> {
    let mut indices_norms:HashMap<IdxSize, f64> = HashMap::new();
    unsafe {
        indices_list_ca.amortized_iter()
            .zip(values_list_ca.amortized_iter())
            .for_each(|(indices_series, values_series)| {
                match (indices_series, values_series) {
                    (Some(indices_series), Some(values_series)) => {
                        let indices_ca = indices_series.as_ref().idx().unwrap();
                        match values_list_ca.inner_dtype() {
                            DataType::Int32 => {
                                let values_ca = values_series.as_ref().i32().unwrap();
                                add_single_list_to_map(indices_ca, values_ca, &mut indices_norms, p);
                            },
                            DataType::Int64 => {
                                let values_ca = values_series.as_ref().i64().unwrap();
                                add_single_list_to_map(indices_ca, values_ca, &mut indices_norms, p);
                            },
                            DataType::Float32 => {
                                let values_ca = values_series.as_ref().f32().unwrap();
                                add_single_list_to_map(indices_ca, values_ca, &mut indices_norms, p);
                            },
                            DataType::Float64 => {
                                let values_ca = values_series.as_ref().f64().unwrap();
                                add_single_list_to_map(indices_ca, values_ca, &mut indices_norms, p);
                            }
                            _ => unreachable!()
                        };
                    },
                    _ => ()
                };
            });
        let exp = 1.0 / p;
        indices_norms.into_iter().map(|(k, v)| (k, v.powf(exp))).collect()
    }
}

fn add_single_list_to_map<T>(indices_ca: &IdxCa, values_ca: &ChunkedArray<T>,
                             indices_norms: &mut HashMap<IdxSize, f64>, p: f64) -> ()
    where
        T: PolarsNumericType,
        T::Native: ToPrimitive,
{
    indices_ca.iter().zip(values_ca.iter())
        .for_each(|(idx, value)| {
            let idx = idx.unwrap();
            let mut value = T::Native::to_f64(&value.unwrap()).unwrap();
            value = value.abs().powf(p);
            let old_idx_value = indices_norms.get(&idx).unwrap_or(&0.0);
            let new_idx_value = old_idx_value + value;
            indices_norms.insert(idx, new_idx_value);
        });
}
