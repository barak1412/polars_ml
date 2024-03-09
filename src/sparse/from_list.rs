#![allow(clippy::unused_unit)]
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::export::num::Signed;
use polars::prelude::*;
use num_traits::Zero;
use polars::chunked_array::builder::list::ListPrimitiveChunkedBuilder;
use polars::chunked_array::builder::AnonymousListBuilder;
use super::{DIM, INDICES, VALUES};


fn sparse_vector(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.data_type() {
        DataType::List(element) => {
            Ok(Field::new(&field.name().clone(),
                          DataType::Struct(vec![Field::new(DIM, DataType::UInt32),
                                                Field::new(INDICES, DataType::List(Box::new(DataType::UInt32))),
                                                Field::new(VALUES, DataType::List(Box::new(*element.clone())))]
                          )))
        },
        _ => unreachable!(),
    }
}

#[polars_expr(output_type_func=sparse_vector)]
fn from_list(inputs: &[Series]) -> PolarsResult<Series> {
    let lst_ca = inputs[0].list()?;
    match lst_ca.inner_dtype() {
        DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64 => impl_from_list(lst_ca),
        dtype => {
            polars_bail!(InvalidOperation:format!("dtype {dtype} not \
            supported for abs_numeric, expected Int32, Int64, Float32, Float64."))
        }
    }
}
fn impl_from_list(series_lst_ca: &ListChunked) -> PolarsResult<Series>{
    // holders for the dimension, indices list and values list
    let mut indices_builder:ListPrimitiveChunkedBuilder<UInt32Type> = ListPrimitiveChunkedBuilder::new(INDICES, series_lst_ca.len(), series_lst_ca.len(), DataType::UInt32);
    let mut values_lst: Vec<Option<Series>> = Vec::with_capacity(series_lst_ca.len());
    let mut dim_vec: Vec<Option<u32>> = Vec::with_capacity(series_lst_ca.len());
    for elements in series_lst_ca {
        match elements {
            Some(elements) => {
                let elements = &elements;
                dim_vec.push(Some(elements.len() as IdxSize));
                let (out_indices, out_values) = match series_lst_ca.inner_dtype() {
                    DataType::Int32 => {
                        let (out_indices, out_values) = impl_from_single_list_to_sparse(elements.i32().unwrap());
                        (out_indices.into_series(), out_values.into_series())
                    },
                    DataType::Int64 => {
                        let (out_indices, out_values) = impl_from_single_list_to_sparse(elements.i64().unwrap());
                        (out_indices.into_series(), out_values.into_series())
                    },
                    DataType::Float32 => {
                        let (out_indices, out_values) = impl_from_single_list_to_sparse(elements.f32().unwrap());
                        (out_indices.into_series(), out_values.into_series())
                    },
                    DataType::Float64 => {
                        let (out_indices, out_values) = impl_from_single_list_to_sparse(elements.f64().unwrap());
                        (out_indices.into_series(), out_values.into_series())
                    },
                    _ => unreachable!()
                };
                indices_builder.append_series(&out_indices).map_err(|err| {
                    return err
                })?;
                values_lst.push(Some(out_values));
            },
            None => {
                dim_vec.push(None);
                indices_builder.append_null();
                values_lst.push(None);
            }
        };
    }
    let dim_ca:IdxCa = dim_vec.into_iter().collect_ca(DIM);
    let dim_series= dim_ca.into_series();
    let indices_series = indices_builder.finish().into_series();
    let mut values_builder = AnonymousListBuilder::new(VALUES, series_lst_ca.len(), Some(series_lst_ca.inner_dtype().clone()));
    for value in &values_lst {
        match value {
            Some(value)=>{
                values_builder.append_series(value).map_err(|err|{
                    return err;
                })?;
            },
            None => {values_builder.append_null();}
        };
    }
    let values_series = values_builder.finish().into_series();
    let out = StructChunked::new(series_lst_ca.name(),
                                 &[dim_series, indices_series, values_series]).unwrap();

    Ok(out.into_series())
}

#[inline]
fn impl_from_single_list_to_sparse<T>(ca: &ChunkedArray<T>) -> (ChunkedArray<UInt32Type>, ChunkedArray<T>)
    where
        T: PolarsNumericType,
        T::Native: Signed,
{
    let mut indices_out: Vec<Option<u32>> = Vec::with_capacity(ca.len());
    let mut values_out: Vec<Option<T::Native>> = Vec::with_capacity(ca.len());
    for (idx, element) in ca.into_iter().enumerate() {
        match element {
            Some(val) => {
                if !Zero::is_zero(&val){
                    indices_out.push(Some(idx as IdxSize));
                    values_out.push(Some(val));
                }
            },
            None => ()
        }
    }

    let indices_out: IdxCa = indices_out.into_iter().collect_ca("");
    let values_out: ChunkedArray<T> = values_out.into_iter().collect_ca("");

    (indices_out, values_out)
}