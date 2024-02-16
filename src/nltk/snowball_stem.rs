#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rust_stemmers::{Algorithm, Stemmer};
use std::fmt::Write;
use serde::Deserialize;

#[derive(Deserialize)]
struct SnowballStemKwargs {
    language: String,
}

#[polars_expr(output_type=String)]
fn snowball_stem(inputs: &[Series], kwargs: SnowballStemKwargs) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let en_stemmer = {
        match kwargs.language.as_str() {
            "english" => Stemmer::create(Algorithm::English),
            "danish" => Stemmer::create(Algorithm::Danish),
            "arabic" => Stemmer::create(Algorithm::Arabic),
            "dutch" => Stemmer::create(Algorithm::Dutch),
            "finnish" => Stemmer::create(Algorithm::Finnish),
            "french" => Stemmer::create(Algorithm::French),
            "german" => Stemmer::create(Algorithm::German),
            "greek" => Stemmer::create(Algorithm::Greek),
            "hungarian" => Stemmer::create(Algorithm::Hungarian),
            "italian" => Stemmer::create(Algorithm::Italian),
            "norwegian" => Stemmer::create(Algorithm::Norwegian),
            "portuguese" => Stemmer::create(Algorithm::Portuguese),
            "romanian" => Stemmer::create(Algorithm::Romanian),
            "russian" => Stemmer::create(Algorithm::Russian),
            "spanish" => Stemmer::create(Algorithm::Spanish),
            "swedish" => Stemmer::create(Algorithm::Swedish),
            "tamil" => Stemmer::create(Algorithm::Tamil),
            "turkish" => Stemmer::create(Algorithm::Turkish),
            language => polars_bail!(ComputeError: "Language '{}' unsuported for snowball stemming.", language),
        }
    };
    let out: StringChunked = ca.apply_to_buffer(|value: &str, output: &mut String| {
        std::write!(output, "{}", en_stemmer.stem(value)).unwrap()
    });
    Ok(out.into_series())
}