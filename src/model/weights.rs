// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Weight loading from `consolidated.safetensors`.
//!
//! Routes tensors from a single safetensors checkpoint file to the three
//! model components (Backbone, FlowMatchingTransformer, Codec) based on
//! their key prefixes.

use std::collections::HashMap;
use std::path::Path;

use half::{bf16, f16};
use safetensors::tensor::{Dtype as SafeDType, TensorView};
use safetensors::SafeTensors;

use crate::config::{QuantizationConfig, VoxtralConfig};
use crate::error::{Result, VoxtralError};
use crate::tensor::{Device, Tensor};

use super::backbone::{Backbone, BackboneConfig};
use super::codec::Codec;
use super::flow_matching::FlowMatchingTransformer;

/// Load all model weights and construct the three main components.
///
/// Reads `consolidated.safetensors` (or multiple shards) from `model_dir`,
/// partitions the tensors by prefix, and instantiates:
///
/// 1. **Backbone** – weights with prefixes `tok_embeddings`, `layers.*`, `norm`, `output`
/// 2. **FlowMatchingTransformer** – weights with prefix `multimodal.acoustic_transformer`
/// 3. **Codec** – weights with prefix `multimodal.audio_tokenizer`
///
/// # Arguments
///
/// * `model_dir` – directory containing `consolidated.safetensors` and `params.json`.
/// * `config` – parsed model configuration.
/// * `device` – target compute device.
///
/// # Returns
///
/// A tuple `(Backbone, FlowMatchingTransformer, Codec)`.
pub fn load_model_weights(
    model_dir: &Path,
    config: &VoxtralConfig,
    device: Device,
) -> Result<(Backbone, FlowMatchingTransformer, Codec)> {
    // Discover safetensors files
    let safetensors_files = find_safetensors_files(model_dir)?;
    if safetensors_files.is_empty() {
        return Err(VoxtralError::ModelLoad(format!(
            "No safetensors files found in {}",
            model_dir.display()
        )));
    }

    tracing::info!(
        "Loading weights from {} safetensors file(s) in {}",
        safetensors_files.len(),
        model_dir.display()
    );

    // Keep weights in BF16 on all backends — libtorch 2.7+ supports BF16 matmul
    // on CPU (Apple Silicon). This matches MLX's native BF16 computation exactly,
    // avoiding precision divergence that causes degenerate semantic codes.

    let all_weights = if let Some(quantization) = config.quantization_config()? {
        tracing::info!(
            "Loading quantized weights: {}-bit, group_size={}",
            quantization.bits,
            quantization.group_size
        );
        load_quantized_weights(&safetensors_files, quantization, device)?
    } else {
        load_standard_weights(&safetensors_files, device)?
    };

    tracing::info!("Loaded {} weight tensors total", all_weights.len());

    // Partition weights by component
    let mut backbone_weights: HashMap<String, Tensor> = HashMap::new();
    let mut flow_weights: HashMap<String, Tensor> = HashMap::new();
    let mut codec_weights: HashMap<String, Tensor> = HashMap::new();

    for (name, tensor) in all_weights {
        if name.starts_with("acoustic_transformer.") {
            flow_weights.insert(name, tensor);
        } else if name.starts_with("audio_tokenizer.") {
            codec_weights.insert(name, tensor);
        } else {
            // Everything else goes to the backbone:
            // mm_audio_embeddings.*, layers.*, norm.*, output.*
            backbone_weights.insert(name, tensor);
        }
    }

    tracing::info!(
        "Weight partitions: backbone={}, flow_matching={}, codec={}",
        backbone_weights.len(),
        flow_weights.len(),
        codec_weights.len(),
    );

    // Build backbone
    let backbone_config = BackboneConfig::from(config);
    let backbone = Backbone::from_weights(&backbone_weights, backbone_config, device);
    tracing::info!(
        "Backbone loaded: {} layers, dim={}, heads={}",
        config.n_layers,
        config.dim,
        config.n_heads,
    );

    // Build flow-matching transformer
    let acoustic_config = config.acoustic_transformer_config();
    let flow_matching =
        FlowMatchingTransformer::from_weights(&flow_weights, acoustic_config, device);
    tracing::info!("Flow-matching transformer loaded");

    // Build codec
    let codec_config = config.audio_tokenizer_config();
    let codec = Codec::from_weights(&codec_weights, codec_config, device);
    tracing::info!("Codec loaded");

    Ok((backbone, flow_matching, codec))
}

fn load_standard_weights(
    paths: &[std::path::PathBuf],
    device: Device,
) -> Result<HashMap<String, Tensor>> {
    let mut all_weights: HashMap<String, Tensor> = HashMap::new();
    for path in paths {
        tracing::debug!("Loading {}", path.display());
        let tensors = Tensor::load_safetensors(path)?;
        for (name, tensor) in tensors {
            all_weights.insert(name, tensor.to_device(device));
        }
    }
    Ok(all_weights)
}

fn load_quantized_weights(
    paths: &[std::path::PathBuf],
    quantization: &QuantizationConfig,
    device: Device,
) -> Result<HashMap<String, Tensor>> {
    let mut all_weights = HashMap::new();

    for path in paths {
        tracing::debug!("Loading quantized {}", path.display());
        let bytes = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes).map_err(|e| {
            VoxtralError::Safetensors(format!("Failed to parse {}: {}", path.display(), e))
        })?;

        let names = tensors.names();
        let mut skip_names = std::collections::HashSet::new();

        for name in &names {
            if let Some(prefix) = name.strip_suffix(".scales") {
                let weight_name = format!("{prefix}.weight");
                let biases_name = format!("{prefix}.biases");
                let qweight_name = format!("{prefix}.qweight");

                if tensors.tensor(&weight_name).is_ok() {
                    let weight = tensors.tensor(&weight_name).map_err(map_safetensor_error)?;
                    let scales = tensors.tensor(name).map_err(map_safetensor_error)?;
                    let biases = tensors.tensor(&biases_name).ok();
                    let dequantized =
                        dequantize_affine_tensor(&weight, &scales, biases.as_ref(), quantization)?
                            .to_device(device);
                    all_weights.insert(weight_name.clone(), dequantized);
                    skip_names.insert(weight_name);
                    skip_names.insert(name.to_string());
                    if biases.is_some() {
                        skip_names.insert(biases_name);
                    }
                } else if tensors.tensor(&qweight_name).is_ok() {
                    return Err(VoxtralError::ModelLoad(format!(
                        "Unsupported quantized tensor format for {} in {}. Found a .qweight tensor suffix instead of .weight. This loader only supports the packed {}.weight + .scales [+ .biases] layout",
                        prefix,
                        path.display(),
                        prefix
                    )));
                }
            }
        }

        for name in names {
            if skip_names.contains(name) {
                continue;
            }
            let view = tensors.tensor(name).map_err(map_safetensor_error)?;
            let tensor = tensor_from_view(&view)?.to_device(device);
            all_weights.insert(name.to_string(), tensor);
        }
    }

    Ok(all_weights)
}

fn map_safetensor_error(e: safetensors::SafeTensorError) -> VoxtralError {
    VoxtralError::Safetensors(e.to_string())
}

fn tensor_from_view(view: &TensorView<'_>) -> Result<Tensor> {
    let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
    let tensor = match view.dtype() {
        SafeDType::BF16 => Tensor::from_slice_bf16(&decode_bf16(view.data())?),
        SafeDType::F16 => Tensor::from_slice_f16(&decode_f16(view.data())?),
        SafeDType::F32 => Tensor::from_slice_f32(&decode_f32(view.data())?),
        SafeDType::I64 => Tensor::from_slice_i64(&decode_i64(view.data())?),
        SafeDType::I32 => Tensor::from_slice_i32(&decode_i32(view.data())?),
        other => {
            return Err(VoxtralError::ModelLoad(format!(
                "Unsupported tensor dtype {:?} for {:?}",
                other, shape
            )))
        }
    };

    Ok(tensor.reshape(&shape))
}

fn dequantize_affine_tensor(
    weight: &TensorView<'_>,
    scales: &TensorView<'_>,
    biases: Option<&TensorView<'_>>,
    quantization: &QuantizationConfig,
) -> Result<Tensor> {
    let weight_shape = weight.shape();
    if weight_shape.len() != 2 {
        return Err(VoxtralError::ModelLoad(format!(
            "Quantized weight must be rank-2, got {:?}",
            weight_shape
        )));
    }

    let scales_shape = scales.shape();
    if scales_shape.len() != 2 {
        return Err(VoxtralError::ModelLoad(format!(
            "Quantization scales must be rank-2, got {:?}",
            scales_shape
        )));
    }

    let out_features = weight_shape[0];
    let packed_in = weight_shape[1];
    let n_groups = scales_shape[1];
    let in_features = n_groups * quantization.group_size;
    let pack_factor = 32 / quantization.bits;
    let expected_packed_in = in_features.div_ceil(pack_factor);

    if packed_in != expected_packed_in {
        return Err(VoxtralError::ModelLoad(format!(
            "Packed weight shape {:?} is incompatible with bits={} and group_size={} (expected packed input dim {})",
            weight_shape,
            quantization.bits,
            quantization.group_size,
            expected_packed_in
        )));
    }

    let packed = decode_u32(weight.data(), weight.dtype())?;
    if scales_shape[0] != out_features {
        return Err(VoxtralError::ModelLoad(format!(
            "Quantization scales shape {:?} does not match weight output dim {}",
            scales_shape, out_features
        )));
    }

    let scale_values = decode_tensor_as_f32(scales)?;
    let bias_values = match biases {
        Some(biases) => {
            let bias_shape = biases.shape();
            if bias_shape != scales_shape {
                return Err(VoxtralError::ModelLoad(format!(
                    "Quantization biases shape {:?} does not match scales shape {:?}",
                    bias_shape, scales_shape
                )));
            }
            decode_tensor_as_f32(biases)?
        }
        None => vec![0.0; scale_values.len()],
    };

    let dequantized = dequantize_affine_packed_values(
        &packed,
        out_features,
        packed_in,
        in_features,
        &scale_values,
        &bias_values,
        n_groups,
        quantization.bits,
        quantization.group_size,
    )?;

    let bf16_values: Vec<bf16> = dequantized.into_iter().map(bf16::from_f32).collect();
    Ok(Tensor::from_slice_bf16(&bf16_values).reshape(&[out_features as i64, in_features as i64]))
}

fn dequantize_affine_packed_values(
    packed: &[u32],
    out_features: usize,
    packed_in: usize,
    in_features: usize,
    scales: &[f32],
    biases: &[f32],
    n_groups: usize,
    bits: usize,
    group_size: usize,
) -> Result<Vec<f32>> {
    if !QuantizationConfig::supports_bits(bits) {
        return Err(VoxtralError::ModelLoad(format!(
            "Unsupported quantization bits {}. Only 4-bit and 6-bit checkpoints are supported",
            bits
        )));
    }
    if scales.len() != out_features * n_groups || biases.len() != out_features * n_groups {
        return Err(VoxtralError::ModelLoad(
            "Invalid quantization scales/biases length".to_string(),
        ));
    }

    let pack_factor = 32 / bits;
    let mask = (1u32 << bits) - 1;
    let mut values = vec![0.0f32; out_features * in_features];

    for out_idx in 0..out_features {
        let packed_row_offset = out_idx * packed_in;
        let quant_row_offset = out_idx * n_groups;
        for in_idx in 0..in_features {
            let group_idx = in_idx / group_size;
            let packed_word = packed[packed_row_offset + in_idx / pack_factor];
            let shift = ((in_idx % pack_factor) * bits) as u32;
            let q = (packed_word >> shift) & mask;
            let scale = scales[quant_row_offset + group_idx];
            let bias = biases[quant_row_offset + group_idx];
            values[out_idx * in_features + in_idx] = q as f32 * scale + bias;
        }
    }

    Ok(values)
}

fn decode_tensor_as_f32(view: &TensorView<'_>) -> Result<Vec<f32>> {
    match view.dtype() {
        SafeDType::F32 => decode_f32(view.data()),
        SafeDType::F16 => Ok(decode_f16(view.data())?
            .into_iter()
            .map(f32::from)
            .collect()),
        SafeDType::BF16 => Ok(decode_bf16(view.data())?
            .into_iter()
            .map(f32::from)
            .collect()),
        other => Err(VoxtralError::ModelLoad(format!(
            "Unsupported floating tensor dtype {:?}",
            other
        ))),
    }
}

fn decode_bf16(data: &[u8]) -> Result<Vec<bf16>> {
    decode_u16_like(data, bf16::from_bits)
}

fn decode_f16(data: &[u8]) -> Result<Vec<f16>> {
    decode_u16_like(data, f16::from_bits)
}

fn decode_u16_like<T>(data: &[u8], map: impl Fn(u16) -> T) -> Result<Vec<T>> {
    let chunks = data.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        return Err(VoxtralError::ModelLoad(
            "Tensor byte length is not aligned to 2 bytes".to_string(),
        ));
    }
    Ok(chunks
        .map(|chunk| map(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn decode_f32(data: &[u8]) -> Result<Vec<f32>> {
    let chunks = data.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(VoxtralError::ModelLoad(
            "Tensor byte length is not aligned to 4 bytes".to_string(),
        ));
    }
    Ok(chunks
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn decode_i32(data: &[u8]) -> Result<Vec<i32>> {
    let chunks = data.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(VoxtralError::ModelLoad(
            "Tensor byte length is not aligned to 4 bytes".to_string(),
        ));
    }
    Ok(chunks
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn decode_i64(data: &[u8]) -> Result<Vec<i64>> {
    let chunks = data.chunks_exact(8);
    if !chunks.remainder().is_empty() {
        return Err(VoxtralError::ModelLoad(
            "Tensor byte length is not aligned to 8 bytes".to_string(),
        ));
    }
    Ok(chunks
        .map(|chunk| {
            i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

fn decode_u32(data: &[u8], dtype: SafeDType) -> Result<Vec<u32>> {
    match dtype {
        SafeDType::U32 => {
            let chunks = data.chunks_exact(4);
            if !chunks.remainder().is_empty() {
                return Err(VoxtralError::ModelLoad(
                    "Quantized tensor byte length is not aligned to 4 bytes".to_string(),
                ));
            }
            Ok(chunks
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        SafeDType::I32 => Err(VoxtralError::ModelLoad(
            "Found signed I32 storage for a packed quantized tensor. Packed weights must use unsigned U32 to prevent sign extension from corrupting unpacked bit-level values".to_string(),
        )),
        other => Err(VoxtralError::ModelLoad(format!(
            "Unsupported packed quantized tensor dtype {:?}",
            other
        ))),
    }
}

/// Find all safetensors files in the model directory.
///
/// Looks for:
/// 1. `consolidated.safetensors` (single-file checkpoint)
/// 2. `model-00001-of-*.safetensors` (sharded checkpoint)
/// 3. Any `.safetensors` file (fallback)
fn find_safetensors_files(model_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    // Try consolidated first
    let consolidated = model_dir.join("consolidated.safetensors");
    if consolidated.exists() {
        return Ok(vec![consolidated]);
    }

    // Try sharded format
    let mut shards: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    shards.push(path);
                }
            }
        }
    }

    if !shards.is_empty() {
        // Sort shards to ensure consistent ordering (model-00001, model-00002, ...)
        shards.sort();
        return Ok(shards);
    }

    // Fallback: any safetensors file
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                files.push(path);
            }
        }
    }

    if files.is_empty() {
        return Err(VoxtralError::ModelLoad(format!(
            "No safetensors files found in {}. Expected consolidated.safetensors or sharded model-*.safetensors",
            model_dir.display()
        )));
    }

    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::dequantize_affine_packed_values;

    fn pack_row(values: &[u32], bits: usize) -> Vec<u32> {
        let pack_factor = 32 / bits;
        let packed_len = values.len().div_ceil(pack_factor);
        let mut packed = vec![0u32; packed_len];
        for (idx, &value) in values.iter().enumerate() {
            // Least-significant bits first, packed left-to-right within each u32.
            let word_idx = idx / pack_factor;
            let shift = ((idx % pack_factor) * bits) as u32;
            packed[word_idx] |= value << shift;
        }
        packed
    }

    #[test]
    fn dequantizes_4bit_affine_weights() {
        let row0 = pack_row(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        let row1 = pack_row(&[8, 7, 6, 5, 4, 3, 2, 1], 4);
        let packed = [row0, row1].concat();
        let scales = vec![1.0, 2.0, 0.5, 1.5];
        let biases = vec![0.0, 10.0, -1.0, 1.0];

        let out =
            dequantize_affine_packed_values(&packed, 2, 1, 8, &scales, &biases, 2, 4, 4).unwrap();

        assert_eq!(
            out,
            vec![
                1.0, 2.0, 3.0, 4.0, 20.0, 22.0, 24.0, 26.0, 3.0, 2.5, 2.0, 1.5, 7.0, 5.5, 4.0, 2.5
            ]
        );
    }

    #[test]
    fn dequantizes_6bit_affine_weights() {
        let packed = pack_row(&[1, 2, 3, 4, 5, 6], 6);
        let scales = vec![0.5, 2.0];
        let biases = vec![1.0, -3.0];

        let out =
            dequantize_affine_packed_values(&packed, 1, 2, 6, &scales, &biases, 2, 6, 3).unwrap();

        assert_eq!(out, vec![1.5, 2.0, 2.5, 5.0, 7.0, 9.0]);
    }
}
