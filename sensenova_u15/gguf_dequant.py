# Adapted from city96/ComfyUI-GGUF (Apache-2.0).
# The implementation is kept local so the SenseNova loader does not depend on
# another custom-node package being installed or imported first.

import gguf
import torch


TORCH_COMPATIBLE_QTYPES = (
    None,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
)


def is_torch_compatible(tensor):
    return tensor is None or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES


def is_quantized(tensor):
    return not is_torch_compatible(tensor)


def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, "tensor_type", None)
    output_shape = getattr(tensor, "tensor_shape", tensor.shape)
    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype=dtype)
    if qtype not in DEQUANTIZE_FUNCTIONS:
        name = getattr(qtype, "name", repr(qtype))
        raise ValueError(f"unsupported SenseNova GGUF tensor type: {name}")
    dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
    return dequantize(tensor.data, qtype, output_shape, dtype=dequant_dtype).to(dtype=dtype)


def dequantize(data, qtype, output_shape, dtype=None):
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    blocks = rows.reshape((rows.numel() // type_size, type_size))
    values = DEQUANTIZE_FUNCTIONS[qtype](blocks, block_size, type_size, dtype)
    return values.reshape(output_shape)


def _to_uint32(value):
    value = value.view(torch.uint8).to(torch.int32)
    return (value[:, 0] | value[:, 1] << 8 | value[:, 2] << 16 | value[:, 3] << 24).unsqueeze(1)


def _split_block_dims(blocks, *sizes):
    return torch.split(blocks, list(sizes) + [blocks.shape[1] - sum(sizes)], dim=1)


def _dequantize_bf16(blocks, _block_size, _type_size, _dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


def _dequantize_q8_0(blocks, _block_size, _type_size, dtype=None):
    scale, values = _split_block_dims(blocks, 2)
    scale = scale.view(torch.float16).to(dtype=dtype)
    return scale * values.view(torch.int8)


QK_K = 256
K_SCALE_SIZE = 12


def _get_scale_min(scales):
    blocks = scales.shape[0]
    scales = scales.view(torch.uint8).reshape((blocks, 3, 4))
    low_scale, low_min, mixed = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    scale = torch.cat([low_scale & 0x3F, (mixed & 0x0F) | ((low_scale >> 2) & 0x30)], dim=-1)
    minimum = torch.cat([low_min & 0x3F, (mixed >> 4) | ((low_min >> 2) & 0x30)], dim=-1)
    return scale.reshape((blocks, 8)), minimum.reshape((blocks, 8))


def _dequantize_q6_k(blocks, _block_size, _type_size, dtype=None):
    block_count = blocks.shape[0]
    low, high, scales, base = _split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype=dtype)
    base = base.view(torch.float16).to(dtype=dtype)
    scale = (base * scales).reshape((block_count, QK_K // 16, 1))
    low = low.reshape((block_count, -1, 1, 64)) >> torch.tensor(
        [0, 4], device=scale.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    low = (low & 0x0F).reshape((block_count, -1, 32))
    high = high.reshape((block_count, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=scale.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    high = (high & 0x03).reshape((block_count, -1, 32))
    values = (low | (high << 4)).to(torch.int8) - 32
    return (scale * values.reshape((block_count, QK_K // 16, -1))).reshape((block_count, QK_K))


def _dequantize_q5_k(blocks, _block_size, _type_size, dtype=None):
    block_count = blocks.shape[0]
    base, base_min, scales, high, low = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    base = base.view(torch.float16).to(dtype=dtype)
    base_min = base_min.view(torch.float16).to(dtype=dtype)
    scale, minimum = _get_scale_min(scales)
    scale = (base * scale).reshape((block_count, -1, 1))
    minimum = (base_min * minimum).reshape((block_count, -1, 1))
    low = low.reshape((block_count, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=scale.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    high = high.reshape((block_count, -1, 1, 32)) >> torch.arange(
        8, device=scale.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    values = (low & 0x0F).reshape((block_count, -1, 32)) | (
        (high & 0x01).reshape((block_count, -1, 32)) << 4
    )
    return (scale * values - minimum).reshape((block_count, QK_K))


def _dequantize_q4_k(blocks, _block_size, _type_size, dtype=None):
    block_count = blocks.shape[0]
    base, base_min, scales, values = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    base = base.view(torch.float16).to(dtype=dtype)
    base_min = base_min.view(torch.float16).to(dtype=dtype)
    scale, minimum = _get_scale_min(scales)
    scale = (base * scale).reshape((block_count, -1, 1))
    minimum = (base_min * minimum).reshape((block_count, -1, 1))
    values = values.reshape((block_count, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=scale.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    values = (values & 0x0F).reshape((block_count, -1, 32))
    return (scale * values - minimum).reshape((block_count, QK_K))


def _dequantize_q3_k(blocks, _block_size, _type_size, dtype=None):
    block_count = blocks.shape[0]
    high_mask, low, scales, base = _split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    base = base.view(torch.float16).to(dtype=dtype)
    low_scales = scales[:, :8].reshape((block_count, 1, 8)) >> torch.tensor(
        [0, 4], device=base.device, dtype=torch.uint8
    ).reshape((1, 2, 1))
    high_scales = scales[:, 8:].reshape((block_count, 1, 4)) >> torch.tensor(
        [0, 2, 4, 6], device=base.device, dtype=torch.uint8
    ).reshape((1, 4, 1))
    scales = (low_scales.reshape((block_count, 16)) & 0x0F) | (
        (high_scales.reshape((block_count, 16)) & 0x03) << 4
    )
    scales = scales.to(torch.int8) - 32
    scale = (base * scales).reshape((block_count, 16, 1))
    low = low.reshape((block_count, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=base.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    high = high_mask.reshape((block_count, -1, 1, 32)) >> torch.arange(
        8, device=base.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    low = low.reshape((block_count, 16, QK_K // 16)) & 3
    high = (high.reshape((block_count, 16, QK_K // 16)) & 1) ^ 1
    values = low.to(torch.int8) - (high << 2).to(torch.int8)
    return (scale * values).reshape((block_count, QK_K))


def _dequantize_q2_k(blocks, _block_size, _type_size, dtype=None):
    block_count = blocks.shape[0]
    scales, values, base, base_min = _split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    base = base.view(torch.float16).to(dtype=dtype)
    base_min = base_min.view(torch.float16).to(dtype=dtype)
    scale = (base * (scales & 0x0F)).reshape((block_count, QK_K // 16, 1))
    minimum = (base_min * (scales >> 4)).reshape((block_count, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=base.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    values = (values.reshape((block_count, -1, 1, 32)) >> shift) & 3
    values = values.reshape((block_count, QK_K // 16, 16))
    return (scale * values - minimum).reshape((block_count, -1))


DEQUANTIZE_FUNCTIONS = {
    gguf.GGMLQuantizationType.BF16: _dequantize_bf16,
    gguf.GGMLQuantizationType.Q8_0: _dequantize_q8_0,
    gguf.GGMLQuantizationType.Q6_K: _dequantize_q6_k,
    gguf.GGMLQuantizationType.Q5_K: _dequantize_q5_k,
    gguf.GGMLQuantizationType.Q4_K: _dequantize_q4_k,
    gguf.GGMLQuantizationType.Q3_K: _dequantize_q3_k,
    gguf.GGMLQuantizationType.Q2_K: _dequantize_q2_k,
}

