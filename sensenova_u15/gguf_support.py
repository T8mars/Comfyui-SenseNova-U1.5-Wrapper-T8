# GGUF tensor loading and on-demand dequantization are adapted from
# city96/ComfyUI-GGUF (Apache-2.0). SenseNova-specific validation, model
# construction, attachments, and lifecycle handling are implemented here.

import collections
import hashlib
import logging
import math
import warnings
from pathlib import Path

import gguf
import torch

import comfy.lora
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils

from .gguf_dequant import dequantize_tensor, is_quantized, is_torch_compatible
from .loader import MODEL_REPO, MODEL_REVISION, _checkpoint_contract
from .model_config import SenseNovaModelConfig


GGUF_SOURCE_REPO = "realrebelai/SenseNova-U1.5-8B_GGUFs"
GGUF_SOURCE_REVISION = "bc2e8f83688489e6b465daa833e9b318ea45c9d9"
GGUF_PROFILES = {
    "q2_k": {
        "file_name": "SenseNova-U1.5-8B-MoT-Q2_K.gguf",
        "file_size": 9_264_536_960,
        "file_sha256": "98f947928474f45e4c0c149f1af6009f15f99abd524b4dd36e2324d29303f2e5",
        "label": "Q2_K",
    },
    "q3_k_m": {
        "file_name": "SenseNova-U1.5-8B-MoT-Q3_K_M.gguf",
        "file_size": 10_920_713_600,
        "file_sha256": "82ccb1ee4cfd24d605ecaa97c99f799eef7bb78577185b0c1662d3d83c399636",
        "label": "Q3_K_M",
    },
    "q5_k_m": {
        "file_name": "SenseNova-U1.5-8B-MoT-Q5_K_M.gguf",
        "file_size": 15_107_169_664,
        "file_sha256": "1c496256eb114a5ff8fef278a63b39a75bd0c36e76f4280e89a06bb6ecb76ade",
        "label": "Q5_K_M",
    },
    "q6_k": {
        "file_name": "SenseNova-U1.5-8B-MoT-Q6_K.gguf",
        "file_size": 17_240_972_672,
        "file_sha256": "ded187014c0e34e13d20702d426a1741e9ec2aa698f3466df95ca0116d0e5ea2",
        "label": "Q6_K",
    },
    "q8_0": {
        "file_name": "SenseNova-U1.5-8B-MoT-Q8_0.gguf",
        "file_size": 21_174_689_152,
        "file_sha256": "61b227f036b7e8094cceab888c23b17a3fffc32d6182b039836d7cb31d688fe2",
        "label": "Q8_0",
    },
}
SUPPORTED_TENSOR_TYPES = {
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.BF16,
    gguf.GGMLQuantizationType.Q2_K,
    gguf.GGMLQuantizationType.Q3_K,
    gguf.GGMLQuantizationType.Q4_K,
    gguf.GGMLQuantizationType.Q5_K,
    gguf.GGMLQuantizationType.Q6_K,
    gguf.GGMLQuantizationType.Q8_0,
}
_VERIFIED_GGUF_FILES = set()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _gguf_profile(model_path, verify_hash=True):
    path = Path(model_path)
    if path.suffix.lower() != ".gguf":
        raise ValueError("SenseNova-U1.5 GGUF loader accepts .gguf files only")
    stat = path.stat()
    matches = [name for name, profile in GGUF_PROFILES.items() if stat.st_size == profile["file_size"]]
    if len(matches) != 1:
        supported = ", ".join(profile["label"] for profile in GGUF_PROFILES.values())
        raise ValueError(
            f"SenseNova-U1.5 GGUF file size is not one of the verified profiles ({supported}): {stat.st_size}"
        )
    profile_name = matches[0]
    profile = GGUF_PROFILES[profile_name]
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if verify_hash and cache_key not in _VERIFIED_GGUF_FILES:
        digest = _sha256_file(path)
        if digest != profile["file_sha256"]:
            raise ValueError(
                f"SenseNova-U1.5 {profile['label']} GGUF SHA256 mismatch: {digest} != {profile['file_sha256']}"
            )
        _VERIFIED_GGUF_FILES.add(cache_key)
    return profile_name


def _field_value(reader, field_name):
    field = reader.get_field(field_name)
    if field is None or len(field.types) != 1:
        return None
    value = field.parts[field.data[-1]]
    if field.types[0] == gguf.GGUFValueType.STRING:
        return str(value, encoding="utf-8")
    if field.types[0] in {
        gguf.GGUFValueType.UINT8,
        gguf.GGUFValueType.INT8,
        gguf.GGUFValueType.UINT16,
        gguf.GGUFValueType.INT16,
        gguf.GGUFValueType.UINT32,
        gguf.GGUFValueType.INT32,
        gguf.GGUFValueType.UINT64,
        gguf.GGUFValueType.INT64,
    }:
        return int(value)
    if field.types[0] in {gguf.GGUFValueType.FLOAT32, gguf.GGUFValueType.FLOAT64}:
        return float(value)
    if field.types[0] == gguf.GGUFValueType.BOOL:
        return bool(value)
    return None


def _metadata(reader):
    out = {}
    for field_name in reader.fields:
        try:
            value = _field_value(reader, field_name)
        except (IndexError, TypeError, ValueError):
            continue
        if value is not None:
            out[field_name] = value
    return out


def _original_shape(reader, tensor_name):
    field = reader.get_field(f"comfy.gguf.orig_shape.{tensor_name}")
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY:
        raise ValueError(f"invalid original-shape metadata for GGUF tensor: {tensor_name}")
    return torch.Size(tuple(int(field.parts[index][0]) for index in field.data))


def _normalize_tensor_names(names, expected):
    candidates = ("", "model.diffusion_model.", "diffusion_model.")
    for prefix in candidates:
        normalized = {
            name[len(prefix) :] if prefix and name.startswith(prefix) else name
            for name in names
        }
        if normalized == expected:
            return prefix
    missing = sorted(expected - set(names))[:5]
    unexpected = sorted(set(names) - expected)[:5]
    raise ValueError(
        f"SenseNova-U1.5 GGUF key mismatch: missing={missing}, unexpected={unexpected}"
    )


class GGMLTensor(torch.Tensor):
    def __new__(cls, *args, tensor_type, tensor_shape, patches=None, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, tensor_type, tensor_shape, patches=None, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = torch.Size(tensor_shape)
        self.patches = list(patches or [])

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        out.tensor_type = getattr(self, "tensor_type", None)
        out.tensor_shape = getattr(self, "tensor_shape", out.size())
        out.patches = list(getattr(self, "patches", []))
        if hasattr(self, "is_largest_weight"):
            out.is_largest_weight = self.is_largest_weight
        return out

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def new_empty(self, size, *args, **kwargs):
        out = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
            out,
            tensor_type=getattr(self, "tensor_type", None),
            tensor_shape=size,
            patches=list(getattr(self, "patches", [])),
        )


def _compiler_disable(function):
    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None)
    return disable(function) if callable(disable) else function


def _move_patch_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_patch_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_patch_to_device(item, device) for item in value]
    return value


class GGMLLayer(torch.nn.Module):
    comfy_cast_weights = True
    dequant_dtype = None
    compute_dtype = torch.bfloat16
    largest_layer = False

    def is_ggml_quantized(self, weight=None, bias=None):
        weight = self.weight if weight is None else weight
        bias = self.bias if bias is None else bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight = state_dict.get(f"{prefix}weight")
        bias = state_dict.get(f"{prefix}bias")
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self, torch.nn.Linear):
            return self._ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if isinstance(self, torch.nn.Embedding) and self.weight.shape[0] >= 64 * 1024:
            return self._ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _ggml_load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        prefix_length = len(prefix)
        for key, value in state_dict.items():
            suffix = key[prefix_length:]
            if suffix == "weight":
                self.weight = torch.nn.Parameter(value, requires_grad=False)
            elif suffix == "bias" and value is not None:
                self.bias = torch.nn.Parameter(value, requires_grad=False)
            else:
                unexpected_keys.append(key)
        if self.weight is None and isinstance(self, torch.nn.Linear):
            self.weight = torch.nn.Parameter(torch.zeros(self.in_features, self.out_features), requires_grad=False)
            missing_keys.append(prefix + "weight")
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if not self.is_ggml_quantized():
            return super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "weight"] = torch.empty_like(self.weight, device=torch.device("meta"))
        if self.bias is not None:
            destination[prefix + "bias"] = torch.empty_like(self.bias, device=torch.device("meta"))
        if self.largest_layer:
            destination[prefix + "temp.weight"] = torch.empty(
                self.weight.shape,
                device=torch.device("meta"),
                dtype=self.compute_dtype,
            )

    def _get_weight(self, tensor, dtype):
        if tensor is None:
            return None
        patch_list = []
        device = tensor.device
        for patches, key in getattr(tensor, "patches", []):
            patch_list.extend(_move_patch_to_device(patches, device))
        weight = dequantize_tensor(tensor, dtype=dtype, dequant_dtype=self.dequant_dtype)
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)
        if patch_list:
            weight = comfy.lora.calculate_weight(patch_list, weight, key)
        return weight

    @_compiler_disable
    def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            dtype = dtype or getattr(input, "dtype", self.compute_dtype)
            bias_dtype = bias_dtype or dtype
            device = device or input.device
        dtype = dtype or self.compute_dtype
        bias_dtype = bias_dtype or dtype
        device = device or self.weight.device
        non_blocking_check = getattr(comfy.model_management, "device_supports_non_blocking", None)
        non_blocking = non_blocking_check(device) if callable(non_blocking_check) else False
        bias = self._get_weight(self.bias.to(device), bias_dtype) if self.bias is not None else None
        weight = self._get_weight(self.weight.to(device), dtype)
        weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
        if bias is not None:
            bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)
        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)
        return torch.Tensor(out) if isinstance(out, GGMLTensor) else out

    def forward_ggml_cast_weights(self, input, *args, **kwargs):
        raise NotImplementedError


class GGMLOps(comfy.ops.manual_cast):
    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def forward_comfy_cast_weights(self, input, out_dtype=None):
            return super().forward_comfy_cast_weights(
                input,
                out_dtype=out_dtype or self.compute_dtype,
            )

        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype or self.compute_dtype
            weight, _ = self.cast_bias_weight(dtype=output_dtype, device=input.device)
            return torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(dtype=output_dtype)


class SenseNovaGGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False
    mmap_released = False
    named_modules_to_munmap = {}

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, **kwargs):
        weight = comfy.utils.get_attr(self.model, key)
        force_cast = kwargs.get("force_cast", False)
        return_weight = kwargs.get("return_weight", False)
        if not is_quantized(weight):
            return super().patch_weight_to_device(
                key,
                device_to=device_to,
                inplace_update=inplace_update,
                **kwargs,
            )
        if key not in self.patches and not force_cast:
            return weight
        out_weight = weight.to(device_to)
        if key in self.patches:
            patch_device = self.load_device if self.patch_on_device else self.offload_device
            patches = _move_patch_to_device(self.patches[key], patch_device)
            out_weight.patches = [(patches, key)]
        if return_weight:
            return out_weight
        comfy.utils.set_attr_param(self.model, key, out_weight)
        return out_weight

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for parameter in self.model.parameters():
                if is_torch_compatible(parameter):
                    continue
                parameter.patches = []
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    def pin_weight_to_device(self, key):
        module_key = key.rsplit(".", 1)[0]
        if not self.mmap_released and module_key in self.named_modules_to_munmap:
            self.named_modules_to_munmap[module_key].to(self.load_device).to(self.offload_device)
            del self.named_modules_to_munmap[module_key]
        return super().pin_weight_to_device(key)

    def load(self, *args, **kwargs):
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())
        kwargs["force_patch_weights"] = True
        super().load(*args, **kwargs)
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for name, module in self.named_modules_to_munmap.items():
                    for attribute in ("weight", "bias"):
                        value = getattr(module, attribute, None)
                        if getattr(value, "device", None) == self.offload_device:
                            linked.append((name, module))
                            break
            if linked and self.load_device != self.offload_device:
                logging.info("Releasing SenseNova GGUF mmap links for %d modules", len(linked))
                for _name, module in linked:
                    module.to(self.load_device).to(self.offload_device)
            self.mmap_released = True
            self.named_modules_to_munmap = {}

    def clone(self, *args, **kwargs):
        out = super().clone(*args, **kwargs)
        out.patch_on_device = self.patch_on_device
        out.mmap_released = self.mmap_released
        return out


def load_gguf_state_dict(model_path, verify_hash=True):
    profile_name = _gguf_profile(model_path, verify_hash=verify_hash)
    reader = gguf.GGUFReader(str(model_path))
    expected_contract = _checkpoint_contract("final")
    expected_keys = set(expected_contract)
    names = {tensor.name for tensor in reader.tensors}
    prefix = _normalize_tensor_names(names, expected_keys)
    prefix_length = len(prefix)
    state_dict = {}
    qtype_counts = collections.Counter()
    for tensor in reader.tensors:
        key = tensor.name[prefix_length:] if prefix and tensor.name.startswith(prefix) else tensor.name
        shape = _original_shape(reader, tensor.name)
        if shape is None:
            shape = torch.Size(tuple(int(value) for value in reversed(tensor.shape)))
        expected_shape = expected_contract[key][0]
        if tuple(shape) != expected_shape:
            raise ValueError(f"SenseNova-U1.5 GGUF shape mismatch for {key}: {tuple(shape)} != {expected_shape}")
        if tensor.tensor_type not in SUPPORTED_TENSOR_TYPES:
            qtype = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
            raise ValueError(f"unsupported SenseNova-U1.5 GGUF tensor type for {key}: {qtype}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            raw = torch.from_numpy(tensor.data)
        wrapped = GGMLTensor(raw, tensor_type=tensor.tensor_type, tensor_shape=shape)
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            wrapped = wrapped.view(*shape)
            wrapped.tensor_type = tensor.tensor_type
            wrapped.tensor_shape = shape
            wrapped.patches = []
        elif len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            wrapped = dequantize_tensor(wrapped, dtype=torch.float32)
        state_dict[key] = wrapped
        qtype_counts[getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))] += 1
    quantized = {key: value for key, value in state_dict.items() if is_quantized(value)}
    if not quantized:
        raise ValueError("SenseNova-U1.5 GGUF contains no quantized tensors")
    largest_key = max(quantized, key=lambda key: math.prod(quantized[key].shape))
    quantized[largest_key].is_largest_weight = True
    logging.info(
        "SenseNova GGUF profile %s tensor types: %s",
        GGUF_PROFILES[profile_name]["label"],
        ", ".join(f"{name} ({count})" for name, count in sorted(qtype_counts.items())),
    )
    return state_dict, _metadata(reader), profile_name


def load_sensenova_gguf_model(model_path, dtype=torch.bfloat16, disable_dynamic=False):
    del disable_dynamic
    model_path = Path(model_path)
    state_dict, metadata, profile_name = load_gguf_state_dict(model_path)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    model_config = SenseNovaModelConfig({})
    model_config.set_inference_dtype(dtype, dtype, device=load_device)
    model_config.custom_operations = GGMLOps()
    model = model_config.get_model(state_dict, device=torch.device("meta"))
    patcher = SenseNovaGGUFModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
    )
    model.load_model_weights(state_dict, assign=True)
    if state_dict:
        raise ValueError(f"SenseNova-U1.5 unused GGUF keys after load: {sorted(state_dict)[:5]}")
    patcher.cached_patcher_init = (load_sensenova_gguf_model, (model_path, dtype))
    profile = GGUF_PROFILES[profile_name]
    patcher.set_attachments(
        "sensenova_checkpoint",
        {
            "variant": "final",
            "profile": profile_name,
            "source_repo": MODEL_REPO,
            "source_revision": MODEL_REVISION,
            "quantization": profile["label"],
            "gguf_source_repo": GGUF_SOURCE_REPO,
            "gguf_source_revision": GGUF_SOURCE_REVISION,
            "gguf_metadata": metadata,
        },
    )
    return patcher
