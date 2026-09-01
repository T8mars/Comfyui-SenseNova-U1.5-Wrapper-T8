import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gguf
import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

PACKAGE_NAME = "comfyui_sensenova_u15_t8_tests"
if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)

from comfyui_sensenova_u15_t8_tests.sensenova_u15.gguf_dequant import dequantize
from comfyui_sensenova_u15_t8_tests.sensenova_u15 import gguf_support
from comfyui_sensenova_u15_t8_tests.sensenova_u15.gguf_support import (
    GGMLOps,
    GGMLTensor,
    _VERIFIED_GGUF_FILES,
    _gguf_profile,
)


QUANTIZED_TYPES = (
    gguf.GGMLQuantizationType.Q2_K,
    gguf.GGMLQuantizationType.Q3_K,
    gguf.GGMLQuantizationType.Q4_K,
    gguf.GGMLQuantizationType.Q5_K,
    gguf.GGMLQuantizationType.Q6_K,
    gguf.GGMLQuantizationType.Q8_0,
)


class GGUFDequantizationTests(unittest.TestCase):
    def test_torch_dequantization_matches_gguf_reference(self):
        random = np.random.default_rng(1234)
        for qtype in QUANTIZED_TYPES:
            block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
            raw = random.integers(0, 256, size=(2, type_size), dtype=np.uint8)
            expected = gguf.quants.dequantize(raw, qtype)
            actual = dequantize(
                torch.from_numpy(raw),
                qtype,
                (2, block_size),
                dtype=torch.float32,
            ).numpy()
            with self.subTest(qtype=qtype.name):
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=True)

    def test_logical_tensor_shape_survives_device_and_dtype_moves(self):
        _block_size, type_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q8_0]
        tensor = GGMLTensor(
            torch.zeros((1, type_size), dtype=torch.uint8),
            tensor_type=gguf.GGMLQuantizationType.Q8_0,
            tensor_shape=(1, 32),
        )
        moved = tensor.to(device="cpu")
        self.assertEqual(tuple(tensor.shape), (1, 32))
        self.assertEqual(tuple(moved.shape), (1, 32))
        self.assertEqual(moved.tensor_type, gguf.GGMLQuantizationType.Q8_0)

    def test_quantized_linear_runs_through_native_torch_module(self):
        _block_size, type_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q8_0]
        layer = GGMLOps.Linear(32, 2, bias=False)
        weight = GGMLTensor(
            torch.zeros((2, type_size), dtype=torch.uint8),
            tensor_type=gguf.GGMLQuantizationType.Q8_0,
            tensor_shape=(2, 32),
        )
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        output = layer(torch.randn((3, 32), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3, 2))
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_f16_embedding_is_normalized_to_bf16_compute_dtype(self):
        layer = GGMLOps.Embedding(4, 3)
        weight = GGMLTensor(
            torch.arange(12, dtype=torch.float16).reshape(4, 3),
            tensor_type=gguf.GGMLQuantizationType.F16,
            tensor_shape=(4, 3),
        )
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        output = layer(torch.tensor([[0, 3]], dtype=torch.long))
        self.assertEqual(output.dtype, torch.bfloat16)
        torch.testing.assert_close(output.float(), weight[[0, 3]].float().unsqueeze(0))


class GGUFProfileTests(unittest.TestCase):
    def setUp(self):
        _VERIFIED_GGUF_FILES.clear()

    def tearDown(self):
        _VERIFIED_GGUF_FILES.clear()

    def test_verified_profile_requires_exact_size_and_sha256(self):
        payload = b"verified SenseNova GGUF fixture"
        digest = hashlib.sha256(payload).hexdigest()
        profiles = {
            "fixture": {
                "file_name": "fixture.gguf",
                "file_size": len(payload),
                "file_sha256": digest,
                "label": "FIXTURE",
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fixture.gguf"
            path.write_bytes(payload)
            with patch.object(gguf_support, "GGUF_PROFILES", profiles):
                self.assertEqual(_gguf_profile(path), "fixture")
                path.write_bytes(b"x")
                with self.assertRaisesRegex(ValueError, "file size"):
                    _gguf_profile(path)

    def test_verified_profile_rejects_hash_mismatch_and_non_gguf_extension(self):
        payload = b"fixture"
        profiles = {
            "fixture": {
                "file_name": "fixture.gguf",
                "file_size": len(payload),
                "file_sha256": "0" * 64,
                "label": "FIXTURE",
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_path = Path(temp_dir) / "fixture.gguf"
            gguf_path.write_bytes(payload)
            with patch.object(gguf_support, "GGUF_PROFILES", profiles):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    _gguf_profile(gguf_path)
                with self.assertRaisesRegex(ValueError, "accepts .gguf files only"):
                    _gguf_profile(Path(temp_dir) / "fixture.bin")


if __name__ == "__main__":
    unittest.main()
