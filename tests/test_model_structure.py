import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

import comfy.ops

import sensenova_u15.model as sensenova_model
from sensenova_u15.model import SenseNovaU15
from sensenova_u15.loader import _checkpoint_contract


class ModelStructureTests(unittest.TestCase):
    def test_state_dict_always_matches_bundled_checkpoint_contract(self):
        model = SenseNovaU15(
            device=torch.device("meta"),
            dtype=torch.bfloat16,
            operations=comfy.ops.disable_weight_init,
        )
        self.assertIs(model.dtype, torch.bfloat16)
        actual = model.state_dict()
        expected = {
            name: shape
            for name, (shape, _dtype) in _checkpoint_contract("final").items()
            if name != "language_model.lm_head.weight"
        }
        self.assertEqual(set(actual), set(expected))
        for name, tensor in actual.items():
            self.assertEqual(tuple(tensor.shape), expected[name], name)

    def test_prefix_attention_mask_matches_query_dtype(self):
        query = torch.empty(1, 32, 3, 128, dtype=torch.bfloat16)
        key = torch.empty(1, 8, 3, 128, dtype=torch.bfloat16)
        value = torch.empty_like(key)
        captured = {}

        def optimized_attention(query, key, value, heads, **kwargs):
            captured.update(query=query, key=key, value=value, heads=heads, kwargs=kwargs)
            return torch.empty(1, 3, 4096, dtype=torch.bfloat16)

        attention = SimpleNamespace(
            _project=lambda hidden_states, indexes, generation: (query, key, value),
            o_proj=lambda output: output,
        )
        with patch.object(sensenova_model, "optimized_attention", optimized_attention):
            output, _, _ = sensenova_model.Attention.forward_prefix(
                attention,
                torch.empty(1, 3, 4096, dtype=torch.bfloat16),
                torch.empty(3, 1, 3),
                torch.zeros(1, 1, 3, 3, dtype=torch.float32),
                {},
            )

        self.assertEqual(tuple(output.shape), (1, 3, 4096))
        self.assertIs(captured["kwargs"]["mask"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
