import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

spec = importlib.util.spec_from_file_location(
    "comfyui_sensenova_u15_t8_interleave_tests",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)

from comfyui_sensenova_u15_t8_interleave_tests.nodes import (
    SenseNovaInterleave,
    SenseNovaTextEncode,
    SenseNovaThinkingPreview,
    interleave_output_samples,
)
from comfyui_sensenova_u15_t8_interleave_tests.sensenova_u15 import model as sensenova_model
from comfyui_sensenova_u15_t8_interleave_tests.sensenova_u15.conditioning import (
    condition_input_ids,
    conditioned_input_length,
)
from comfyui_sensenova_u15_t8_interleave_tests.sensenova_u15.interleave import (
    InterleaveResult,
    SenseNovaInterleaveSession,
    build_interleave_result,
    interleave_result_to_markdown,
    prefix_arguments,
)
from comfyui_sensenova_u15_t8_interleave_tests.sensenova_u15.model_config import (
    CONDSharedList,
    CONDSharedRegular,
    SenseNovaBaseModel,
)
from comfyui_sensenova_u15_t8_interleave_tests.sensenova_u15.text_encoder import SenseNovaTokenizer


class InterleaveSessionTests(unittest.TestCase):
    def test_generates_text_image_and_resumed_text(self):
        image_start = 151670
        eos = 151645

        class Model:
            def _preprocess_prefix_state(self, input_ids, *args):
                branch = int(input_ids.item())
                return (101 if branch == 1 else 0), [[]], [[]], torch.tensor([3])

            def _next_text_token(self, hidden):
                return torch.tensor([hidden])

            def _decode_text_token(self, token, keys, values, prefix_time, transformer_options=None):
                token_id = int(token.item())
                keys = [keys[0] + [token_id]]
                return {101: image_start, image_start: 0, 102: eos}[token_id], keys, values, prefix_time + 1

            def append_interleave_image(self, image, keys, values, prefix_time, transformer_options=None):
                self.last_image = image
                return 102, [keys[0] + ["image"]], values, prefix_time + 2

        sampled_prefixes = []

        def sample_image(positive, negative):
            sampled_prefixes.append((list(positive.keys[0]), list(negative.keys[0])))
            return torch.full((1, 3, 32, 32), 0.25)

        session = SenseNovaInterleaveSession(
            Model(),
            positive_prefix=(torch.tensor([[1]]),),
            negative_prefix=(torch.tensor([[2]]),),
            decode_tokens=lambda values: {101: "before ", 102: "after"}[values[0]],
        )
        result = session.generate(sample_image, max_text_tokens=8, max_images=1)

        self.assertEqual(result.text, "before <image>after")
        self.assertEqual(result.token_ids, [101, image_start, 102, eos])
        self.assertEqual(result.stop_reason, "eos")
        self.assertEqual(sampled_prefixes, [([101, image_start], [image_start])])
        torch.testing.assert_close(result.images[0], torch.full((1, 3, 32, 32), 0.25))

    def test_multiple_images_share_one_session(self):
        image_start = 151670
        eos = 151645

        class Model:
            append_calls = 0

            def _preprocess_prefix_state(self, input_ids, *args):
                return (101 if int(input_ids.item()) == 1 else 0), [[]], [[]], torch.tensor([3])

            def _next_text_token(self, hidden):
                return torch.tensor([hidden])

            def _decode_text_token(self, token, keys, values, prefix_time, transformer_options=None):
                token_id = int(token.item())
                return {101: image_start, image_start: 0, 102: image_start, 103: eos}[token_id], keys, values, prefix_time + 1

            def append_interleave_image(self, image, keys, values, prefix_time, transformer_options=None):
                self.append_calls += 1
                return (102 if self.append_calls <= 2 else 103), keys, values, prefix_time + 2

        images = []

        def sample_image(*_args):
            image = torch.full((1, 3, 32, 32), len(images) + 1.0)
            images.append(image)
            return image

        session = SenseNovaInterleaveSession(
            Model(),
            positive_prefix=(torch.tensor([[1]]),),
            negative_prefix=(torch.tensor([[2]]),),
            decode_tokens=lambda values: {101: "first", 102: "second", 103: "end"}[values[0]],
        )
        result = session.generate(sample_image, max_text_tokens=8, max_images=2)

        self.assertEqual(result.text, "first<image>second<image>end")
        self.assertEqual(len(result.images), 2)
        self.assertEqual(result.stop_reason, "eos")


class ThinkingTests(unittest.TestCase):
    def test_decode_appends_stop_and_image_suffix(self):
        model = object.__new__(sensenova_model.SenseNovaU15)
        torch.nn.Module.__init__(model)
        model.has_lm_head = True
        tokens = iter((42, sensenova_model.THINK_END_TOKEN_ID))
        decoded = []
        model._preprocess_prefix_state = lambda *args: (
            torch.zeros(1, 1, 1),
            [torch.zeros(1, 1, 1, 1)],
            [torch.zeros(1, 1, 1, 1)],
            torch.tensor([3]),
        )
        model._next_text_token = lambda hidden: torch.tensor([next(tokens)])

        def decode(token, keys, values, prefix_time, transformer_options=None):
            decoded.append(int(token.item()))
            return torch.zeros(1, 1, 1), keys, values, prefix_time + 1

        model._decode_text_token = decode
        _, _, prefix_time = model.preprocess_thinking_prefix(torch.tensor([[1, 2, 3]]), max_think_tokens=4)

        self.assertEqual(
            decoded,
            [42, sensenova_model.THINK_END_TOKEN_ID, *sensenova_model.THINK_SUFFIX_TOKEN_IDS],
        )
        self.assertEqual(prefix_time.tolist(), [7])

    def test_decode_returns_visible_thinking_token_ids(self):
        model = object.__new__(sensenova_model.SenseNovaU15)
        torch.nn.Module.__init__(model)
        model.has_lm_head = True
        tokens = iter((42, sensenova_model.THINK_END_TOKEN_ID))
        model._preprocess_prefix_state = lambda *args: (
            torch.zeros(1, 1, 1),
            [torch.zeros(1, 1, 1, 1)],
            [torch.zeros(1, 1, 1, 1)],
            torch.tensor([3]),
        )
        model._next_text_token = lambda hidden: torch.tensor([next(tokens)])
        model._decode_text_token = lambda token, keys, values, prefix_time, transformer_options=None: (
            torch.zeros(1, 1, 1), keys, values, prefix_time + 1
        )

        _, _, _, token_ids = model.preprocess_thinking_prefix_with_tokens(
            torch.tensor([[1, 2, 3]]), max_think_tokens=4
        )

        self.assertEqual(token_ids, [42, sensenova_model.THINK_END_TOKEN_ID])

    def test_checkpoint_without_lm_head_is_rejected_for_thinking(self):
        model = object.__new__(sensenova_model.SenseNovaU15)
        torch.nn.Module.__init__(model)
        model.has_lm_head = False
        with self.assertRaisesRegex(RuntimeError, "lm_head"):
            model.preprocess_thinking_prefix(torch.tensor([[1]]))


class InterleaveConditioningTests(unittest.TestCase):
    def test_live_prefix_bypasses_text_conditioning(self):
        model = object.__new__(SenseNovaBaseModel)
        torch.nn.Module.__init__(model)
        model.concat_keys = ()
        model.manual_cast_dtype = None
        keys = [torch.zeros(1, 1, 3, 1)]
        values = [torch.ones(1, 1, 3, 1)]
        time = torch.tensor([3])

        conds = model.extra_conds(prefix_keys=keys, prefix_values=values, prefix_time=time)

        self.assertIsInstance(conds["prefix_keys"], CONDSharedList)
        self.assertIsInstance(conds["prefix_time"], CONDSharedRegular)
        self.assertIs(conds["prefix_keys"].cond, keys)
        self.assertIs(conds["prefix_values"].cond, values)
        self.assertIs(conds["prefix_time"].cond, time)

    def test_negative_reference_waits_for_generated_image_event(self):
        tokenizer = SenseNovaTokenizer()
        pairs = tokenizer.tokenize_with_weights("", mode="interleave")["sensenova_u15"][0]
        input_ids = torch.tensor([[int(pair[0]) for pair in pairs]])
        conditioned = condition_input_ids(
            input_ids,
            [(1, 2)],
            image_only=True,
            append_image_start=False,
        )

        self.assertEqual(torch.count_nonzero(conditioned == 151669).item(), 2)
        self.assertEqual(conditioned[0, -1].item(), 198)
        self.assertEqual(
            conditioned.shape[1],
            conditioned_input_length(
                input_ids.shape[1],
                [(1, 2)],
                image_only=True,
                append_image_start=False,
            ),
        )

    def test_custom_node_reference_key_is_in_initial_prefix(self):
        tokenizer = SenseNovaTokenizer()
        pairs = tokenizer.tokenize_with_weights("continue", mode="interleave")["sensenova_u15"][0]
        input_ids = torch.tensor([[int(pair[0]) for pair in pairs]])
        conditioned, references, indexes, prefix_mask = prefix_arguments(
            {
                "text_input_ids": input_ids,
                "sensenova_reference_images": [torch.ones(1, 33, 65, 3)],
            },
            torch.device("cpu"),
            torch.float32,
            image_only=False,
        )

        expected_tokens = (references[0].shape[-2] // 32) * (references[0].shape[-1] // 32)
        self.assertEqual(torch.count_nonzero(conditioned == 151669).item(), expected_tokens)
        self.assertEqual(references[0].shape, (1, 3, 384, 736))
        self.assertEqual(indexes.shape[-1], conditioned.shape[1])
        self.assertEqual(prefix_mask.shape[-1], conditioned.shape[1])


class InterleaveNodeTests(unittest.TestCase):
    def test_text_encode_sets_protocol_metadata(self):
        calls = []

        class Clip:
            def tokenize(self, text, **kwargs):
                calls.append(("tokenize", text, kwargs))
                return {"tokens": text}

            def encode_from_tokens_scheduled(self, tokens, add_dict):
                calls.append(("encode", tokens, add_dict))
                return [[torch.empty(1), add_dict]]

        output = SenseNovaTextEncode.execute(
            clip=Clip(),
            text="test",
            thinking=True,
            max_think_tokens=64,
            mode="interleave",
        )[0]

        self.assertEqual(calls[0], ("tokenize", "test", {"thinking": True, "mode": "interleave"}))
        self.assertTrue(output[0][1]["sensenova_interleave"])
        self.assertTrue(output[0][1]["sensenova_thinking"])
        self.assertEqual(
            output[0][1]["sensenova_thinking_result"],
            {"enabled": True, "token_ids": None},
        )

    def test_thinking_preview_decodes_tokens_after_sampling(self):
        calls = []

        class Clip:
            def decode(self, token_ids, skip_special_tokens=True):
                calls.append((token_ids, skip_special_tokens))
                return "  inspect the layout  "

        conditioning = [[None, {"sensenova_thinking_result": {"enabled": True, "token_ids": [41, 42]}}]]
        output = SenseNovaThinkingPreview.execute(
            clip=Clip(),
            conditioning=conditioning,
            samples={"samples": torch.empty(1, 3, 8, 8)},
        )

        self.assertEqual(output[0], "inspect the layout")
        self.assertEqual(output.ui.as_dict(), {"text": ("inspect the layout",)})
        self.assertEqual(calls, [([41, 42], True)])

    def test_thinking_preview_explains_unavailable_result(self):
        output = SenseNovaThinkingPreview.execute(
            clip=SimpleNamespace(decode=lambda *_args, **_kwargs: "unused"),
            conditioning=[[None, {"sensenova_thinking_result": {"enabled": True, "token_ids": None}}]],
            samples={"samples": torch.empty(1, 3, 8, 8)},
        )
        self.assertIn("has not run", output[0])

    def test_schema_uses_standard_sampling_inputs(self):
        schema = SenseNovaInterleave.define_schema()
        self.assertEqual(
            {value.id for value in schema.inputs},
            {
                "model", "clip", "positive", "negative", "noise_seed", "cfg", "sampler",
                "sigmas", "latent_image", "max_text_tokens", "max_images",
            },
        )
        self.assertEqual([value.display_name for value in schema.outputs], ["samples", "text", "interleave_result"])

    def test_result_preserves_order_thinking_and_missing_images(self):
        result = InterleaveResult(
            "<think>plan</think>Hello<image>Middle<image>After",
            [torch.empty(1, 3, 8, 8)],
            [1, 2, 3],
            "eos",
        )
        payload = build_interleave_result(result)

        self.assertEqual(payload["parts"][0], {"type": "think", "text": "plan"})
        self.assertEqual(payload["parts"][2], {"type": "image", "index": 0})
        self.assertEqual(payload["parts"][4], {"type": "image", "index": 1, "missing": True})
        self.assertEqual(
            interleave_result_to_markdown(payload, include_think=False),
            "Hello\n\n[image:0]\n\nMiddle\n\n[image:1]\n\nAfter",
        )

    def test_no_generated_image_keeps_input_latent(self):
        latent = torch.randn(1, 3, 8, 8)
        self.assertIs(interleave_output_samples(InterleaveResult("text", [], [1], "eos"), latent), latent)


class AppendImageTests(unittest.TestCase):
    def test_generated_image_extends_prefix_and_masks_image_end(self):
        captured = {}

        class Layer:
            def forward_decode(self, hidden_states, indexes, prefix_key, prefix_value, transformer_options, attention_mask=None):
                captured["hidden"] = hidden_states
                captured["indexes"] = indexes
                captured["mask"] = attention_mask
                count = hidden_states.shape[1]
                key = torch.cat((prefix_key, torch.full((1, 1, count, 1), 7.0)), dim=2)
                value = torch.cat((prefix_value, torch.full((1, 1, count, 1), 8.0)), dim=2)
                return hidden_states + 1, key, value

        class VisionModel:
            def __call__(self, image):
                captured["vision_input"] = image
                return torch.zeros(1, 2, sensenova_model.HIDDEN_SIZE)

        def embed_tokens(token_ids):
            self.assertEqual(token_ids.tolist(), [[151671]])
            return torch.ones(1, 1, sensenova_model.HIDDEN_SIZE)

        model = SimpleNamespace(
            vision_model=VisionModel(),
            language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embed_tokens, layers=[Layer()])),
        )
        image = torch.zeros(1, 3, 32, 64)
        keys = [torch.zeros(1, 1, 3, 1)]
        values = [torch.zeros(1, 1, 3, 1)]

        _, next_keys, next_values, next_time = sensenova_model.SenseNovaU15.append_interleave_image(
            model, image, keys, values, torch.tensor([3]), transformer_options={}
        )

        self.assertEqual(captured["indexes"].tolist(), [[[3, 3, 4]], [[0, 0, 0]], [[0, 1, 0]]])
        self.assertEqual(captured["mask"].shape, (1, 1, 3, 6))
        self.assertTrue(torch.isneginf(captured["mask"][0, 0, :2, 5]).all())
        self.assertEqual(next_keys[0].shape, (1, 1, 6, 1))
        self.assertEqual(next_values[0].shape, (1, 1, 6, 1))
        self.assertEqual(next_time.tolist(), [5])


if __name__ == "__main__":
    unittest.main()
