import sys
import unittest
from pathlib import Path


COMFY_ROOT = Path(__file__).resolve().parents[3]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

from transformers import Qwen2Tokenizer

from sensenova_u15.text_encoder import (
    SenseNovaTokenizer,
    build_generation_prompt,
    build_interleave_prompt,
    build_interleave_unconditional_prompt,
    build_unconditional_prompt,
)


class TextEncoderTests(unittest.TestCase):
    def test_generation_tokens_match_official_tokenizer(self):
        prompt = "画一只蓝眼睛的狐狸"
        asset_dir = Path(__file__).resolve().parents[1] / "sensenova_u15" / "tokenizer"
        reference = Qwen2Tokenizer.from_pretrained(asset_dir, local_files_only=True)
        expected = reference(build_generation_prompt(prompt), add_special_tokens=True)["input_ids"]

        tokenizer = SenseNovaTokenizer()
        actual = tokenizer.tokenize_with_weights(prompt)["sensenova_u15"][0]
        actual = [int(value[0]) for value in actual]
        self.assertEqual(actual, expected)
        self.assertEqual(actual[-1], 151670)

    def test_empty_prompt_matches_official_unconditional_query(self):
        asset_dir = Path(__file__).resolve().parents[1] / "sensenova_u15" / "tokenizer"
        reference = Qwen2Tokenizer.from_pretrained(asset_dir, local_files_only=True)
        expected = reference(build_unconditional_prompt(), add_special_tokens=True)["input_ids"]
        actual = SenseNovaTokenizer().tokenize_with_weights("")["sensenova_u15"][0]
        self.assertEqual([int(value[0]) for value in actual], expected)

    def test_generation_prompt_selects_thinking_protocol(self):
        no_thinking = build_generation_prompt("test")
        thinking = build_generation_prompt("test", thinking=True)

        self.assertTrue(no_thinking.endswith("<think>\n\n</think>\n\n<img>"))
        self.assertTrue(thinking.endswith("<think>\n"))
        self.assertEqual(thinking.rsplit("<|im_start|>assistant\n", 1)[-1], "<think>\n")

    def test_interleave_prompt_leaves_image_event_to_the_model(self):
        no_thinking = build_interleave_prompt("test")
        thinking = build_interleave_prompt("test", thinking=True)

        self.assertTrue(no_thinking.endswith("<think>\n\n</think>\n\n"))
        self.assertFalse(no_thinking.endswith("<img>"))
        self.assertTrue(thinking.endswith("<|im_start|>assistant\n"))
        self.assertEqual(
            build_interleave_unconditional_prompt(),
            "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n",
        )

    def test_tokenizer_selects_interleave_protocol(self):
        values = SenseNovaTokenizer().tokenize_with_weights("test", mode="interleave")["sensenova_u15"][0]
        self.assertNotEqual(int(values[-1][0]), 151670)


if __name__ == "__main__":
    unittest.main()
