import json
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class ExampleWorkflowTests(unittest.TestCase):
    def load_example(self, name):
        return json.loads((PACKAGE_ROOT / "examples" / name).read_text(encoding="utf-8"))

    def test_examples_are_frontend_workflows(self):
        examples = sorted((PACKAGE_ROOT / "examples").glob("*.json"))
        self.assertEqual([path.name for path in examples], [
            "batch_t2i_workflow.json",
            "core_edit_workflow.json",
            "core_t2i_workflow.json",
            "edit_workflow.json",
            "gguf_edit_workflow.json",
            "gguf_t2i_workflow.json",
            "multi_reference_edit_workflow.json",
            "sft_edit_workflow.json",
            "sft_t2i_workflow.json",
            "t2i_8step_workflow.json",
            "t2i_workflow.json",
        ])
        for path in examples:
            workflow = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(name=path.name):
                self.assertIsInstance(workflow.get("nodes"), list)
                self.assertIsInstance(workflow.get("links"), list)
                self.assertIn("version", workflow)

    def test_frontend_workflows_have_resolved_links(self):
        for name in (
            "batch_t2i_workflow.json",
            "core_t2i_workflow.json",
            "core_edit_workflow.json",
            "t2i_workflow.json",
            "t2i_8step_workflow.json",
            "edit_workflow.json",
            "gguf_edit_workflow.json",
            "gguf_t2i_workflow.json",
            "multi_reference_edit_workflow.json",
            "sft_t2i_workflow.json",
            "sft_edit_workflow.json",
        ):
            workflow = self.load_example(name)
            nodes = {node["id"]: node for node in workflow["nodes"]}
            links = {link[0]: link for link in workflow["links"]}
            with self.subTest(name=name):
                self.assertEqual(workflow["last_node_id"], max(nodes))
                self.assertEqual(workflow["last_link_id"], max(links))
                for link_id, (_, origin_id, origin_slot, target_id, target_slot, link_type) in links.items():
                    self.assertEqual(nodes[origin_id]["outputs"][origin_slot]["type"], link_type, link_id)
                    self.assertEqual(nodes[target_id]["inputs"][target_slot]["type"], link_type, link_id)
                    self.assertEqual(nodes[target_id]["inputs"][target_slot]["link"], link_id, link_id)

    def test_core_workflows_use_native_loaders_and_nodes(self):
        for name in ("core_t2i_workflow.json", "core_edit_workflow.json"):
            workflow = self.load_example(name)
            loader = next(node for node in workflow["nodes"] if node["type"] == "CheckpointLoaderSimple")
            options = next(node for node in workflow["nodes"] if node["type"] == "SenseNovaSamplingOptions")
            latent = next(node for node in workflow["nodes"] if node["type"] == "EmptyHiDreamO1LatentImage")
            with self.subTest(name=name):
                self.assertEqual(loader["widgets_values"], ["SenseNova-U1.5-8B-MoT-BF16-T8.safetensors"])
                self.assertEqual(options["widgets_values"], [3])
                self.assertEqual(latent["widgets_values"], [1024, 1024, 1])

        edit = self.load_example("core_edit_workflow.json")
        reference = next(node for node in edit["nodes"] if node["type"] == "HiDreamO1ReferenceImages")
        self.assertEqual([value["name"] for value in reference["inputs"]], ["positive", "negative", "image_1"])
        self.assertEqual([value["name"] for value in reference["outputs"]], ["positive", "negative"])

    def test_frontend_t2i_uses_official_defaults(self):
        workflow = self.load_example("t2i_workflow.json")
        loader = next(node for node in workflow["nodes"] if node["type"] == "SenseNovaU15Loader")
        sampler = next(node for node in workflow["nodes"] if node["type"] == "KSampler")
        self.assertEqual(loader["widgets_values"], ["SenseNova-U1.5-8B-MoT-BF16-T8.safetensors"])
        self.assertEqual(sampler["widgets_values"][2:], [50, 4, "euler", "normal", 1])

    def test_gguf_examples_use_verified_q3_loader_and_native_pipeline(self):
        for name in ("gguf_t2i_workflow.json", "gguf_edit_workflow.json"):
            workflow = self.load_example(name)
            loader = next(node for node in workflow["nodes"] if node["type"] == "SenseNovaU15GGUFLoader")
            sampler = next(node for node in workflow["nodes"] if node["type"] == "KSampler")
            options = next(node for node in workflow["nodes"] if node["type"] == "SenseNovaSamplingOptions")
            with self.subTest(name=name):
                self.assertEqual(loader["widgets_values"], ["SenseNova-U1.5-8B-MoT-Q3_K_M.gguf"])
                self.assertEqual(sampler["widgets_values"][2:], [50, 4, "euler", "normal", 1])
                self.assertEqual(options["widgets_values"], [3])

    def test_batch_t2i_generates_two_variants_at_a_safe_example_resolution(self):
        workflow = self.load_example("batch_t2i_workflow.json")
        latent = next(node for node in workflow["nodes"] if node["type"] == "EmptySenseNovaLatentImage")
        sampler = next(node for node in workflow["nodes"] if node["type"] == "KSampler")
        self.assertEqual(latent["widgets_values"], [768, 768, 2, "Custom (use width / height)"])
        self.assertEqual(sampler["widgets_values"][2:], [50, 4, "euler", "normal", 1])

    def test_sft_examples_select_sft_checkpoint_and_keep_50_steps(self):
        for name in ("sft_t2i_workflow.json", "sft_edit_workflow.json"):
            workflow = self.load_example(name)
            loader = next(node for node in workflow["nodes"] if node["type"] == "SenseNovaU15Loader")
            with self.subTest(name=name):
                self.assertEqual(loader["widgets_values"], ["SenseNova-U1.5-8B-MoT-SFT-T8.safetensors"])
        sampler = next(
            node for node in self.load_example("sft_t2i_workflow.json")["nodes"] if node["type"] == "KSampler"
        )
        self.assertEqual(sampler["widgets_values"][2:], [50, 4, "euler", "normal", 1])

    def test_frontend_8step_uses_guarded_native_lora_and_official_defaults(self):
        workflow = self.load_example("t2i_8step_workflow.json")
        lora = next(node for node in workflow["nodes"] if node["type"] == "SenseNovaU15EightStepLoRA")
        sampler = next(node for node in workflow["nodes"] if node["type"] == "KSampler")
        options = next(node for node in workflow["nodes"] if node["type"] == "SenseNovaSamplingOptions")
        latent = next(node for node in workflow["nodes"] if node["type"] == "EmptySenseNovaLatentImage")
        self.assertEqual(lora["widgets_values"], ["SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors", 1])
        self.assertEqual(sampler["widgets_values"][2:], [8, 1, "euler", "normal", 1])
        self.assertEqual(options["widgets_values"], [3])
        self.assertEqual(latent["widgets_values"], [2048, 2048, 1, "1:1 — 2048 × 2048"])
        links = {link[0]: link for link in workflow["links"]}
        self.assertEqual(links[lora["inputs"][0]["link"]][1:3], [1, 0])
        self.assertEqual(links[options["inputs"][0]["link"]][1:3], [2, 0])
        self.assertEqual(links[sampler["inputs"][0]["link"]][1:3], [3, 0])

    def test_frontend_edit_and_multi_reference_contracts(self):
        edit = self.load_example("edit_workflow.json")
        edit_reference = next(node for node in edit["nodes"] if node["type"] == "SenseNovaReferenceImage")
        self.assertEqual([value["name"] for value in edit_reference["inputs"]], ["positive", "negative", "Image-1"])
        self.assertEqual(edit_reference["inputs"][-1]["localized_name"], "参考图 1 (Image-1)")
        self.assertEqual(edit_reference["inputs"][-1]["label"], "参考图 1 (Image-1)")

        multi = self.load_example("multi_reference_edit_workflow.json")
        multi_reference = next(node for node in multi["nodes"] if node["type"] == "SenseNovaReferenceImage")
        self.assertEqual([value["name"] for value in multi_reference["inputs"][-2:]], ["Image-1", "Image-2"])
        self.assertEqual([value["localized_name"] for value in multi_reference["inputs"][-2:]], ["参考图 1 (Image-1)", "参考图 2 (Image-2)"])
        self.assertEqual([value["label"] for value in multi_reference["inputs"][-2:]], ["参考图 1 (Image-1)", "参考图 2 (Image-2)"])
        guider = next(node for node in multi["nodes"] if node["type"] == "SenseNovaEditGuider")
        scheduler = next(node for node in multi["nodes"] if node["type"] == "BasicScheduler")
        self.assertEqual(guider["widgets_values"], [4, 1, "global", 0, 1])
        self.assertEqual(scheduler["widgets_values"], ["normal", 50, 1])
        latent = next(node for node in multi["nodes"] if node["type"] == "EmptySenseNovaLatentImage")
        self.assertEqual(latent["widgets_values"], [2048, 2048, 1, "1:1 — 2048 × 2048"])
        links = {link[0]: link for link in multi["links"]}
        guider_model = links[guider["inputs"][0]["link"]][1:3]
        scheduler_model = links[scheduler["inputs"][0]["link"]][1:3]
        self.assertEqual(guider_model, scheduler_model)


if __name__ == "__main__":
    unittest.main()
