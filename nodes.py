import math
from pathlib import Path

from typing_extensions import override

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.samplers
import folder_paths
import node_helpers
from comfy_api.latest import ComfyExtension, io

from .sensenova_u15.loader import load_pixel_vae, load_sensenova_clip, load_sensenova_model
from .sensenova_u15.lora import apply_eight_step_lora
from .sensenova_u15.guidance import (
    CFG_NORM_MODES,
    build_structured_edit_prompt,
    edit_guidance,
    rescale_denoised_guidance,
)
from .sensenova_u15.sampling import SenseNovaModelSampling


GGUF_MODEL_DIR = Path(folder_paths.models_dir) / "gguf"
GGUF_MODEL_DIR.mkdir(parents=True, exist_ok=True)
folder_paths.folder_names_and_paths["sensenova_gguf"] = ([str(GGUF_MODEL_DIR)], {".gguf"})


RESOLUTION_PRESETS = {
    "Custom (use width / height)": None,
    "1:1 — 2048 × 2048": (2048, 2048),
    "16:9 — 2720 × 1536": (2720, 1536),
    "9:16 — 1536 × 2720": (1536, 2720),
    "2:3 — 1664 × 2496": (1664, 2496),
    "3:2 — 2496 × 1664": (2496, 1664),
}


class SenseNovaU15Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaU15Loader",
            display_name="SenseNova U1.5 Loader (Final / SFT)",
            category="loaders/SenseNova",
            description="Load a verified single-file SenseNova-U1.5 Final or SFT checkpoint. Preview is rejected.",
            inputs=[
                io.Combo.Input(
                    id="model_name",
                    options=folder_paths.get_filename_list("diffusion_models"),
                ),
            ],
            outputs=[io.Model.Output(), io.Clip.Output(), io.Vae.Output()],
        )

    @classmethod
    def execute(cls, *, model_name):
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        clip = load_sensenova_clip()
        model = load_sensenova_model(model_path, torch.bfloat16)
        return io.NodeOutput(model, clip, load_pixel_vae())


class SenseNovaU15GGUFLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaU15GGUFLoader",
            display_name="SenseNova U1.5 GGUF Loader (Final)",
            category="loaders/SenseNova",
            description=(
                "Load a verified Q2_K, Q3_K_M, Q5_K_M, Q6_K, or Q8_0 quantization of the "
                "official SenseNova U1.5 Final model while keeping the native ComfyUI pipeline."
            ),
            inputs=[
                io.Combo.Input(
                    id="model_name",
                    options=folder_paths.get_filename_list("sensenova_gguf"),
                ),
            ],
            outputs=[io.Model.Output(), io.Clip.Output(), io.Vae.Output()],
        )

    @classmethod
    def execute(cls, *, model_name):
        try:
            from .sensenova_u15.gguf_support import load_sensenova_gguf_model
        except ImportError as error:
            raise ImportError(
                "SenseNova U1.5 GGUF support requires the Python package `gguf>=0.13.0`. "
                "Reinstall or update this custom node through ComfyUI Manager."
            ) from error
        model_path = folder_paths.get_full_path_or_raise("sensenova_gguf", model_name)
        clip = load_sensenova_clip()
        model = load_sensenova_gguf_model(model_path, torch.bfloat16)
        return io.NodeOutput(model, clip, load_pixel_vae())


class EmptySenseNovaLatentImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptySenseNovaLatentImage",
            display_name="Empty SenseNova Pixel Latent",
            category="latent/SenseNova",
            inputs=[
                io.Int.Input(id="width", default=2048, min=64, max=4096, step=32),
                io.Int.Input(id="height", default=2048, min=64, max=4096, step=32),
                io.Int.Input(
                    id="batch_size",
                    default=1,
                    min=1,
                    max=16,
                    tooltip="Generate 1-16 variants with the same prompt and reference images. Lower the resolution when using larger batches.",
                ),
                io.Combo.Input(
                    id="resolution_preset",
                    options=list(RESOLUTION_PRESETS),
                    default="Custom (use width / height)",
                    optional=True,
                    tooltip="Official SenseNova aspect-ratio presets. Custom keeps the width and height above.",
                ),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, *, width, height, batch_size=1, resolution_preset="Custom (use width / height)"):
        if resolution_preset not in RESOLUTION_PRESETS:
            raise ValueError(f"unsupported SenseNova resolution preset: {resolution_preset}")
        preset = RESOLUTION_PRESETS[resolution_preset]
        if preset is not None:
            width, height = preset
        samples = torch.zeros(
            (batch_size, 3, height, width),
            device=comfy.model_management.intermediate_device(),
        )
        return io.NodeOutput({"samples": samples})


class SenseNovaU15EightStepLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaU15EightStepLoRA",
            display_name="SenseNova U1.5 8-Step LoRA (Final only)",
            category="loaders/SenseNova",
            description="Apply the verified official 8-step LoRA to a SenseNova U1.5 Final model.",
            inputs=[
                io.Model.Input(id="model"),
                io.Combo.Input(id="lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input(id="strength_model", default=1.0, min=0.0, max=2.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, *, model, lora_name, strength_model):
        return io.NodeOutput(apply_eight_step_lora(model, lora_name, strength_model))


class SenseNovaSamplingOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaSamplingOptions",
            display_name="SenseNova Sampling Options",
            category="model/patch/SenseNova",
            description="Set the official flow timestep shift while preserving the upstream sigma trajectory.",
            inputs=[
                io.Model.Input(id="model"),
                io.Float.Input(id="shift", default=3.0, min=0.01, max=100.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, *, model, shift):
        patched = model.clone()
        model_sampling = SenseNovaModelSampling(patched.model.model_config)
        model_sampling.set_parameters(shift=shift)
        patched.add_object_patch("model_sampling", model_sampling)
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "sensenova_prefix_cache",
            _prefix_cache_sample_wrapper,
        )
        return io.NodeOutput(patched)


def _prefix_cache_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    original_model_options = guider.model_options
    guider.model_options = comfy.model_patcher.create_model_options_clone(original_model_options)
    cache = {}
    guider.model_options.setdefault("transformer_options", {})["sensenova_prefix_cache"] = cache
    try:
        return executor(*args, **kwargs)
    finally:
        cache.clear()
        guider.model_options = original_model_options


def _reference_image_inputs(max_images):
    return [
        io.Image.Input(
            id=f"Image-{index}",
            display_name=f"参考图 {index} (Image-{index})",
            optional=index > 1,
            tooltip=(
                "Main/source image. In a garment edit, connect the person here."
                if index == 1
                else "Optional second reference. In a garment edit, connect the clothing here."
                if index == 2
                else f"Optional reference image {index}."
            ),
        )
        for index in range(1, max_images + 1)
    ]


class SenseNovaReferenceImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaReferenceImage",
            display_name="SenseNova Reference Image",
            category="conditioning/SenseNova",
            description="Attach one or two source images for instruction editing. Use the 1-10 node for larger reference sets.",
            inputs=[
                io.Conditioning.Input(id="positive"),
                io.Conditioning.Input(id="negative"),
                *_reference_image_inputs(2),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="image_condition"),
            ],
        )

    @classmethod
    def execute(cls, *, positive, negative, images=None, **kwargs):
        # ``images`` keeps direct-call compatibility with the former Autogrow
        # implementation. Frontend workflows use the explicit Image-N inputs.
        if images is not None:
            references = [
                images[name]
                for name in ["image"] + [f"image_{index}" for index in range(2, 11)]
                if name in images
            ]
        else:
            references = [kwargs[f"Image-{index}"] for index in range(1, 11) if f"Image-{index}" in kwargs]
        for image in references:
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
                raise ValueError("Each SenseNova reference input requires one IMAGE with at least three channels")
        positive = node_helpers.conditioning_set_values(
            positive,
            {
                "sensenova_reference_images": references,
                "sensenova_reference_mode": "condition",
            },
            append=True,
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {
                "sensenova_reference_images": references,
                "sensenova_reference_mode": "image_only",
            },
            append=True,
        )
        return io.NodeOutput(positive, negative)


class SenseNovaReferenceImageAdvanced(SenseNovaReferenceImage):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaReferenceImageAdvanced",
            display_name="SenseNova Reference Images (1-10)",
            category="conditioning/SenseNova",
            description="Attach 1-10 source images for advanced multi-reference editing.",
            inputs=[
                io.Conditioning.Input(id="positive"),
                io.Conditioning.Input(id="negative"),
                *_reference_image_inputs(10),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="image_condition"),
            ],
        )


class SenseNovaStructuredEditPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaStructuredEditPrompt",
            display_name="SenseNova Structured Edit Prompt",
            category="conditioning/SenseNova",
            description="Turn an edit request into explicit modification, reference-role, preservation, and exclusion sections.",
            inputs=[
                io.String.Input(
                    id="instruction",
                    multiline=True,
                    default="",
                    tooltip="Describe only the change you want, for example: make the person in Image-1 wear the clothes from Image-2.",
                ),
                io.String.Input(
                    id="image_roles",
                    multiline=True,
                    default="参考图作为主画面和待编辑对象。多图时请明确写 Image-1、Image-2 各自提供什么。",
                    tooltip="Assign a single clear role to each reference image. Image-1 is the first connected socket.",
                ),
                io.String.Input(
                    id="preserve",
                    multiline=True,
                    default="保持主体身份、姿势、构图、背景、光线、镜头和画幅比例不变。",
                    tooltip="List everything that must remain consistent with the main image.",
                ),
                io.String.Input(
                    id="avoid",
                    multiline=True,
                    default="不要增加无关主体，不要改变未指定区域，不要生成水印或乱码文字。",
                    tooltip="List unwanted transfers, extra subjects, text, or other failure modes.",
                ),
            ],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, *, instruction, image_roles, preserve, avoid):
        return io.NodeOutput(build_structured_edit_prompt(instruction, image_roles, preserve, avoid))


class SenseNovaEditGuiderImpl(comfy.samplers.CFGGuider):
    def set_conds(self, positive, image_condition, negative):
        image_condition = node_helpers.conditioning_set_values(image_condition, {"prompt_type": "negative"})
        negative = node_helpers.conditioning_set_values(negative, {"prompt_type": "negative"})
        self.inner_set_conds(
            {
                "positive": positive,
                "image_condition": image_condition,
                "negative": negative,
            }
        )

    def set_cfg(self, cfg, img_cfg, cfg_norm="none", cfg_interval_start=0.0, cfg_interval_end=1.0):
        if cfg_norm not in CFG_NORM_MODES:
            raise ValueError(f"unsupported SenseNova CFG norm mode: {cfg_norm}")
        if not 0.0 <= cfg_interval_start <= cfg_interval_end <= 1.0:
            raise ValueError("SenseNova CFG interval must satisfy 0 <= start <= end <= 1")
        self.cfg = cfg
        self.img_cfg = img_cfg
        self.cfg_norm = cfg_norm
        self.cfg_interval_start = cfg_interval_start
        self.cfg_interval_end = cfg_interval_end

    def _uses_cfg(self, timestep):
        sigma = float(torch.as_tensor(timestep).flatten()[0])
        progress = 1.0 - sigma
        return self.cfg_interval_start <= progress <= self.cfg_interval_end

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive = self.conds.get("positive")
        image_condition = self.conds.get("image_condition")
        negative = self.conds.get("negative")

        if not self._uses_cfg(timestep):
            return comfy.samplers.calc_cond_batch(self.inner_model, [positive], x, timestep, model_options)[0]

        if math.isclose(self.cfg, 1.0) and math.isclose(self.img_cfg, 1.0):
            return comfy.samplers.calc_cond_batch(self.inner_model, [positive], x, timestep, model_options)[0]
        if math.isclose(self.img_cfg, 1.0):
            image_out, positive_out = comfy.samplers.calc_cond_batch(
                self.inner_model, [image_condition, positive], x, timestep, model_options
            )
            guided = image_out + self.cfg * (positive_out - image_out)
        elif math.isclose(self.cfg, self.img_cfg):
            negative_out, positive_out = comfy.samplers.calc_cond_batch(
                self.inner_model, [negative, positive], x, timestep, model_options
            )
            guided = negative_out + self.cfg * (positive_out - negative_out)
        else:
            negative_out, image_out, positive_out = comfy.samplers.calc_cond_batch(
                self.inner_model, [negative, image_condition, positive], x, timestep, model_options
            )
            guided = edit_guidance(positive_out, image_out, negative_out, self.cfg, self.img_cfg)
        norm_mode = self.cfg_norm if self.cfg > 1.0 or self.img_cfg > 1.0 else "none"
        return rescale_denoised_guidance(guided, positive_out, x, timestep, mode=norm_mode)


class SenseNovaEditGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaEditGuider",
            display_name="SenseNova Edit Guider",
            category="sampling/custom_sampling/guiders/SenseNova",
            description="Three-branch SenseNova editing guidance for SamplerCustomAdvanced.",
            inputs=[
                io.Model.Input(id="model"),
                io.Conditioning.Input(id="positive"),
                io.Conditioning.Input(id="image_condition"),
                io.Conditioning.Input(id="negative"),
                io.Float.Input(
                    id="cfg",
                    default=4.0,
                    min=0.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Text instruction strength. Start with 4.",
                ),
                io.Float.Input(
                    id="img_cfg",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Reference-image guidance strength. Start with 1; this is different from text CFG.",
                ),
                io.Combo.Input(
                    id="cfg_norm",
                    options=list(CFG_NORM_MODES),
                    default="none",
                    tooltip="Limit guidance overshoot. Try global for complex edits or overly saturated results.",
                ),
                io.Float.Input(
                    id="cfg_interval_start",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Normalized denoising progress where CFG starts. 0 means the first step.",
                ),
                io.Float.Input(
                    id="cfg_interval_end",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Normalized denoising progress where CFG stops. 1 means the final step.",
                ),
            ],
            outputs=[io.Guider.Output()],
        )

    @classmethod
    def execute(
        cls,
        *,
        model,
        positive,
        image_condition,
        negative,
        cfg,
        img_cfg,
        cfg_norm="none",
        cfg_interval_start=0.0,
        cfg_interval_end=1.0,
    ):
        guider = SenseNovaEditGuiderImpl(model)
        guider.set_conds(positive, image_condition, negative)
        guider.set_cfg(cfg, img_cfg, cfg_norm, cfg_interval_start, cfg_interval_end)
        return io.NodeOutput(guider)


class SenseNovaExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [
            SenseNovaU15Loader,
            SenseNovaU15GGUFLoader,
            SenseNovaU15EightStepLoRA,
            EmptySenseNovaLatentImage,
            SenseNovaSamplingOptions,
            SenseNovaReferenceImage,
            SenseNovaReferenceImageAdvanced,
            SenseNovaStructuredEditPrompt,
            SenseNovaEditGuider,
        ]


async def comfy_entrypoint():
    return SenseNovaExtension()
