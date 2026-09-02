import math

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.supported_models_base

from .model import HEAD_DIM, MERGED_PATCH_SIZE, NUM_KV_HEADS, NUM_LAYERS, THINK_SUFFIX_TOKEN_IDS, SenseNovaU15
from .conditioning import block_causal_mask, condition_input_ids, conditioned_input_length, preprocess_references, smart_resize, thw_indexes
from .sampling import SenseNovaModelSampling, time_snr_shift
from .text_encoder import SenseNovaTextEncoder, SenseNovaTokenizer


class CONDSharedRegular(comfy.conds.CONDRegular):
    """Keep a prefix tensor at one copy per guidance branch.

    ComfyUI normally repeats every condition to the latent batch. SenseNova's
    text/reference prefix is identical for all generated variants, so the model
    computes it once and expands only the much smaller per-layer KV tensors.
    """

    def process_cond(self, batch_size, **kwargs):
        return self._copy_with(self.cond)


class CONDSharedList(comfy.conds.CONDList):
    """List counterpart to :class:`CONDSharedRegular` for reference images."""

    def process_cond(self, batch_size, **kwargs):
        return self._copy_with(self.cond)


class SenseNovaBaseModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, device=None):
        super().__init__(
            model_config,
            comfy.model_base.ModelType.FLOW,
            device=device,
            unet_model=SenseNovaU15,
        )
        self.model_sampling = SenseNovaModelSampling(model_config)
        self.memory_usage_factor_conds = ("reference_images",)

    def process_timestep(self, timestep, **kwargs):
        base_timestep = timestep / self.model_sampling.multiplier
        return 1.0 - time_snr_shift(self.model_sampling.shift, 1.0 - base_timestep)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        prefix_keys = kwargs.get("prefix_keys")
        if prefix_keys is not None:
            out["prefix_keys"] = CONDSharedList(prefix_keys)
            out["prefix_values"] = CONDSharedList(kwargs["prefix_values"])
            out["prefix_time"] = CONDSharedRegular(kwargs["prefix_time"])
            return out

        text_input_ids = kwargs.get("text_input_ids")
        if text_input_ids is not None:
            reference_images = kwargs.get("sensenova_reference_images")
            image_only = (
                kwargs.get("sensenova_reference_mode") == "image_only"
                or kwargs.get("prompt_type") == "negative"
            )
            thinking = bool(kwargs.get("sensenova_thinking", False)) and not image_only
            thinking_result = kwargs.get("sensenova_thinking_result")
            if reference_images:
                reference_images = preprocess_references(reference_images)
                reference_grids = [
                    (
                        max(1, math.ceil(image.shape[-2] / MERGED_PATCH_SIZE)),
                        max(1, math.ceil(image.shape[-1] / MERGED_PATCH_SIZE)),
                    )
                    for image in reference_images
                ]
                text_input_ids = condition_input_ids(
                    text_input_ids,
                    reference_grids,
                    image_only=image_only,
                )
                indexes = thw_indexes(text_input_ids, reference_grids)
                out["prefix_indexes"] = CONDSharedRegular(indexes)
                out["prefix_mask"] = CONDSharedRegular(
                    block_causal_mask(indexes, dtype=self.get_dtype_inference())
                )
                out["reference_images"] = CONDSharedList(reference_images)
            out["text_input_ids"] = CONDSharedRegular(text_input_ids)
            if thinking:
                out["sensenova_thinking"] = comfy.conds.CONDConstant(True)
                out["sensenova_max_think_tokens"] = comfy.conds.CONDConstant(
                    int(kwargs.get("sensenova_max_think_tokens", 1024))
                )
                if isinstance(thinking_result, dict):
                    out["sensenova_thinking_result"] = comfy.conds.CONDConstant(thinking_result)
        return out

    def extra_conds_shapes(self, **kwargs):
        images = kwargs.get("sensenova_reference_images")
        images = images if images is not None else []
        resized = []
        if images:
            max_pixels = min(2048 * 2048, (4096 * 4096) // len(images))
            resized = [smart_resize(*image.shape[1:3], max_pixels=max_pixels) for image in images]
        reference_grids = [
            (
                max(1, math.ceil(height / MERGED_PATCH_SIZE)),
                max(1, math.ceil(width / MERGED_PATCH_SIZE)),
            )
            for height, width in resized
        ]
        out = {}
        if resized:
            out["reference_images"] = [1, 3, sum(height * width for height, width in resized)]
        text_input_ids = kwargs.get("text_input_ids")
        if text_input_ids is not None:
            image_only = (
                kwargs.get("sensenova_reference_mode") == "image_only"
                or kwargs.get("prompt_type") == "negative"
            )
            length = text_input_ids.shape[1]
            if reference_grids:
                length = conditioned_input_length(
                    length,
                    reference_grids,
                    image_only=image_only,
                )
            out["prefix_mask"] = [1, 1, length, length]
            thinking = bool(kwargs.get("sensenova_thinking", False)) and not image_only
            if thinking:
                length += int(kwargs.get("sensenova_max_think_tokens", 1024))
                length += 1 + len(THINK_SUFFIX_TOKEN_IDS)
            prefix_shape = [1, NUM_KV_HEADS, NUM_LAYERS * length * HEAD_DIM]
            out["prefix_keys"] = prefix_shape
            out["prefix_values"] = prefix_shape
        return out

    def memory_required(self, input_shape, cond_shapes={}):
        memory = super().memory_required(input_shape, cond_shapes)
        dtype_size = comfy.model_management.dtype_size(self.get_dtype_inference())
        return memory + sum(
            math.prod(shape) * dtype_size
            for key in ("prefix_mask", "prefix_keys", "prefix_values")
            for shape in cond_shapes.get(key, ())
        )


class SenseNovaModelConfig(comfy.supported_models_base.BASE):
    unet_config = {"image_model": "sensenova_u15", "has_lm_head": True}
    sampling_settings = {"shift": 3.0, "noise_scale": 1.0}
    latent_format = comfy.latent_formats.HiDreamO1Pixel
    memory_usage_factor = 0.033
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    optimizations = {"fp8": False}

    def get_model(self, state_dict, prefix="", device=None):
        return SenseNovaBaseModel(self, device=device)

    def process_vae_state_dict(self, state_dict):
        return {"pixel_space_vae": torch.tensor(1.0)}

    def clip_target(self, state_dict={}):
        return comfy.supported_models_base.ClipTarget(SenseNovaTokenizer, SenseNovaTextEncoder)
