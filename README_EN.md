# SenseNova U1.5 for ComfyUI

English | [简体中文](README.md)

[![CI](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/actions/workflows/ci.yml/badge.svg)](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/actions/workflows/ci.yml)

[Changelog](CHANGELOG.md) · [GitHub Releases](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/releases)

Native ComfyUI nodes for SenseNova U1.5. The model, sampler, scheduler, VRAM offloading, LoRA loading, and workflows all use ComfyUI's native pipeline.

Supported features:

- Text-to-image generation
- Single-image editing
- Multi-reference editing with 1 to 10 images
- Generate 1 to 16 different results from the same prompt and references
- Standard ComfyUI `KSampler`
- Official U1.5 Final and U1.5 SFT checkpoints
- Official U1.5 8-step LoRA through ComfyUI's native LoRA and `ModelPatcher` pipeline
- Three-branch guidance with a separate `img_cfg`
- CFG Norm and configurable CFG intervals
- A structured prompt helper for complex image-editing tasks
- Execution-local text and reference-image prefix KV cache

The nodes only read local model files. They never download models while ComfyUI is running.

## Installation

The easiest option is to search for `SenseNova U1.5 (T8)` in ComfyUI-Manager, install it, and restart ComfyUI.

- Registry: [sensenova-u15-t8](https://registry.comfy.org/nodes/sensenova-u15-t8)
- Comfy CLI: `comfy node install sensenova-u15-t8`

Manual installation:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8.git
```

This custom node has no extra Python dependencies.

## Download the models

- [Hugging Face: t8star/SenseNova-U1.5-Comfy](https://huggingface.co/t8star/SenseNova-U1.5-Comfy/)
- [Quark model mirror](https://pan.quark.cn/s/6b756fdae32d)

Download only the files you need:

| File | Place it in | Purpose |
|---|---|---|
| `SenseNova-U1.5-8B-MoT-BF16-T8.safetensors` | `ComfyUI/models/diffusion_models/` | Current all-BF16 U1.5 Final single file, about 35 GB; recommended |
| `SenseNova-U1.5-8B-MoT-T8.safetensors` | `ComfyUI/models/diffusion_models/` | Legacy mixed-precision U1.5 Final single file, about 50 GB; still supported |
| `SenseNova-U1.5-8B-MoT-SFT-T8.safetensors` | `ComfyUI/models/diffusion_models/` | U1.5 SFT single-file checkpoint, about 35 GB |
| `SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors` | `ComfyUI/models/loras/` | ComfyUI-native conversion of the official 8-step LoRA, about 815 MB |

Base-model directory:

```text
ComfyUI/models/diffusion_models/
```

LoRA directory:

```text
ComfyUI/models/loras/
```

ComfyUI-Manager installs the nodes only. It does not download these model files.

Final and SFT are both SenseNova U1.5 checkpoints. The current BF16 Final is the official all-BF16 conversion and re-shard of the same Final model; this node strictly supports both the current 35 GB Final and the legacy 50 GB Final. SFT is a separate training-stage checkpoint.

The official 8-step LoRA must be used with Final. Do not apply it to SFT or Preview. The dedicated `SenseNova U1.5 8-Step LoRA` node checks the base model and gives a clear error if the combination is invalid.

| File or combination | Supported | Notes |
|---|---:|---|
| U1.5 Final BF16, 50-step generation/editing | ✅ | Current recommended checkpoint, about 35 GB |
| Legacy mixed-precision U1.5 Final, 50-step generation/editing | ✅ | Existing downloads remain supported, about 50 GB |
| U1.5 Final + `-ComfyUI` 8-step LoRA | ✅ | 8-step text-to-image only |
| U1.5 SFT, 50-step generation/editing | ✅ | Standalone checkpoint; do not add the 8-step LoRA |
| U1.5 Preview | ❌ | Older preview checkpoint |
| Unconverted official raw LoRA | ❌ | Use the `-ComfyUI` file or convert it with the included tool |

## Ready-to-use workflows

These are normal ComfyUI canvas workflows. Download a JSON file and drag it onto the ComfyUI canvas. There are no API-format workflows in this repository. For editing workflows, select your own image in each `Load Image` node after importing.

- [Text-to-image](examples/t2i_workflow.json)
- [Batch text-to-image, two results by default](examples/batch_t2i_workflow.json)
- [8-step LoRA text-to-image](examples/t2i_8step_workflow.json)
- [Standard image editing, img_cfg=1](examples/edit_workflow.json)
- [Stable multi-reference editing, virtual try-on example](examples/multi_reference_edit_workflow.json)
- [SFT text-to-image](examples/sft_t2i_workflow.json)
- [SFT image editing](examples/sft_edit_workflow.json)

### Native ComfyUI core workflows

These two workflows target ComfyUI builds that include native SenseNova U1.5 core support and do not depend on this repository's custom loader:

- [Native core text-to-image](examples/core_t2i_workflow.json)
- [Native core image editing](examples/core_edit_workflow.json)

The core workflows use ComfyUI's built-in `CheckpointLoaderSimple`, so place the base checkpoint in `ComfyUI/models/checkpoints/`. The 8-step LoRA can use the built-in `LoraLoaderModelOnly`; keep the LoRA file in `ComfyUI/models/loras/`.
The merged core implementation reuses `EmptyHiDreamO1LatentImage` and `HiDreamO1ReferenceImages`, so use a ComfyUI build that includes [PR #15922](https://github.com/Comfy-Org/ComfyUI/pull/15922).

Start with these settings:

```text
steps: 50
CFG: 4
img_cfg: 1
shift: 3
sampler: euler
scheduler: normal
denoise: 1
```

`Empty SenseNova Pixel Latent` also exposes the official suggested resolution presets. Select `Custom` to keep using the width and height fields:

| Aspect ratio | Resolution |
|---|---:|
| 1:1 | 2048 × 2048 |
| 16:9 | 2720 × 1536 |
| 9:16 | 1536 × 2720 |
| 2:3 | 1664 × 2496 |
| 3:2 | 2496 × 1664 |

For more complex editing, begin with these values in `SenseNova Edit Guider`:

```text
CFG: 4
img_cfg: 1
cfg_norm: global
cfg_interval: 0 → 1
```

`global` CFG Norm pulls excessive guidance back toward the magnitude of the positive condition. It often reduces oversaturation, over-sharpening, and subject drift. Switch back to `none` if the edit becomes too conservative. `channel` normalizes each 32×32 generation token independently and can help with localized over-guidance.

`cfg_interval` uses ComfyUI's normalized denoising progress: `0` is the first step and `1` is the last. Both boundaries are inclusive. Keep `0 → 1` for full-time CFG, which matches the official default behavior.

Use the official settings for the 8-step LoRA:

```text
LoRA strength: 1
steps: 8
CFG: 1
cfg_norm: none
shift: 3
sampler: euler
scheduler: normal
denoise: 1
```

### Text-to-image

![SenseNova text-to-image workflow](docs/images/t2i-workflow.jpg)

The basic connection is:

```text
Loader → Sampling Options → KSampler → VAE Decode → Save Image
```

Always pass `MODEL` through `SenseNova Sampling Options` before sampling. Keep `shift` at `3` unless you intentionally want to experiment.

### Batch generation

Set `batch_size` in `Empty SenseNova Pixel Latent` to a value from `2` to `16`. Each result uses different noise while sharing the same prompt and reference set. `Save Image` saves every result separately.

VRAM use increases with batch size. The batch example uses `768×768, batch_size=2`. Do not begin with `2048×2048, batch_size=16`. On a 24 GB GPU, start with 512 or 768 pixels and a batch size of 2.

A full dual-reference editing test at `512×512, batch_size=2, 50 steps` took about 495 seconds. Both results followed the clothing-transfer instruction, and total GPU memory usage peaked at 22,986 MiB.

### 8-step LoRA text-to-image

The protected 8-step node still uses ComfyUI's native LoRA mapping and `ModelPatcher` internally:

```text
SenseNova Loader (Final) → SenseNova U1.5 8-Step LoRA → Sampling Options → KSampler
```

Keep LoRA strength at `1`. The official LoRA is intended for fast text-to-image generation. Use the regular 50-step workflow without the LoRA for image editing.

### Standard image editing

![SenseNova standard editing workflow](docs/images/edit-workflow.jpg)

Connect the reference image to `SenseNova Reference Image`; do not use it as the latent input. When `img_cfg=1`, the standard `KSampler` works. Connect the node's `image_condition` output to KSampler's negative input.

### Multiple references and custom guidance

The standard `SenseNova Reference Image` node exposes `Image-1` and an optional `Image-2`. Use `SenseNova Reference Images (1-10)` when you need 3 to 10 inputs.

Image order matches the labels used in the prompt. For virtual try-on or clothing transfer:

- Put the person or main scene in `Image-1`.
- Put the garment reference in `Image-2`.

Older workflows using legacy `images.image` socket names are migrated automatically. Older workflows with more than two references are also migrated to the 1-to-10-image node.

For complex edits, avoid vague prompts such as “make her wear this.” Use `SenseNova Structured Edit Prompt`, or write the same structure directly in `CLIP Text Encode`:

```text
[Main change] Make the person in Image-1 wear the garment from Image-2.
[Reference roles] Image-1 provides only the person; Image-2 provides only the garment. Do not copy the mannequin or background from Image-2.
[Must preserve] Keep the face, pose, lighting, background, and framing from Image-1 unchanged.
[Must avoid] Do not add another person and do not change unspecified regions.
```

When `img_cfg` is not 1, use `SenseNova Edit Guider` with ComfyUI's built-in `SamplerCustomAdvanced`. The same `MODEL` output from `Sampling Options` must be connected to both `Edit Guider` and `BasicScheduler`:

```text
SenseNova Sampling Options (MODEL)
├── SenseNova Edit Guider ───────────┐
└── BasicScheduler ──────────────────┤
RandomNoise + KSamplerSelect + Latent├──→ SamplerCustomAdvanced
                                     ┘
```

The node inserts official `Image-1`, `Image-2`, and later labels, and processes every reference at its own supported size. It does not concatenate the images into one strip.

## Real results

All images below were generated by this custom node. They were not color-graded or retouched afterward.

### 2048×2048 dual-reference clothing transfer

[Open the original 2048×2048 PNG](docs/images/result-garment-edit-2048.png)

![SenseNova U1.5 dual-reference clothing transfer](docs/images/result-garment-edit-2048.png)

Final checkpoint, 2048×2048, 50 steps, CFG 4, img_cfg 1, global CFG Norm, shift 3, Euler/normal, seed 31082026. Image-1 supplied the person, face, pose, and background; Image-2 supplied only the black-and-white garment. The result preserved the hand-on-chin pose and indoor composition while transferring the apron, ruffles, bow, and cuffs. It completed in about 506 seconds on an RTX 5090 Laptop GPU with 24 GB VRAM.

### U1.5 SFT: 2048×2048, 50-step text-heavy generation

[Open the original 2048×2048 PNG](docs/images/result-sft-t2i-2048.png)

![SenseNova U1.5 SFT Chinese fried-chicken infographic](docs/images/result-sft-t2i-2048.png)

SFT checkpoint, 2048×2048, 50 steps, CFG 4, shift 3, Euler/normal, seed 42. The title, ingredient amounts, three steps, and 170°C note were generated directly by the model without text correction. The run took about 297 seconds on an RTX 5090 Laptop GPU with 24 GB VRAM.

### 2048×2048, 8-step text-heavy generation

[Open the original 2048×2048 PNG](docs/images/result-t2i-8step-2048.png)

![SenseNova U1.5 8-step Chinese fried-chicken infographic](docs/images/result-t2i-8step-2048.png)

2048×2048, 8 steps, CFG 1, shift 3, LoRA strength 1, Euler/normal, seed 42. The title, subtitle, five ingredients, three steps, and temperature note were all generated by the model. The run took about 86 seconds on an RTX 5090 Laptop GPU with 24 GB VRAM.

### 2048×2048 text-to-image

[Open the original 2048×2048 PNG](docs/images/result-t2i-2048.png)

![SenseNova 2048 text-to-image result](docs/images/result-t2i-2048.png)

2048×2048, 50 steps, CFG 4, shift 3, Euler/normal, seed 42. Successfully tested on 24 GB VRAM.

### 2048×2048 dual-reference editing

[Open the original 2048×2048 PNG](docs/images/result-multi-reference-2048.png)

![SenseNova 2048 dual-reference result](docs/images/result-multi-reference-2048.png)

2048×2048, 50 steps, CFG 4, img_cfg 1, shift 3, Euler/normal, seed 42. Image-1 supplied the notebook layout and text density; Image-2 supplied the fried-chicken subject. The prompt requested a title, ingredients, three steps, and a tip. The large title and main sections are readable, while some small text still contains spelling errors and overlaps. The image was not corrected afterward.

## What the KV cache does

SenseNova uses the same text and reference-image prefix at every denoising step. `SenseNova Sampling Options` caches those prefix keys and values for the current execution, so later steps do not encode the same references again.

For batch generation, the text and reference prefix is computed once per guidance branch, and only the smaller per-layer KV data is expanded across generated variants. The complete reference-image encoder is not repeated `batch_size` times.

The cache exists only during the current job. It is cleared when the job finishes, fails, or is cancelled, so it does not keep VRAM allocated between jobs. Cached and uncached three-branch editing paths were verified to be element-wise identical.

## What if the colors are too saturated?

First check the prompt for words such as `bright`, `vivid`, `neon`, or `highly saturated`. They can strongly increase saturation.

Try the following:

- Start with `CFG 4`; try 3 to 3.5 if the image still looks over-guided.
- Keep `img_cfg` at 1 initially.
- Use `global` CFG Norm for complex edits or overcooked-looking images.
- Add `natural colors` or `restrained color grading` to the prompt.

## System requirements

Local and CI validation coverage:

- Local ComfyUI 0.33.x
- CI: minimum supported ComfyUI 0.31 and current stable ComfyUI v0.34.0
- Python 3.10, 3.12, 3.13, and 3.14
- NVIDIA CUDA with BF16 support
- RTX 5090 Laptop GPU, 24 GB VRAM
- 64 GB system RAM

2048×2048 50-step text-to-image generation, dual-reference editing, and `512×512, batch_size=2` full-model batch execution all completed on 24 GB VRAM. Loading and offloading the model also uses substantial system memory. 64 GB RAM and enough virtual memory are recommended.

## Current limitations

- Only NVIDIA CUDA with BF16 has been fully validated.
- Models are not downloaded automatically at runtime.
- Quantization, bbox/marker controls, and think mode are not exposed yet.
- Complex subject replacement, multi-region edits, and heavily constrained edits can drift.
- FP16, ROCm, MPS, DirectML, XPU, and NPU have not been validated.

## Model verification

Current BF16 Final checkpoint (recommended):

```text
File: SenseNova-U1.5-8B-MoT-BF16-T8.safetensors
Size: 35,065,860,328 bytes
SHA256: a32b117f40ad4575c6709b3ad6efb1c6b743ef1c1c3d75360f14090b997f1d29
Official revision: 19bc874ef6ffc97fda9837b40fc1d1301806158a
Tensors: 1116, all stored as BF16
```

Legacy mixed-precision Final checkpoint (still supported):

```text
File: SenseNova-U1.5-8B-MoT-T8.safetensors
Size: 50,222,155,152 bytes
SHA256: 2e5c4451969a8af9d7bcbf9d00a0fe463b15ed44149d8d79f31409e671587615
Tensors: 1116
Source revision: 1f6ec60423d29939dde4202fd82ae340b144e280
```

SFT checkpoint:

```text
Size: 35,065,860,320 bytes
SHA256: 9c105bb4baaf244bbd99f814c36f190228c5878f8889295e3dba285441442f2f
Tensors: 1116, all stored as BF16
Source revision: 661834c5b5aee0f89958353511d6ac0ccaacb646
```

The loader distinguishes current Final, legacy Final, and SFT files and checks metadata, exact file size, all tensor names, shapes, and each profile's storage dtype. Invalid, incomplete, and unsupported checkpoints fail with a clear error instead of loading silently.

### If you see `checkpoint key mismatch`

Update this custom node to version `1.3.5` or newer, completely close ComfyUI, and start it again. Do not modify the loader, remove reported keys, or disable dynamic model loading to bypass verification. Those workarounds may allow the model to run with incorrect weights, causing blurred output, unusual colors, or poor prompt following.

If the error remains:

- Check `ComfyUI/custom_nodes/` for duplicate copies of this custom node.
- Compare your model's exact size and SHA256 with the values above.
- Keep the full error message. Newer versions include the actual `model=` and `loader=` paths, which make stale installations easy to identify.

8-step LoRA verification:

```text
Official source: sensenova/SenseNova-U1.5-8B-MoT-LoRAs
Source revision: e909f4636d119d65fe4cba8770c19daff2ac102e
Official file SHA256: 3ef32180cdf1e30a870a83f4f136e897ea50b7ee467f863d75633464ebb25708
ComfyUI file SHA256: dd5320f06986688dd41b0a4a2cb6ebd0036308f8a8a2d0c349ca22875a805aa1
Modules: 294
Tensors: 882
```

The conversion only adds the `diffusion_model.` prefix required by ComfyUI. All LoRA tensor data remains byte-for-byte identical. Most users should download the converted file. Advanced users who cloned the source repository can also run [`tools/convert_lora_to_comfy.py`](tools/convert_lora_to_comfy.py).

Manual hash checks on Windows:

```powershell
Get-FileHash .\SenseNova-U1.5-8B-MoT-BF16-T8.safetensors -Algorithm SHA256
Get-FileHash .\SenseNova-U1.5-8B-MoT-T8.safetensors -Algorithm SHA256
Get-FileHash .\SenseNova-U1.5-8B-MoT-SFT-T8.safetensors -Algorithm SHA256
Get-FileHash .\SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors -Algorithm SHA256
```

## Other links

- [Bilibili](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [AI API](https://api.seedance.nz/sign-up?aff=5f4w)
- [Online AI applications](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI package](https://pan.quark.cn/s/264edb7e36bd)
- [Model mirror](https://pan.quark.cn/s/6b756fdae32d)
- [T8star on Hugging Face](https://huggingface.co/t8star)

## Source and license

SenseNova U1.5 and its reference implementation come from [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1), licensed under Apache License 2.0.

- [Official U1.5 Final](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)
- [Official U1.5 SFT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-SFT)
- [Official U1.5 LoRAs](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs)

This repository provides the local ComfyUI integration only. It does not contain model weights. See [NOTICE](NOTICE) for full attribution.
