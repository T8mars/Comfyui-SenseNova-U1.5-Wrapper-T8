# SenseNova U1.5 for ComfyUI

English | [简体中文](README.md)

[![CI](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/actions/workflows/ci.yml/badge.svg)](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/actions/workflows/ci.yml)

[Changelog](CHANGELOG.md) · [GitHub Releases](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/releases)

Native ComfyUI nodes for SenseNova U1.5. The model, sampler, scheduler, VRAM offloading, LoRA loading, and workflows all use ComfyUI's native pipeline.

Supported features:

- Text-to-image generation
- Thinking image generation, with model reasoning before diffusion sampling
- Interleaved text/image generation with generated-image feedback into later turns
- Single-image editing
- Multi-reference editing with 1 to 10 images
- Generate 1 to 16 different results from the same prompt and references
- Standard ComfyUI `KSampler`
- Official U1.5 Final and U1.5 SFT checkpoints
- Q2_K, Q3_K_M, Q5_K_M, Q6_K, and Q8_0 GGUF quantizations of U1.5 Final
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

GGUF support uses `gguf>=0.13.0`, which Manager and the Comfy CLI install automatically. After a manual clone, run `pip install "gguf>=0.13.0"` if the package is missing.

## Download the models

- [Hugging Face: t8star/SenseNova-U1.5-Comfy](https://huggingface.co/t8star/SenseNova-U1.5-Comfy/)
- [Hugging Face: realrebelai/SenseNova-U1.5-8B_GGUFs](https://huggingface.co/realrebelai/SenseNova-U1.5-8B_GGUFs)
- [Quark model mirror](https://pan.quark.cn/s/6b756fdae32d)

Download only the files you need:

| File | Place it in | Purpose |
|---|---|---|
| `SenseNova-U1.5-8B-MoT-BF16-T8.safetensors` | `ComfyUI/models/diffusion_models/` | Current all-BF16 U1.5 Final single file, about 35 GB; recommended |
| `SenseNova-U1.5-8B-MoT-T8.safetensors` | `ComfyUI/models/diffusion_models/` | Legacy mixed-precision U1.5 Final single file, about 50 GB; still supported |
| `SenseNova-U1.5-8B-MoT-SFT-T8.safetensors` | `ComfyUI/models/diffusion_models/` | U1.5 SFT single-file checkpoint, about 35 GB |
| `SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors` | `ComfyUI/models/loras/` | ComfyUI-native conversion of the official 8-step LoRA, about 815 MB |
| `SenseNova-U1.5-8B-MoT-Q2_K.gguf` | `ComfyUI/models/gguf/` | Final Q2_K, about 9.26 GB |
| `SenseNova-U1.5-8B-MoT-Q3_K_M.gguf` | `ComfyUI/models/gguf/` | Final Q3_K_M, about 10.92 GB; suggested low-VRAM starting point |
| `SenseNova-U1.5-8B-MoT-Q5_K_M.gguf` | `ComfyUI/models/gguf/` | Final Q5_K_M, about 15.11 GB |
| `SenseNova-U1.5-8B-MoT-Q6_K.gguf` | `ComfyUI/models/gguf/` | Final Q6_K, about 17.24 GB |
| `SenseNova-U1.5-8B-MoT-Q8_0.gguf` | `ComfyUI/models/gguf/` | Final Q8_0, about 21.17 GB; quality-first quantization |

Base-model directory:

```text
ComfyUI/models/diffusion_models/
```

LoRA directory:

```text
ComfyUI/models/loras/
```

GGUF directory:

```text
ComfyUI/models/gguf/
```

ComfyUI-Manager installs the nodes only. It does not download these model files.

Final and SFT are both SenseNova U1.5 checkpoints. The current BF16 Final is the official all-BF16 conversion and re-shard of the same Final model; this node strictly supports both the current 35 GB Final and the legacy 50 GB Final. SFT is a separate training-stage checkpoint.

The official 8-step LoRA must be used with Final. Do not apply it to SFT or Preview. The dedicated `SenseNova U1.5 8-Step LoRA` node checks the base model and gives a clear error if the combination is invalid.

| File or combination | Supported | Notes |
|---|---:|---|
| U1.5 Final BF16, 50-step generation/editing | ✅ | Current recommended checkpoint, about 35 GB |
| Legacy mixed-precision U1.5 Final, 50-step generation/editing | ✅ | Existing downloads remain supported, about 50 GB |
| U1.5 Final + `-ComfyUI` 8-step LoRA | ✅ | 8-step text-to-image only |
| U1.5 Final GGUF, 50-step generation/editing | ✅ | Five quantizations verified by exact size, SHA256, tensor names, and shapes |
| U1.5 Final GGUF + `-ComfyUI` 8-step LoRA | ✅ | Native ModelPatcher path; validate the environment at 50 steps first |
| U1.5 SFT, 50-step generation/editing | ✅ | Standalone checkpoint; do not add the 8-step LoRA |
| U1.5 Preview | ❌ | Older preview checkpoint |
| Unconverted official raw LoRA | ❌ | Use the `-ComfyUI` file or convert it with the included tool |

## Ready-to-use workflows

These are normal ComfyUI canvas workflows. Download a JSON file and drag it onto the ComfyUI canvas. There are no API-format workflows in this repository. For editing workflows, select your own image in each `Load Image` node after importing.

- [Text-to-image](examples/t2i_workflow.json)
- [Thinking text-to-image](examples/thinking_t2i_workflow.json)
- [Interleaved text/image generation](examples/interleave_workflow.json)
- [GGUF text-to-image](examples/gguf_t2i_workflow.json)
- [GGUF image editing](examples/gguf_edit_workflow.json)
- [GGUF live-validation record](docs/gguf-validation.md)
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

This repository now adapts the thinking and interleave implementation from [ComfyUI core PR #16032](https://github.com/Comfy-Org/ComfyUI/pull/16032) to the MODEL / CLIP / VAE outputs of its own loaders. PR #16032 is still open, so these two workflows do not require patching or switching your local ComfyUI core. Once the core PR lands, both paths use the same prompt protocol and generation logic.

The GGUF workflows use this repository's `SenseNova U1.5 GGUF Loader (Final)`. It dequantizes weights on demand while keeping MODEL, CLIP, VAE, sampling, scheduling, LoRA, and VRAM offloading on ComfyUI's native interfaces. It does not require the separate ComfyUI-GGUF custom node. No Q4 file is currently published in the linked model repository, so Q4 is not listed as a verified download.

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

### Thinking and interleaved generation

For thinking image generation, replace `CLIP Text Encode` with `SenseNova 1.x Text Encode`, select `mode=image`, and enable `thinking`. Start with `max_think_tokens=512`; keep thinking disabled on the negative prompt. Connect the same positive conditioning and the KSampler samples to `SenseNova Thinking Preview` to inspect the reasoning text after sampling completes. The model performs autoregressive reasoning before diffusion sampling, so the first image takes additional time and KV memory compared with ordinary text-to-image generation.

For interleaved output, encode both positive and negative prompts with `mode=interleave`, then connect them to `SenseNova 1.x Interleave`. The node uses standard `KSamplerSelect`, `BasicScheduler`, and a pixel latent. Whenever the model emits an image event, the node samples that image, appends it to both live KV prefixes, and continues generating later text or images. `max_images` caps the images in one session. Decode the latent batch with `VAE Decode` and connect it, together with the structured result, to `SenseNova Interleave Preview` to preserve the original output order.

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

If the reference was just generated by SenseNova text-to-image, set KSampler to a seed different from the one used to create that image. Upstream confirmed that reusing the same seed can cause a distribution shift and a collapsed edit. If changing the seed does not resolve it, try the nearest official resolution preset. See the [upstream explanation](https://github.com/OpenSenseNova/SenseNova-U1/issues/278#issuecomment-5503345718).

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

### Q6_K GGUF: 512×512, 50-step text-to-image

[Open the original 512×512 PNG](docs/images/result-gguf-q6-t2i-512.png)

![SenseNova U1.5 Q6_K GGUF glass teapot](docs/images/result-gguf-q6-t2i-512.png)

Q6_K GGUF, 512×512, 50 steps, CFG 4, shift 3, Euler/normal, seed 424242. On an RTX 5090 Laptop GPU with 24 GB VRAM and PyTorch 2.7.0 + CUDA 12.8, the warmed-up sampler took 288 seconds, or about 5.7 seconds per step. Device usage approached 24 GB. The first run also pays a substantial quantized-kernel cold-start cost; Q3 and Q6 differ sharply in both speed and quality, so file size alone should not determine the choice.

### Q6_K GGUF + official LoRA: 512×512, 8-step text-to-image

[Open the original 512×512 PNG](docs/images/result-gguf-q6-lora-8step-512.png)

![SenseNova U1.5 Q6_K GGUF 8-step LoRA glass teapot](docs/images/result-gguf-q6-lora-8step-512.png)

Q6_K GGUF, official `-ComfyUI` 8-step LoRA, 512×512, 8 steps, CFG 1, shift 3, Euler/normal, seed 424242. All 294 LoRA patches were applied through the native ModelPatcher path, and core sampling took 19.9 seconds. See the [GGUF live-validation record](docs/gguf-validation.md) for the complete Q3/Q6/Q8 comparison and test boundaries.

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
- Real Q3_K_M, Q6_K, and Q8_0 GGUF SHA256 verification, the complete 1116-tensor contract, full model loading, T2I, Q6 image editing, and the official 8-step LoRA path

2048×2048 50-step text-to-image generation, dual-reference editing, and `512×512, batch_size=2` full-model batch execution all completed on 24 GB VRAM. Loading and offloading the model also uses substantial system memory. 64 GB RAM and enough virtual memory are recommended.

## Current limitations

- Only NVIDIA CUDA with BF16 has been fully validated.
- Models are not downloaded automatically at runtime.
- Bbox/marker controls are not exposed yet.
- Complex subject replacement, multi-region edits, and heavily constrained edits can drift.
- Small diagonal or vertical text can degrade during editing; upstream has confirmed this as a [model limitation targeted for improvement](https://github.com/OpenSenseNova/SenseNova-U1/issues/275).
- Reusing the same seed when editing a SenseNova-generated image can cause a distribution shift and a collapsed result. Follow the [upstream-confirmed workaround](https://github.com/OpenSenseNova/SenseNova-U1/issues/278#issuecomment-5503345718) and change the seed; if the problem remains, try the nearest official resolution preset.
- Q8_0 ran close to the VRAM limit on the 24 GB test machine. Q2_K and Q5_K_M have strict file and tensor validation but were not part of this live-sampling matrix.
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

Final GGUF files (source revision `bc2e8f83688489e6b465daa833e9b318ea45c9d9`):

| File | Exact size (bytes) | SHA256 |
|---|---:|---|
| `SenseNova-U1.5-8B-MoT-Q2_K.gguf` | 9,264,536,960 | `98f947928474f45e4c0c149f1af6009f15f99abd524b4dd36e2324d29303f2e5` |
| `SenseNova-U1.5-8B-MoT-Q3_K_M.gguf` | 10,920,713,600 | `82ccb1ee4cfd24d605ecaa97c99f799eef7bb78577185b0c1662d3d83c399636` |
| `SenseNova-U1.5-8B-MoT-Q5_K_M.gguf` | 15,107,169,664 | `1c496256eb114a5ff8fef278a63b39a75bd0c36e76f4280e89a06bb6ecb76ade` |
| `SenseNova-U1.5-8B-MoT-Q6_K.gguf` | 17,240,972,672 | `ded187014c0e34e13d20702d426a1741e9ec2aa698f3466df95ca0116d0e5ea2` |
| `SenseNova-U1.5-8B-MoT-Q8_0.gguf` | 21,174,689,152 | `61b227f036b7e8094cceab888c23b17a3fffc32d6182b039836d7cb31d688fe2` |

The loader distinguishes current Final, legacy Final, SFT, and the five GGUF profiles. It checks metadata, exact size, SHA256, every tensor name and shape, and allowed storage or quantization types. Invalid, incomplete, and unsupported checkpoints fail with a clear error instead of loading silently.

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
