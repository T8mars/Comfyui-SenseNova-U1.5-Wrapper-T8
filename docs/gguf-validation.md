# SenseNova U1.5 GGUF validation

This document records the checks used for the native GGUF loader in v1.4.0. It is intended to make the support claim reproducible and to separate file-format validation from image-quality testing.

## Source and integrity

- Source: [`realrebelai/SenseNova-U1.5-8B_GGUFs`](https://huggingface.co/realrebelai/SenseNova-U1.5-8B_GGUFs)
- Pinned revision: `bc2e8f83688489e6b465daa833e9b318ea45c9d9`
- The loader accepts only the five exact file sizes and SHA256 values listed in the README.
- Every file must match the Final checkpoint's complete 1116-tensor name and shape contract before model construction.
- Dequantization for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0 is compared block-for-block against the `gguf` reference implementation in the no-weight test suite.

Q4 dequantization is covered because Q3_K_M contains Q4_K tensors. The source repository does not currently publish a standalone Q4 checkpoint.

## Real-file validation

The following files were downloaded from the pinned revision and checked locally. “Live run” means a complete ComfyUI API workflow including tokenizer, MODEL, CLIP, sampling, VAE decoding, and image saving—not merely parsing the GGUF header.

| Profile | SHA256 and size | Tensor contract | Full model load | Live run |
| --- | --- | --- | --- | --- |
| Q3_K_M | pass | 1116 tensors: 550 Q3_K, 40 Q4_K, 2 Q5_K, 523 F32, 1 F16 | pass | 512×512 text-to-image, 1 step |
| Q6_K | pass | 1116 tensors: 592 Q6_K, 523 F32, 1 F16 | pass | 1-step and 50-step text-to-image; 1-step image editing; official 8-step LoRA |
| Q8_0 | pass | 1116 tensors: 592 Q8_0, 523 F32, 1 F16 | pass | 512×512 text-to-image, 1 step |

Q2_K and Q5_K_M are admitted only through their pinned size/SHA256 profiles and the same strict tensor-contract checks. They were not included in this 24 GB GPU live-run matrix.

## Reproducible diagnostic comparison

Hardware and runtime:

- RTX 5090 Laptop GPU, 24 GB VRAM
- 64 GB system RAM
- ComfyUI 0.33.0
- Python 3.10.10
- PyTorch 2.7.0 + CUDA 12.8
- `gguf` 0.14.0

All four diagnostic runs used the same 512×512 prompt, seed 424242, CFG 4, shift 3, Euler/normal, and one sampling step. A one-step image is intentionally under-denoised; this is a numerical path comparison, not a quality benchmark.

| Checkpoint | MSE vs BF16 | MAE vs BF16 | PSNR | SSIM | Core sampling |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q3_K_M | 0.028373 | 0.137218 | 15.47 dB | 0.6531 | 196.65 s cold |
| Q6_K | 0.000692 | 0.018963 | 31.60 dB | 0.8876 | 110.39 s cold |
| Q8_0 | 0.000602 | 0.017464 | 32.21 dB | 0.8895 | 72.14 s |

The Q6_K 50-step quality run used the same prompt and seed. After quantized operations were warm, core sampling took 288 seconds (about 5.7 seconds per step), with device usage approaching 24 GB. The output is stored at [`docs/images/result-gguf-q6-t2i-512.png`](images/result-gguf-q6-t2i-512.png).

The official `-ComfyUI` 8-step LoRA was also applied to Q6_K through the native `ModelPatcher` path. All 294 LoRA patches were accepted, and a full 8-step workflow completed in 19.9 seconds of core sampling. The output is stored at [`docs/images/result-gguf-q6-lora-8step-512.png`](images/result-gguf-q6-lora-8step-512.png).

## Interpretation and limits

- Q6_K is the strongest tested balance on the 24 GB validation machine. Q8_0 ran successfully but stayed near the VRAM limit.
- Cold-start cost is substantial because weights are dequantized on demand. Warm throughput should not be inferred from first-run latency.
- The BF16 50-step comparison was not completed on this 24 GB system because the approximately 35 GB model streams between RAM and VRAM. The published one-step metrics and Q6_K 50-step image are deliberately reported as separate tests.
- CUDA + BF16 is the only fully exercised device path. FP16, ROCm, MPS, DirectML, XPU, and NPU remain unverified.
