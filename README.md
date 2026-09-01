# SenseNova-U1.5 ComfyUI 节点

[English](README_EN.md) | 简体中文

[![CI](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/actions/workflows/ci.yml/badge.svg)](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/actions/workflows/ci.yml)

[版本更新记录](CHANGELOG.md) · [GitHub Releases](https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8/releases)

这是 SenseNova-U1.5 的 ComfyUI 原生节点。模型、采样器、调度器、显存卸载和工作流都走 ComfyUI 管道，支持：

- 文生图
- 单图编辑
- 1～10 张参考图编辑
- 同一提示词/参考图一次生成 1～16 个不同结果
- 普通 `KSampler`
- U1.5 Final 和 U1.5 SFT 两套官方权重
- U1.5 Final 的 Q2_K / Q3_K_M / Q5_K_M / Q6_K / Q8_0 GGUF 量化权重
- 官方 U1.5 8-step LoRA（底层使用 ComfyUI 原生 LoRA/ModelPatcher 管道）
- 自定义 `img_cfg` 的三路引导、CFG Norm 和 CFG 生效区间
- 用明确的“修改 / 参考图职责 / 保持 / 禁止”结构整理复杂编辑提示词
- 执行期间的文本/参考图 prefix KV cache

节点只读取本地模型，运行时不会联网下载文件。

## 安装

最简单的方法是在 ComfyUI-Manager 里搜索 `SenseNova U1.5 (T8)`，安装后重启 ComfyUI。

- Registry：[sensenova-u15-t8](https://registry.comfy.org/nodes/sensenova-u15-t8)
- Comfy CLI：`comfy node install sensenova-u15-t8`

也可以手动安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-SenseNova-U1.5-Wrapper-T8.git
```

GGUF 支持使用 `gguf>=0.13.0`；通过 Manager 或 Comfy CLI 安装时会自动安装。手动克隆后如缺少依赖，可运行 `pip install "gguf>=0.13.0"`。

## 下载模型

- [Hugging Face：t8star/SenseNova-U1.5-Comfy](https://huggingface.co/t8star/SenseNova-U1.5-Comfy/)
- [Hugging Face：realrebelai/SenseNova-U1.5-8B_GGUFs](https://huggingface.co/realrebelai/SenseNova-U1.5-8B_GGUFs)
- [模型网盘](https://pan.quark.cn/s/6b756fdae32d)

按需要下载：

| 文件 | 放置位置 | 用途 |
|---|---|---|
| `SenseNova-U1.5-8B-MoT-BF16-T8.safetensors` | `ComfyUI/models/diffusion_models/` | U1.5 Final 新版全 BF16 单文件，约 35 GB，推荐下载 |
| `SenseNova-U1.5-8B-MoT-T8.safetensors` | `ComfyUI/models/diffusion_models/` | U1.5 Final 旧版混合精度单文件，约 50 GB，继续兼容 |
| `SenseNova-U1.5-8B-MoT-SFT-T8.safetensors` | `ComfyUI/models/diffusion_models/` | U1.5 SFT 单文件底模，约 35 GB |
| `SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors` | `ComfyUI/models/loras/` | 官方 8-step LoRA 的 ComfyUI 原生键名版本，约 815 MB |
| `SenseNova-U1.5-8B-MoT-Q2_K.gguf` | `ComfyUI/models/gguf/` | Final Q2_K，约 9.26 GB |
| `SenseNova-U1.5-8B-MoT-Q3_K_M.gguf` | `ComfyUI/models/gguf/` | Final Q3_K_M，约 10.92 GB，低显存建议起点 |
| `SenseNova-U1.5-8B-MoT-Q5_K_M.gguf` | `ComfyUI/models/gguf/` | Final Q5_K_M，约 15.11 GB |
| `SenseNova-U1.5-8B-MoT-Q6_K.gguf` | `ComfyUI/models/gguf/` | Final Q6_K，约 17.24 GB |
| `SenseNova-U1.5-8B-MoT-Q8_0.gguf` | `ComfyUI/models/gguf/` | Final Q8_0，约 21.17 GB，量化质量优先 |

底模路径：

```text
ComfyUI/models/diffusion_models/
```

LoRA 路径：

```text
ComfyUI/models/loras/
```

GGUF 路径：

```text
ComfyUI/models/gguf/
```

Manager 只安装节点，不会自动下载模型。

Final 和 SFT 都是 SenseNova U1.5，本节点都支持 50 步文生图和图像编辑。新版 BF16 Final 是官方在相同 Final 模型上进行的全 BF16 转换和重新分片；节点同时严格支持新版 35 GB Final 和旧版 50 GB Final。SFT 是不同训练阶段的独立权重，不要混为同一个文件。

注意：官方 8-step LoRA 必须搭配 Final，不能搭配 SFT 或 Preview。专用的 `SenseNova U1.5 8-Step LoRA` 节点会检查底模，接错时直接给出说明。

| 文件/组合 | 本节点支持 | 说明 |
|---|---:|---|
| U1.5 Final BF16，50 步生成/编辑 | ✅ | 当前推荐模型，约 35 GB |
| U1.5 Final 旧版混合精度，50 步生成/编辑 | ✅ | 兼容已有下载，约 50 GB |
| U1.5 Final + `-ComfyUI` 8-step LoRA | ✅ | 仅用于 8 步文生图 |
| U1.5 Final GGUF，50 步生成/编辑 | ✅ | 五种经过文件大小、SHA256、tensor 名称与 shape 严格校验的量化文件 |
| U1.5 Final GGUF + `-ComfyUI` 8-step LoRA | ✅ | 原生 ModelPatcher 路径；仍建议先用 50 步验证环境 |
| U1.5 SFT，50 步生成/编辑 | ✅ | 独立单文件底模，不叠加 8-step LoRA |
| U1.5 Preview | ❌ | 旧预览权重 |
| 官方未转换的 raw LoRA | ❌ | 先使用仓库转换工具，或直接下载 `-ComfyUI` 文件 |

## 直接使用工作流

下面都是 ComfyUI 画布工作流，下载 JSON 后可以直接拖进 ComfyUI。没有 API 工作流。编辑工作流打开后，先在 `Load Image` 中选择自己的图片。

- [文生图工作流](examples/t2i_workflow.json)
- [GGUF 文生图工作流](examples/gguf_t2i_workflow.json)
- [GGUF 图像编辑工作流](examples/gguf_edit_workflow.json)
- [GGUF 实机验证记录](docs/gguf-validation.md)
- [批量文生图工作流（默认一次 2 张）](examples/batch_t2i_workflow.json)
- [8-step LoRA 文生图工作流](examples/t2i_8step_workflow.json)
- [普通编辑工作流（img_cfg=1）](examples/edit_workflow.json)
- [稳定多参考编辑工作流（人物换装案例）](examples/multi_reference_edit_workflow.json)
- [SFT 文生图工作流](examples/sft_t2i_workflow.json)
- [SFT 图像编辑工作流](examples/sft_edit_workflow.json)

### ComfyUI core 原生工作流

下面两个工作流用于已包含 SenseNova U1.5 core 支持的 ComfyUI，不依赖本仓库的自定义 Loader：

- [core 原生文生图工作流](examples/core_t2i_workflow.json)
- [core 原生图像编辑工作流](examples/core_edit_workflow.json)

core 工作流使用 ComfyUI 自带的 `CheckpointLoaderSimple`，因此底模要放到 `ComfyUI/models/checkpoints/`。8-step LoRA 可直接使用自带的 `LoraLoaderModelOnly`，LoRA 文件仍放在 `ComfyUI/models/loras/`。
合并后的 core 实现复用 `EmptyHiDreamO1LatentImage` 和 `HiDreamO1ReferenceImages`，因此请使用包含 [ComfyUI PR #15922](https://github.com/Comfy-Org/ComfyUI/pull/15922) 的版本。

GGUF 工作流使用本仓库的 `SenseNova U1.5 GGUF Loader (Final)`。Loader 会按需解量化权重，但 MODEL、CLIP、VAE、采样器、调度器、LoRA 和显存卸载仍使用 ComfyUI 原生接口；不要求另外安装 ComfyUI-GGUF 自定义节点。Q4 文件目前没有在上述模型仓库实际发布，因此不会显示为受验证的下载选项。

推荐先保持这些参数：

```text
steps: 50
CFG: 4
img_cfg: 1
shift: 3
sampler: euler
scheduler: normal
denoise: 1
```

`Empty SenseNova Pixel Latent` 还提供官方建议的分辨率预设；选择 `Custom` 时继续使用节点上的 width 和 height：

| 比例 | 分辨率 |
|---|---:|
| 1:1 | 2048 × 2048 |
| 16:9 | 2720 × 1536 |
| 9:16 | 1536 × 2720 |
| 2:3 | 1664 × 2496 |
| 3:2 | 2496 × 1664 |

复杂编辑建议在 `SenseNova Edit Guider` 中先用：

```text
CFG: 4
img_cfg: 1
cfg_norm: global
cfg_interval: 0 → 1
```

`global` 会把过强的引导幅度拉回正向条件的范围，通常能减轻高饱和、过度锐化和主体漂移；如果结果变得太保守，再切回 `none`。`channel` 按 32×32 生成 token 分别归一化，适合局部区域容易过冲的场景。

`cfg_interval` 使用 ComfyUI 的归一化去噪进度，`0` 是第一步、`1` 是最后一步，起止点都包含在区间内。这里有意让 start 和 end 始终都生效，避免官方参考代码在 `start=0` 时忽略 end 的边界问题；保持 `0 → 1` 就是官方默认的全程 CFG。

8-step LoRA 请用官方参数：

```text
LoRA strength: 1
steps: 8
CFG: 1
cfg_norm: none（CFG=1 时不做额外 CFG norm）
shift: 3
sampler: euler
scheduler: normal
denoise: 1
```

### 文生图

![SenseNova 文生图工作流](docs/images/t2i-workflow.jpg)

连接顺序很简单：

```text
Loader → Sampling Options → KSampler → VAE Decode → Save Image
```

`MODEL` 必须先经过 `SenseNova Sampling Options`，`shift` 保持 `3`。

### 批量生成

把 `Empty SenseNova Pixel Latent` 的 `batch_size` 改成 `2～16`，同一个提示词会产生多张不同结果，`Save Image` 会逐张保存。批量编辑也走同一接口，所有结果共用同一组参考图；目前完整实测范围是 `512×512、batch_size=2`，更大的编辑批量需要根据参考图数量和显存逐步增加。

显存开销会随批量增加。示例工作流默认用 `768×768、batch_size=2`；不要直接用 `2048×2048、batch_size=16`。24 GB 显存建议从 `512/768、batch_size=2` 开始。

双参考换装的 `512×512、batch_size=2、50 步` 完整编辑实测用时约 495 秒，两张结果不同且都遵守换装要求；任务期间整张显卡显存采样峰值为 22,986 MiB。

### 8-step LoRA 文生图

8-step 工作流使用本项目的保护节点，内部仍走 ComfyUI 原生 LoRA 映射和 `ModelPatcher`：

```text
SenseNova Loader (Final) → SenseNova U1.5 8-Step LoRA → Sampling Options → KSampler
```

LoRA 强度保持 `1`。这个 LoRA 是官方发布的快速文生图适配器；图像编辑仍建议使用不加 LoRA 的 50 步编辑工作流。

### 普通图像编辑

![SenseNova 普通编辑工作流](docs/images/edit-workflow.jpg)

参考图要接到 `SenseNova Reference Image`，不要把参考图当作 latent。`img_cfg=1` 时，可以继续用普通 `KSampler`；把节点输出的 `image_condition` 接到 KSampler 的 negative。

### 多参考图和自定义引导

普通的 `SenseNova Reference Image` 节点只显示 `Image-1` 和可选的 `Image-2`，不会再多出一个容易误接的空白第三插槽。需要 3～10 张图时，改用 `SenseNova Reference Images (1-10)` 节点。图像顺序就是提示词中的 `Image-1`、`Image-2`。人物换装时，`Image-1` 放人物主图，`Image-2` 放服装图。旧版工作流中的 `images.image` 名称会在导入时自动迁移；旧工作流使用 3 张以上参考图时，也会自动切换到 1～10 张版本。

复杂任务不要只写“让她穿上这件衣服”。可以使用 `SenseNova Structured Edit Prompt` 节点，把要求拆成四项：主要修改、每张参考图的职责、必须保持的内容、禁止出现的内容；也可以直接照下面的格式写进 `CLIP Text Encode`：

```text
【主要修改】让 Image-1 的人物穿上 Image-2 的服装。
【参考图职责】Image-1 只提供人物；Image-2 只提供服装，不复制人台和背景。
【必须保持】保持 Image-1 的脸、姿势、光线、背景和画幅不变。
【禁止出现】不要增加第二个人，不要改变未指定区域。
```

当 `img_cfg` 不是 1 时，要使用 `SenseNova Edit Guider` 和 ComfyUI 自带的 `SamplerCustomAdvanced`。最重要的一点：`Sampling Options` 输出的同一个 MODEL，要同时连接 `Edit Guider` 和 `BasicScheduler`。

```text
SenseNova Sampling Options (MODEL)
├── SenseNova Edit Guider ───────────┐
└── BasicScheduler ──────────────────┤
RandomNoise + KSamplerSelect + Latent├──→ SamplerCustomAdvanced
                                     ┘
```

节点会按官方规则给多张图插入 `Image-1`、`Image-2` 等标签，并分别处理尺寸，不会把多张图片简单拼接。稳定工作流已经使用 `CFG 4、img_cfg 1、global CFG Norm`；它优先保留人物身份和原始构图，不会为了“改得更多”盲目把 `img_cfg` 拉高。

## 实际结果

下面图片都由本节点生成，参数不是后期调色结果。

### Q6_K GGUF：512×512、50 步文生图

[查看原始 512×512 PNG](docs/images/result-gguf-q6-t2i-512.png)

![SenseNova U1.5 Q6_K GGUF 玻璃茶壶](docs/images/result-gguf-q6-t2i-512.png)

参数：Q6_K GGUF、512×512、50 步、CFG 4、shift 3、Euler/normal、seed 424242。RTX 5090 Laptop 24 GB、PyTorch 2.7.0 + CUDA 12.8 上，量化算子预热后的核心采样为 288 秒，稳定约 5.7 秒/步；峰值设备占用接近 24 GB。首次运行还会有明显的量化算子冷启动开销，Q3/Q6 的速度和质量差异都很大，不应只按文件大小选择。

### Q6_K GGUF + 官方 LoRA：512×512、8 步文生图

[查看原始 512×512 PNG](docs/images/result-gguf-q6-lora-8step-512.png)

![SenseNova U1.5 Q6_K GGUF 8-step LoRA 玻璃茶壶](docs/images/result-gguf-q6-lora-8step-512.png)

参数：Q6_K GGUF、官方 `-ComfyUI` 8-step LoRA、512×512、8 步、CFG 1、shift 3、Euler/normal、seed 424242。294 个 LoRA patch 全部通过原生 ModelPatcher 应用，核心采样为 19.9 秒。完整的 Q3/Q6/Q8 对照数据和测试边界见 [GGUF 实机验证记录](docs/gguf-validation.md)。

### 2048×2048 双参考人物换装

[查看原始 2048×2048 PNG](docs/images/result-garment-edit-2048.png)

![SenseNova U1.5 双参考人物换装](docs/images/result-garment-edit-2048.png)

参数：Final 单文件底模、2048×2048、50 步、CFG 4、img_cfg 1、global CFG Norm、shift 3、Euler/normal、seed 31082026。Image-1 提供人物、脸、姿势和背景，Image-2 只提供黑白裙装；输出保留了托腮手势与室内构图，并迁移了白色围裙、荷叶边、蝴蝶结和袖口。没有后期修图，RTX 5090 Laptop 24 GB 上任务约 506 秒完成。

### U1.5 SFT：2048×2048、50 步文字密集文生图

[查看原始 2048×2048 PNG](docs/images/result-sft-t2i-2048.png)

![SenseNova U1.5 SFT 中文炸鸡信息图](docs/images/result-sft-t2i-2048.png)

参数：SFT 单文件底模、2048×2048、50 步、CFG 4、shift 3、Euler/normal、seed 42。标题、材料数量、3 个步骤和 170°C 提示直接由模型生成，没有后期修字。RTX 5090 Laptop 24 GB 上任务约 297 秒完成。

### 2048×2048、8 步文字密集文生图

[查看原始 2048×2048 PNG](docs/images/result-t2i-8step-2048.png)

![SenseNova U1.5 8-step 中文炸鸡信息图](docs/images/result-t2i-8step-2048.png)

参数：2048×2048、8 步、CFG 1、shift 3、LoRA strength 1、Euler/normal、seed 42。标题、副标题、5 项材料、3 个步骤和温度提示均直接由模型生成，没有后期修字。RTX 5090 Laptop 24 GB 上任务约 86 秒完成。

### 2048×2048 文生图

[查看原始 2048×2048 PNG](docs/images/result-t2i-2048.png)

![SenseNova 2048 文生图结果](docs/images/result-t2i-2048.png)

参数：2048×2048、50 步、CFG 4、shift 3、Euler/normal、seed 42。24 GB 显存实测完成。

### 2048×2048 双参考图编辑

[查看原始 2048×2048 PNG](docs/images/result-multi-reference-2048.png)

![SenseNova 2048 双参考图结果](docs/images/result-multi-reference-2048.png)

参数：2048×2048、50 步、CFG 4、img_cfg 1、shift 3、Euler/normal、seed 42。第一张图提供手账版式和文字密度，第二张图提供炸鸡主体；提示词明确要求标题、材料、三步做法和小贴士。大标题、材料和主要步骤可读，局部小字仍有错字和重叠，本图没有后期修字。24 GB 显存实测完成。

## KV cache 做了什么

SenseNova 的文字和参考图 prefix 在每一步都相同。`SenseNova Sampling Options` 会在一次采样任务内缓存它们，后续 step 直接复用，避免重复计算参考图。批量生成时，文字和参考图 prefix 也只按每个引导分支计算一份，再把每层较小的 KV 扩展到各个结果，不会把整套参考图编码重复 `batch_size` 次。

缓存只存在于当前任务中；任务完成、报错或取消时都会清空，不会跨任务保存，也不会偷偷占用长期显存。缓存与无缓存的三路编辑 A/B 测试结果逐元素一致。

## 颜色太艳怎么办

先检查提示词里有没有 `bright`、`vivid`、`neon`、`highly saturated`。这些词会明显提高饱和度。建议：

- `CFG` 先用 4，不满意再试 3～3.5
- `img_cfg` 先保持 1
- 复杂编辑或画面过冲时把 `cfg_norm` 改成 `global`
- 提示词加入 `natural colors`、`restrained color grading`

## 运行要求

当前实机和 CI 验证范围：

- 实机 ComfyUI `v0.33.x`
- CI：最低支持的 ComfyUI `0.31`，以及当前稳定版 `v0.34.0`
- Python `3.10`、`3.12`、`3.13`、`3.14`
- NVIDIA CUDA + BF16
- RTX 5090 Laptop 24 GB
- 64 GB 系统内存
- 真实 Q3_K_M / Q6_K / Q8_0 GGUF 文件的 SHA256、1116 tensor 契约、完整模型加载、T2I、Q6 图像编辑与官方 8-step LoRA 路径

2048×2048、50 步文生图和双参考图编辑，以及 `512×512、batch_size=2` 的完整模型批量执行，都能在 24 GB 显存下完成。模型加载和卸载还会占用较多系统内存，建议准备 64 GB RAM 和足够的虚拟内存。

## 当前限制

- 只验证了 NVIDIA CUDA + BF16
- 不支持运行时自动下载模型
- bbox/marker 和 think mode 暂未开放
- 复杂主体替换、多区域或多约束编辑可能出现内容漂移
- 斜排、竖排的小字在编辑中可能损坏，这是上游已确认会继续改进的[模型限制](https://github.com/OpenSenseNova/SenseNova-U1/issues/275)
- 上游正在调查个别“保持原分辨率”编辑崩坏的[案例](https://github.com/OpenSenseNova/SenseNova-U1/issues/278)；遇到时可先改用最接近的官方分辨率桶，并保留工作流、输入图和 seed 反馈
- Q8_0 在 24 GB 实测机上接近显存上限；Q2_K / Q5_K_M 已纳入严格文件与 tensor 校验，但尚未进入本次实机采样矩阵
- FP16、ROCm、MPS、DirectML、XPU、NPU 暂未验证

## 模型校验

Final BF16（推荐）：

```text
文件：SenseNova-U1.5-8B-MoT-BF16-T8.safetensors
大小：35,065,860,328 bytes
SHA256：a32b117f40ad4575c6709b3ad6efb1c6b743ef1c1c3d75360f14090b997f1d29
官方 revision：19bc874ef6ffc97fda9837b40fc1d1301806158a
tensor：1116（全部 BF16）
```

Final 旧版混合精度（继续兼容）：

```text
文件：SenseNova-U1.5-8B-MoT-T8.safetensors
大小：50,222,155,152 bytes
SHA256：2e5c4451969a8af9d7bcbf9d00a0fe463b15ed44149d8d79f31409e671587615
tensor：1116
revision：1f6ec60423d29939dde4202fd82ae340b144e280
```

SFT：

```text
大小：35,065,860,320 bytes
SHA256：9c105bb4baaf244bbd99f814c36f190228c5878f8889295e3dba285441442f2f
tensor：1116（全部 BF16）
revision：661834c5b5aee0f89958353511d6ac0ccaacb646
```

Final GGUF（来源 revision `bc2e8f83688489e6b465daa833e9b318ea45c9d9`）：

| 文件 | 精确大小（bytes） | SHA256 |
|---|---:|---|
| `SenseNova-U1.5-8B-MoT-Q2_K.gguf` | 9,264,536,960 | `98f947928474f45e4c0c149f1af6009f15f99abd524b4dd36e2324d29303f2e5` |
| `SenseNova-U1.5-8B-MoT-Q3_K_M.gguf` | 10,920,713,600 | `82ccb1ee4cfd24d605ecaa97c99f799eef7bb78577185b0c1662d3d83c399636` |
| `SenseNova-U1.5-8B-MoT-Q5_K_M.gguf` | 15,107,169,664 | `1c496256eb114a5ff8fef278a63b39a75bd0c36e76f4280e89a06bb6ecb76ade` |
| `SenseNova-U1.5-8B-MoT-Q6_K.gguf` | 17,240,972,672 | `ded187014c0e34e13d20702d426a1741e9ec2aa698f3466df95ca0116d0e5ea2` |
| `SenseNova-U1.5-8B-MoT-Q8_0.gguf` | 21,174,689,152 | `61b227f036b7e8094cceab888c23b17a3fffc32d6182b039836d7cb31d688fe2` |

节点会区分新版 Final、旧版 Final、SFT 和五种 GGUF，并检查 metadata、精确大小、SHA256、全部 tensor 名称、shape 以及允许的存储/量化类型。如果下载不完整或版本不对，会直接报错，不会静默加载错误权重。

### 出现 `checkpoint key mismatch` 怎么办

先把节点更新到 `1.3.5` 或更高版本，然后彻底关闭并重启 ComfyUI。不要通过修改 loader、关闭动态加载或删除报错键来绕过校验，这可能让模型虽然能运行，但输出模糊、偏色或不遵循提示词。

如果更新后仍报错：

- 检查 `ComfyUI/custom_nodes/` 下是否装了两份本节点，旧目录也会被 ComfyUI 导入。
- 对照上面的大小和 SHA256，确认底模是本项目发布的 Final 或 SFT 单文件。
- 保留完整报错；新版错误会同时显示实际 `model=` 和 `loader=` 路径，可直接看出 ComfyUI 加载的是哪一份文件。

8-step LoRA 校验：

```text
官方来源：sensenova/SenseNova-U1.5-8B-MoT-LoRAs
revision：e909f4636d119d65fe4cba8770c19daff2ac102e
官方文件 SHA256：3ef32180cdf1e30a870a83f4f136e897ea50b7ee467f863d75633464ebb25708
ComfyUI 文件 SHA256：dd5320f06986688dd41b0a4a2cb6ebd0036308f8a8a2d0c349ca22875a805aa1
module：294
tensor：882
```

转换只给键名添加 `diffusion_model.` 前缀，LoRA 张量数据逐字节不变。普通用户直接下载转换好的文件即可；从 GitHub 克隆源码的高级用户也可以运行 [`tools/convert_lora_to_comfy.py`](tools/convert_lora_to_comfy.py)。

需要手动检查下载文件时：

```powershell
Get-FileHash .\SenseNova-U1.5-8B-MoT-BF16-T8.safetensors -Algorithm SHA256
Get-FileHash .\SenseNova-U1.5-8B-MoT-T8.safetensors -Algorithm SHA256
Get-FileHash .\SenseNova-U1.5-8B-MoT-SFT-T8.safetensors -Algorithm SHA256
Get-FileHash .\SenseNova-U1.5-8B-MoT-LoRA-8step-ComfyUI.safetensors -Algorithm SHA256
```

## 其他链接

- [B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [AI API](https://api.seedance.nz/sign-up?aff=5f4w)
- [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
- [模型网盘](https://pan.quark.cn/s/6b756fdae32d)
- [Hugging Face 主页](https://huggingface.co/t8star)

## 来源与许可

SenseNova-U1.5 模型和参考实现来自 [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)，原项目使用 Apache License 2.0。

- [官方 U1.5 Final](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)
- [官方 U1.5 SFT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-SFT)
- [官方 U1.5 LoRAs](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs)

本仓库只提供 ComfyUI 本地推理适配，不包含模型权重。详细归因见 [NOTICE](NOTICE)。
