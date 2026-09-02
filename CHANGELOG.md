# Changelog

本文件记录 ComfyUI 节点本身的版本变化。模型权重的下载和说明见 Hugging Face 模型页。

## [1.5.0] - 2026-09-02

- 将 ComfyUI core PR [#16032](https://github.com/Comfy-Org/ComfyUI/pull/16032) 的 SenseNova thinking 与交错文本/图像生成能力适配到本节点；无需修改本地 ComfyUI core。
- 新增 `SenseNova 1.x Text Encode` 和 `SenseNova Thinking Preview`，支持普通图像模式、thinking 模式和 interleave 模式，可限制最大思考 token 数并在采样后查看实际思考文本。
- 新增 `SenseNova 1.x Interleave`：在同一次会话中生成文本和多张图，生成图会重新编码进正/负 KV 前缀，供后续文本与图像继续生成。
- 新增顺序化 `SenseNova Interleave Preview`，按生成顺序显示文本、可选 thinking 和图片，同时输出 Markdown。
- 保留并加载官方 checkpoint 中的 `language_model.lm_head.weight`；Final、SFT 与五种 GGUF Loader 共用原有严格校验、动态卸载和采样期 KV cache 管道。
- 新增 thinking 文生图与 interleave 故事工作流，以及思考闭合、双前缀、多图回填、输出顺序和前端节点协议测试。

## [1.4.1] - 2026-09-02

- 根据上游确认的结论，补充“编辑 SenseNova 自生成图片时应更换 seed”的中英文说明，避免复用文生图 seed 引起分布偏移和画面崩坏。
- 同步更新参考图节点和 ComfyUI-Manager 节点清单的帮助文字。

## [1.4.0] - 2026-09-01

- 新增独立的 `SenseNova U1.5 GGUF Loader (Final)`，支持社区发布并经官方 README 收录的 Q2_K、Q3_K_M、Q5_K_M、Q6_K、Q8_0 Final 量化权重。
- GGUF 模型继续输出标准 MODEL / CLIP / VAE，支持原有文生图、图像编辑、采样选项、显存卸载和 8-step LoRA 管道，不依赖另一个自定义节点。
- 加载前严格检查扩展名、精确文件大小、SHA256、1116 个 tensor 名称、shape 与量化类型；加入逐量化块参考对照测试。
- 将 GGUF 中唯一的 F16 词嵌入输出统一为 BF16 计算精度，避免前缀 FP16 KV 与 BF16 生成分支拼接后提升为 FP32。
- 新增 GGUF 文生图与图像编辑工作流，并补充双语安装、模型目录和格式说明。

## [1.3.8] - 2026-09-01

- 同步已合并的 ComfyUI core SenseNova U1.5 实现，更新原生文生图与图像编辑工作流。
- core 工作流改用官方复用的 `EmptyHiDreamO1LatentImage` 和 `HiDreamO1ReferenceImages` 节点，并增加接口回归测试。

## [1.3.7] - 2026-08-31

- 修复使用 PyTorch SDPA attention 后端时，prefix attention mask 保持 FP32、而 Q/K/V 为 BF16 所导致的有限但数值错误的 attention 输出和异常生成结果。
- prefix mask 现在会在调用 ComfyUI attention 后端前转换为 query dtype，并新增对应的 dtype 回归测试。

## [1.3.6] - 2026-08-27

- 添加官方 ComfyUI-Manager 使用的 `node_list.json`，让 V3 扩展入口注册的全部 8 个节点能被“安装缺失节点”功能可靠识别。
- 增加节点清单与 V3 schema ID 的一致性测试，防止新增或重命名节点时遗漏 Manager 映射。

## [1.3.5] - 2026-08-26

- 支持官方 revision `19bc874e` 的全 BF16 Final 权重及约 35 GB 的新版 ComfyUI 单文件，同时继续严格校验并兼容原有约 50 GB 的混合精度 Final；两者都可使用现有 8-step LoRA。
- `Empty SenseNova Pixel Latent` 新增官方建议的 1:1、16:9、9:16、2:3、3:2 分辨率预设，保留原有自定义宽高和旧工作流输入顺序。
- CI 更新到 ComfyUI v0.34.0，覆盖 Python 3.10～3.14，并加入 Ruff 静态检查。
- 添加 GitHub Bug/Feature Issue 表单与 GitHub Actions Dependabot 更新配置；主分支启用必需 CI、禁止强推和删除保护。

## [1.3.4] - 2026-08-25

- 修复 CUDA 13 / Blackwell 环境启用 `comfy-kitchen` CUDA 后端时，split-half RoPE 可能返回有限但错误的数值，导致生成结果严重偏色、过饱和和结构异常的问题；语言层 RoPE 现在固定使用与官方一致的 PyTorch 参考公式。
- 视觉 RoPE 改用 `comfy-kitchen` 支持的标准 4D 输入和 6D rotation 布局，启用 `--enable-triton-backend` 时不再因 3D tensor 解包失败。
- 新增后端隔离和 accelerated-backend tensor rank 回归测试。

## [1.3.3] - 2026-08-24

- 修复部分环境加载官方 Final/SFT 单文件时，把 `timestep_embedder` 和 `noise_scale_embedder` 错报为多余键的问题（#1、#2）。
- 节点现在随包携带从正式 Final/SFT 转换清单生成的固定 1116 tensor contract，不再根据当前 PyTorch/ComfyUI 环境临时推导权重结构。
- 加载前检查模型 metadata、文件大小、全部 tensor 名称、shape 和各版本 dtype；错误信息会显示实际模型与 loader 路径，便于发现旧节点或重复安装。
- CI 不再因缺少本地大模型 manifest 而跳过结构测试，并新增 Python 3.13 + ComfyUI 0.33.1 组合。

## [1.3.2] - 2026-08-24

- CI 同时验证最低支持的 ComfyUI 0.31 和当前稳定版 ComfyUI 0.33.1。
- Registry 发布后自动写入对应版本的更新说明。
- 补全 Registry、模型下载、问题反馈和发布记录入口。
- 同步 Hugging Face 模卡中的仓库地址、批量生成和多参考图说明。

本版本不修改模型加载、采样或图像生成逻辑，推理结果与 1.3.1 一致。

## [1.3.1] - 2026-08-22

- 修复普通参考图节点的插槽命名和默认工作流连线。
- 旧工作流导入时自动迁移历史插槽名称。
- 批量生成时复用文本和参考图 prefix，降低重复计算。

## [1.3.0] - 2026-08-22

- 支持一次生成 1～16 张不同结果。
- 新增 CFG Norm、CFG 生效区间和编辑提示词整理节点。
- 新增 1～10 张参考图的高级节点。
