# VoxCPM2 Swift / MLX 实现

> ⚠️ **Work in Progress**: 这是一个最小化骨架实现，已注册到 TTS 工厂并可以编译通过。核心的前向推理（forward / generate）和子模块（MiniCPM4、LocEnc、DiT、Audio VAE）尚未实现。

---

## 模型架构（基于 config.json + Python 参考实现）

VoxCPM2 是一个 **tokenizer-free、diffusion autoregressive** TTS 模型，由以下核心组件构成：

### 1. Base LM — 文本/语义大语言模型
- **架构**: MiniCPM4 (与 Qwen3/Llama 结构相似)
- **规模**: 28 层, hidden_size=2048, 16 heads, GQA (2 kv heads)
- **位置编码**: LongRoPE (`rope_scaling.type == "longrope"`)
- **特殊配置**: `scale_emb=12`, `dim_model_base=256`, `scale_depth=1.4`, `use_mup=false`
- **词表**: 73,448

### 2. Residual LM — 残差声学语言模型
- **架构**: MiniCPM4 (共享大部分超参)
- **规模**: 8 层 (`residual_lm_num_layers: 8`)
- **特性**: 无词表 (`vocab_size=0`)，无 RoPE (`no_rope: true`)

### 3. Local Encoder — 条件编码器
- **架构**: Transformer Encoder (非 causal)
- **规模**: 12 层, hidden_dim=1024, ffn_dim=4096, 16 heads
- **输入**: 音频特征 patch (`feat_dim=64`)
- **Python 类**: `VoxCPMLocEnc`

### 4. DiT + CFM — 扩散 Transformer + 条件流匹配
- **DiT 规模**: 12 层, hidden_dim=1024, ffn_dim=4096, 16 heads
- **Python 类**: `VoxCPMLocDiTV2` (本质上是 MiniCPMModel 作为 Transformer backbone)
- **CFM 配置**:
  - `solver`: euler
  - `t_scheduler`: log-norm
  - `sigma_min`: 1e-6
  - `inference_cfg_rate`: 2.0
- **Python 类**: `UnifiedCFM`

### 5. Audio VAE V2 — 非对称编解码器
- **输入采样率**: 16kHz (参考音频 / 训练音频)
- **输出采样率**: 48kHz (无需外部超分)
- **Encoder**: 128-dim 起始, strides `[2, 5, 8, 8]`, latent_dim=64
- **Decoder**: 2048-dim 起始, strides `[8, 6, 5, 2, 2, 2]`, 输出 1ch
- **特色**: 因果卷积、Snake 激活、采样率条件层 (`sr_bin_boundaries`)
- **Python 类**: `AudioVAEV2`

### 6. 其他层
- `fsq_layer`: Scalar Quantization (将 LM hidden 量化到离散 latent)
- `enc_to_lm_proj`, `lm_to_dit_proj`, `res_to_dit_proj`, `fusion_concat_proj`: 维度投影
- `stop_proj` + `stop_head`: 停止预测器（判断生成是否结束）

---

## 量化配置

```json
{
  "bits": 4,
  "group_size": 64,
  "targets": ["base_lm", "residual_lm"]
}
```

- 仅对 `base_lm` 和 `residual_lm` 做 4-bit 量化。
- `mlx-community/VoxCPM2-4bit` 已经将权重转换为 MLX 原生量化格式（safetensors 单文件，约 1.x GB）。
- 加载时会自动检测 `.scales` 后缀键并调用 `quantize(model:filter:)`。

---

## 文件结构

```
Sources/MLXAudioTTS/Models/VoxCPM2/
├── VoxCPM2Config.swift   # config.json 全量解析
├── VoxCPM2.swift         # 主模型骨架 + SpeechGenerationModel 协议
└── README.md             # 本文件
```

---

## 已注册到 TTS 工厂

在 `Sources/MLXAudioTTS/TTSModel.swift` 中增加了：

```swift
case "voxcpm2", "voxcpm":
    return try await VoxCPM2Model.fromPretrained(modelRepo, cache: cache)
```

以及 infer 逻辑：

```swift
if lower.contains("voxcpm2") || lower.contains("voxcpm") {
    return "voxcpm2"
}
```

---

## 实现 TODO

### 高优先级（阻塞基础推理）
1. **MiniCPM4 模型实现**
   - 目前骨架里用 `Qwen3Model` 占位。
   - 差异点：MiniCPM4 的 Attention 没有 `q_norm`/`k_norm`；支持 `scale_depth` 的残差缩放；LongRoPE 需要扩展因子的支持。
   - 建议：复制 `Qwen3.swift` 的 Attention/MLP/Block，移除 `qNorm`/`kNorm`，增加 `scaleDepth` 分支。

2. **VoxCPMLocEnc (Local Encoder)**
   - 12 层 Transformer encoder。
   - 输入维度 `feat_dim=64`。
   - 可复用 MiniCPM4 Block，但需关闭 causal mask，并加上输入投影。

3. **VoxCPM2DiT (Diffusion Transformer)**
   - 核心也是 MiniCPM4 Block（非 causal）。
   - 需要 timestep embedding (`SinusoidalPosEmb` + `TimestepEmbedding`)。
   - 需要 `in_proj` / `cond_proj` / `out_proj`。
   - 需要实现 `UnifiedCFM` 的 Euler solver 和 log-norm 噪声调度。

4. **Audio VAE V2**
   - 最复杂的子模块之一。
   - 包含 `CausalEncoder`、`CausalDecoder`、`SampleRateConditionLayer`。
   - 激活函数 `Snake1d`。
   - 权重键前缀 `audio_vae.encoder.*` / `audio_vae.decoder.*`。

5. **Forward / Generate 逻辑**
   - 参考 Python `VoxCPM2Model._generate`:
     - 文本 tokenize -> base LM -> FSQ -> residual LM -> DiT+CFM -> Audio VAE decode
     - 支持 voice design / voice cloning / ultimate cloning 三种模式
     - streaming 生成支持

### 中优先级
6. **权重键名映射 (sanitize)**
   - Python 权重键名与 Swift `ModuleInfo` 键名需要完全对齐。
   - 目前已知前缀：`base_lm.*`, `residual_lm.*`, `feat_encoder.*`, `feat_decoder.estimator.*`, `audio_vae.*`。

7. **4-bit 量化加载验证**
   - 确认 `mlx-community/VoxCPM2-4bit` 的 `model.safetensors` 中的量化键名格式。
   - 确保 `quantize(model:)` 仅作用于 `base_lm` 和 `residual_lm` 的 Linear 层。

---

## 参考资源

- **Python 官方仓库**: https://github.com/OpenBMB/VoxCPM
- **HuggingFace 权重**: https://huggingface.co/openbmb/VoxCPM2
- **MLX 4-bit 权重**: https://huggingface.co/mlx-community/VoxCPM2-4bit
- **vLLM-omni 示例**: https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/voxcpm2

---

## 编译状态

✅ `swift build --target MLXAudioTTS` 编译通过。
