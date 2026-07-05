# Wan2.2-I2V-A14B Video Generation Server

类似 llama-server，启动时把 Wan2.2-I2V-A14B（bf16 无量化）加载到显存，常驻等待请求生成视频。提供网页 UI，支持**图生视频（首帧）**与**首尾帧过渡（首帧 + 尾帧 + 文字，FLF2V）**、任务队列、取消、状态查询。

面向 **DGX Spark（GB10，128GB 统一内存）** 设计：双专家 MoE 全量常驻，追求最高画质。

## 模型说明

- 模型：[`Wan-AI/Wan2.2-I2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
- 结构：MoE 双专家（`transformer` 高噪声 + `transformer_2` 低噪声，`boundary_ratio=0.9`），`text_encoder` 为 UMT5-XXL，`vae` 为 AutoencoderKLWan，无需 CLIP image_encoder。
- 精度：bf16 无量化（VAE 用 fp32 提升解码质量）。
- 静态权重约 **80 GB**（双专家各 ~28GB + UMT5-XXL ~22GB）。一次 `from_pretrained` 自动拉起全部组件。

> **为什么首帧必填**：Wan2.2-I2V-A14B 是图生视频模型，`image`（首帧）是必填输入，没有纯文生视频路径。核心用法是「首帧 + prompt」或「首帧 + 尾帧 + prompt」。

## 从头开始

**前提**：NVIDIA 显卡 + 足够内存装下约 80GB 权重（DGX Spark 128GB 统一内存最合适），Python 3.10+。

### 1. 装 PyTorch

不需要全局 CUDA toolkit，只需 NVIDIA 驱动 + 对应 CUDA 版的 torch：

```bash
# DGX Spark（GB10，Blackwell sm_121）必须用 CUDA 13 版
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

> **选对 CUDA 版本很关键**：`cu130` 适配较新 GPU（含 Blackwell / DGX Spark GB10，架构 sm_121）。老卡（Ampere/Ada）用 `cu126`。装错会在加载时报 `CUDA error: no kernel image is available for execution on the device`。

验证 GPU 可用：

```bash
python -c "import torch; print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0)); print('cc', torch.cuda.get_device_capability(0)); print('arch_list', torch.cuda.get_arch_list())"
# cuda 必须是 True；GB10 的 cc 是 (12, 1)=sm_121，arch_list 里必须有 sm_120/sm_121
```

### 2. 进项目装依赖

```bash
cd wan-server
pip install -r requirements.txt
```

### 3. 预下载模型（只需一次）

```bash
python prepare_base.py
```

下载到 `./Wan2.2-I2V-A14B-Diffusers/`，约 80 GB+。下载**完整仓库**（双专家 + text_encoder + vae + scheduler + tokenizer），启动时零下载、纯离线加载。

最终目录结构类似：

```text
wan-server/
├─ server.py
├─ prepare_base.py
├─ requirements.txt
├─ static/index.html
└─ Wan2.2-I2V-A14B-Diffusers/
   ├─ model_index.json
   ├─ transformer/          # 高噪声专家
   ├─ transformer_2/        # 低噪声专家
   ├─ text_encoder/         # UMT5-XXL
   ├─ vae/
   ├─ scheduler/
   └─ tokenizer/
```

### 4. 启动

```bash
python server.py --model ./Wan2.2-I2V-A14B-Diffusers --port 4323
```

看到 `Server ready` 就成功了。服务只使用本地目录，目录不存在会直接报错，不会在启动时下载大文件。

### 5. 打开 Web UI

浏览器访问 `http://localhost:8080/`，上传首帧（可选尾帧）+ 填 prompt，即可发请求、看队列、取消任务、下载视频。

> 不要直接双击 `static/index.html`，那样走 `file://` 协议发不了请求。

## Web UI 参数（档位）

- **条件图**：首帧必填（图生视频起点）；尾帧可选（提供则从首帧过渡到尾帧，FLF2V）。
- **Resolution**：Wan2.2 官方训练分辨率 **480P / 720P**，提供 16:9 / 9:16 / 1:1，全部对齐到 16 像素。720P 明显更慢。不提供 1080p 以上（超出训练分辨率会画质崩坏且极慢）。
- **Duration + FPS**：选时长和帧率，前端自动换算成合法的 `4k+1` 帧数并夹到 5~241，实时显示实际帧数与时长。fps 默认 16（官方默认）。
- **Quality（质量档）**：20/30/40/50/60 推理步，默认 40（官方默认）。
- **Guidance Scale**：默认 3.5（官方默认）。

## 为什么不做 CPU offload

DGX Spark 是 CPU/GPU 共享的统一内存（128GB LPDDR5x），"offload 到 CPU" 没有跨设备搬运意义，反而引入无谓拷贝。因此服务把双专家、text_encoder、VAE 全部 `.to("cuda")` 常驻，零搬运最快。

高分辨率/长视频若解码时显存吃紧，可加 `--vae-tiling` 开启 VAE 分块解码，降低解码激活峰值（略慢）。

## 速度预期

DGX Spark 的统一内存带宽（约 273 GB/s）远低于 H100 级 HBM，且双专家逐 timestep 参与，**720P 出片按分钟计**。UI 已是异步队列 + 进度轮询，提交后可关页面、之后回来下载。

## 输出文件

生成完成后，mp4 写入项目下的 `outputs/` 目录：

```text
outputs/<task_id>.mp4
```

任务状态记录在内存里保留约 24 小时后自动清理，但 `outputs/` 里的 mp4 会保留。

## 离线部署

联网机器上做完步骤 3，把整个项目目录（含 `Wan2.2-I2V-A14B-Diffusers/`）拷到离线机器，然后步骤 2 + 4。

## API 速查

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/video/generate` | 提交任务，返回 task_id（首帧 `image_first` 必填） |
| GET | `/v1/video/tasks` | 列出当前仍在内存中的任务 |
| GET | `/v1/video/status/{id}` | 查状态 |
| POST | `/v1/video/cancel/{id}` | 取消排队 |
| GET | `/v1/video/result/{id}` | 下载 mp4 |
| GET | `/health` | 服务状态 |

状态：`queued` → `waiting` → `processing` → `done` / `failed` / `cancelled`

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `./Wan2.2-I2V-A14B-Diffusers` | 本地 Diffusers 模型目录；必须先用 `prepare_base.py` 预下载好 |
| `--host` | `0.0.0.0` | |
| `--port` | `8080` | |
| `--queue-size` | `8` | 排满即拒 |
| `--vae-tiling` | 关 | 开启 VAE 分块解码，降低高分辨率/长视频解码显存峰值（略慢） |
| `--compile` | 关 | 对两个 transformer 做 `torch.compile`（实验性，可能失败并自动回退） |

## 请求参数约束

| 参数 | 约束 |
|------|------|
| `image_first` | 必填，base64（可含 data URL 前缀） |
| `image_last` | 可选，提供则做 FLF2V |
| `width` / `height` | 必须能被 16 整除，≤ 1280 |
| `num_frames` | 必须满足 `4k+1`（如 81, 121, 161），范围 5~241 |
| `prompt` | 1~2000 字符 |

## 常见错误码

| HTTP | 含义 |
|------|------|
| 400 | 参数不对（含首帧缺失、宽高非 16 倍数、帧数非 4k+1） |
| 404 | task_id 不存在 |
| 409 | 已开始，不能取消 |
| 410 | 已取消 |
| 425 | 还没生成完 |
| 500 | 生成出错 |
| 503 | 队列满 / 模型没加载完 |
