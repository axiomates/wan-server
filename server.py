"""
Wan2.2-I2V-A14B Video Generation Server
类似 llama-server，启动时加载模型到显存，常驻等待请求。
支持 image-to-video（首帧）与 first-last-frame（首帧+尾帧+文字，FLF2V）、任务队列、取消、状态查询。

模型：Wan-AI/Wan2.2-I2V-A14B（diffusers，bf16 无量化）。MoE 双专家
（transformer 高噪声 + transformer_2 低噪声，boundary_ratio=0.9），
一次 from_pretrained 自动拉起。面向 DGX Spark（128GB 统一内存）全常驻，不做 CPU offload。
"""

import gc
import io
import time
import uuid
import base64
import asyncio
import logging
import threading
from enum import Enum
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wan-server")

# ---- 全局状态 ----
pipe = None
task_queue: "asyncio.Queue[Task]" = None
tasks: dict[str, "Task"] = {}          # task_id -> Task
tasks_lock = threading.RLock()
generation_lock = threading.Lock()      # 单个 pipeline 不并发执行，避免线程安全和显存问题
RESULTS_DIR = Path(__file__).parent / "outputs"

# Wan2.2 官方默认 negative prompt（中文），来自模型卡示例
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG 压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# Wan VAE：空间下采样 8×、时间下采样 4×；transformer patch 宽 2。
# 宽高须整除 mod_value = 8×2 = 16；帧数须满足 num_frames-1 能被 4 整除（4k+1）。
DIM_MULTIPLE = 16
FRAME_TEMPORAL = 4


class TaskStatus(str, Enum):
    QUEUED = "queued"
    WAITING = "waiting"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task:
    def __init__(self, task_id: str, params: dict):
        self.id = task_id
        self.params = params
        self.status = TaskStatus.QUEUED
        self.error: str | None = None
        self.result_path: Path | None = None
        self.frame_count: int = 0
        self.current_step: int = 0
        self.total_steps: int = 0
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        # 条件图（PIL.Image）。仅存内存、不进 params，避免轮询响应回传图片数据。
        # Wan I2V 首帧必填；尾帧可选（提供则做 FLF2V 首尾过渡）。
        self.cond_first = None
        self.cond_last = None


# ---- 请求/响应模型 ----

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="文本提示词，中英文均可")
    negative_prompt: str = Field(
        default=DEFAULT_NEGATIVE_PROMPT,
        max_length=2000,
        description="负面提示词"
    )
    width: int = Field(default=832, ge=64, le=1280, description="视频宽度，需被16整除")
    height: int = Field(default=480, ge=64, le=1280, description="视频高度，需被16整除")
    num_frames: int = Field(default=81, ge=5, le=241, description="帧数，需满足 4k+1，如 81, 121, 161")
    num_inference_steps: int = Field(default=40, ge=1, le=100, description="推理步数")
    guidance_scale: float = Field(default=3.5, ge=1.0, le=20.0, description="引导强度")
    seed: int = Field(default=-1, ge=-1, le=2**63 - 1, description="随机种子，-1 为随机")
    fps: int = Field(default=16, ge=1, le=60, description="输出帧率")
    image_first: str = Field(
        ...,
        description="首帧条件图，base64（可含 data URL 前缀）。Wan I2V 必填，视频从此帧开始演变。",
    )
    image_last: str | None = Field(
        default=None,
        description="尾帧条件图，base64（可含 data URL 前缀）。提供则做首尾过渡（FLF2V）。",
    )


class GenerateResponse(BaseModel):
    task_id: str
    status: str
    queue_position: int | None = None


class TaskStatusResponse(BaseModel):
    id: str
    status: str
    queue_position: int | None = None
    current_step: int = 0
    total_steps: int = 0
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    params: dict | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    queue_size: int
    queue_max: int
    waiting: int
    processing: int
    uptime_seconds: float


# ---- 模型加载 ----

def load_pipeline(model_id: str, vae_tiling: bool = False, compile: bool = False):
    """
    加载 Wan2.2-I2V-A14B 到显存，常驻不释放。

    一次 from_pretrained 依据 model_index.json 自动实例化双专家
    （transformer 高噪声 + transformer_2 低噪声）与 boundary_ratio=0.9，
    无需手工构造。VAE 用 fp32 提升解码质量，其余组件 bf16。

    面向 DGX Spark（128GB 统一内存）：全部 .to("cuda") 常驻，不做 CPU offload
    （CPU/GPU 共享同一物理内存，offload 无收益还引入拷贝开销）。

    - vae_tiling: 高分辨率/长视频时开启 VAE 分块解码，降低解码激活峰值。
    - compile: 对两个 transformer 做 torch.compile（实验性）。
    """
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Wan2.2-I2V-A14B (bf16, dual-expert MoE): %s", model_id)
    t0 = time.time()

    try:
        # VAE 单独用 fp32（官方建议，解码质量更好）
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=torch.bfloat16
        )
    except Exception as e:
        logger.error("Failed to load model from '%s': %s", model_id, e)
        raise RuntimeError(f"Model load failed: {e}") from e

    _move_to_cuda(pipe, device)
    _optimize_pipeline(pipe, vae_tiling=vae_tiling, skip_compile=not compile)

    elapsed = time.time() - t0
    logger.info("Model loaded in %.1fs on %s (all experts resident, no offload)", elapsed, device)
    return pipe


def _move_to_cuda(pipe, device):
    """把整个 pipeline（双专家 + text_encoder + vae）常驻 GPU，处理 OOM"""
    try:
        pipe.to(device)
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM while loading — Wan2.2-I2V-A14B bf16 需约 80GB 权重内存")
        raise
    except Exception as e:
        logger.error("Failed to move pipeline to %s: %s", device, e)
        raise


def _optimize_pipeline(pipe, vae_tiling=False, skip_compile=True):
    """可选 VAE tiling 省解码显存、可选 torch.compile 加速"""
    if vae_tiling and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
        logger.info("VAE tiling enabled")

    if skip_compile:
        return

    try:
        # 双专家都编译
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
        if hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
            pipe.transformer_2 = torch.compile(pipe.transformer_2, mode="reduce-overhead")
        logger.info("Transformer(s) compiled with torch.compile")
    except Exception as e:
        logger.warning("torch.compile failed (will run without it): %s", e)


# ---- 队列工作线程 ----

def run_generation(task: Task):
    """在独立线程中执行生成（同步 GPU 操作）"""
    logger.info("[%s] Started: '%s...'", task.id, task.params["prompt"][:60])
    output_path = RESULTS_DIR / f"{task.id}.mp4"
    tmp_path = RESULTS_DIR / f"{task.id}.tmp.mp4"
    generated_frame_count = 0

    try:
        p = task.params
        generator = None
        if p["seed"] >= 0:
            generator = torch.Generator(device="cpu").manual_seed(p["seed"])

        def on_step_end(pipeline, step, timestep, callback_kwargs):
            # 扩散每步结束回调；step 从 0 开始，故 +1 表示已完成步数
            with tasks_lock:
                task.current_step = step + 1
            return callback_kwargs

        # 首帧/尾帧 resize 到目标尺寸（宽高已由前端/后端对齐到 16 的倍数）
        first = task.cond_first.resize((p["width"], p["height"]))
        last = task.cond_last.resize((p["width"], p["height"])) if task.cond_last is not None else None

        call_kwargs = dict(
            image=first,
            prompt=p["prompt"],
            negative_prompt=p["negative_prompt"],
            width=p["width"],
            height=p["height"],
            num_frames=p["num_frames"],
            num_inference_steps=p["num_inference_steps"],
            guidance_scale=p["guidance_scale"],
            generator=generator,
            callback_on_step_end=on_step_end,
            output_type="np",
            return_dict=True,
        )
        # 尾帧：经 VAE 编码放入 latent 序列末尾，中间帧由模型生成（FLF2V）。
        # guidance_scale_2 不传，boundary_ratio 非 None 时自动回退 = guidance_scale。
        if last is not None:
            call_kwargs["last_image"] = last

        tmp_path.unlink(missing_ok=True)

        with generation_lock:
            with tasks_lock:
                if task.status == TaskStatus.CANCELLED:
                    return
                task.status = TaskStatus.PROCESSING
                task.started_at = time.time()
                task.total_steps = p["num_inference_steps"]

            out = pipe(**call_kwargs)
            video = out.frames[0]  # (F, H, W, C)，值域 0-1，无音频
            generated_frame_count = frame_count(video)
            write_video_file(tmp_path, video, p["fps"])
            del out, video

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tmp_path.replace(output_path)
        with tasks_lock:
            task.frame_count = generated_frame_count
            task.result_path = output_path
            task.status = TaskStatus.DONE

    except torch.cuda.OutOfMemoryError:
        logger.error("[%s] CUDA OOM", task.id)
        tmp_path.unlink(missing_ok=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        with tasks_lock:
            task.error = "GPU out of memory. Try reducing width/height/num_frames, or start with --vae-tiling."
            task.status = TaskStatus.FAILED

    except Exception as e:
        logger.error("[%s] Generation failed: %s", task.id, e)
        tmp_path.unlink(missing_ok=True)
        with tasks_lock:
            task.error = str(e)
            task.status = TaskStatus.FAILED

    finally:
        # 释放条件图，避免完成/失败的任务在内存里长期持有 PIL 图像
        task.cond_first = None
        task.cond_last = None
        with tasks_lock:
            task.finished_at = time.time()
            elapsed = task.finished_at - task.started_at if task.started_at else 0
            status = task.status.value.upper()
            frames = task.frame_count
        logger.info(
            "[%s] %s in %.1fs (%d frames)",
            task.id, status, elapsed, frames
        )


async def queue_worker(worker_id: int):
    """后台循环：从队列取任务，在线程池执行"""
    loop = asyncio.get_running_loop()
    while True:
        task = await task_queue.get()
        with tasks_lock:
            if task.status == TaskStatus.CANCELLED:
                task_queue.task_done()
                continue
            task.status = TaskStatus.WAITING

        try:
            await loop.run_in_executor(None, run_generation, task)
        finally:
            task_queue.task_done()


async def cleanup_expired_tasks():
    """定期清理已完成/失败/取消的过期任务；输出文件保留在 outputs/"""
    TTL = 24 * 60 * 60  # 24 小时后从内存中清除
    while True:
        await asyncio.sleep(60)
        now = time.time()
        removed = 0
        with tasks_lock:
            expired = [
                tid for tid, t in tasks.items()
                if t.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)
                and t.finished_at
                and (now - t.finished_at > TTL)
            ]
            for tid in expired:
                del tasks[tid]
                removed += 1
        if removed:
            logger.info("Cleaned up %d expired task(s)", removed)


# ---- FastAPI 生命周期 ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe, task_queue
    model_id = app.state.model_id
    RESULTS_DIR.mkdir(exist_ok=True)

    try:
        pipe = load_pipeline(
            model_id,
            vae_tiling=app.state.vae_tiling,
            compile=app.state.compile,
        )
    except Exception:
        logger.critical("Failed to load model — server cannot start")
        raise

    task_queue = asyncio.Queue(maxsize=app.state.queue_max_size)
    worker = asyncio.create_task(queue_worker(0))
    cleanup_task = asyncio.create_task(cleanup_expired_tasks())

    app.state.start_time = time.time()
    logger.info("Server ready — queue max=%d", app.state.queue_max_size)
    yield

    # shutdown
    logger.info("Shutting down...")
    cleanup_task.cancel()
    worker.cancel()
    # 等正在执行的生成结束（run_generation 在 executor 线程里，cancel 停不了它）
    # 再删 pipe，避免生成中途模型被销毁。
    with generation_lock:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("Shutdown complete")


app = FastAPI(title="Wan2.2-I2V Video Server", version="1.0", lifespan=lifespan)

# 允许任意来源跨域调用 API（本服务无凭证鉴权，故 allow_credentials=False）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 静态文件 / UI ----

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ---- 端点 ----

def decode_base64_image(data: str):
    """把 base64（可含 data URL 前缀）解码为 RGB PIL.Image；失败抛 ValueError。"""
    from PIL import Image, UnidentifiedImageError

    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, base64.binascii.Error) as e:
        raise ValueError(f"无法解码 base64 图片: {e}") from e
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"无法识别图片内容: {e}") from e
    return img.convert("RGB")


@app.get("/health", response_model=HealthResponse)
async def health():
    with tasks_lock:
        waiting = sum(1 for t in tasks.values() if t.status == TaskStatus.WAITING)
        processing = sum(1 for t in tasks.values() if t.status == TaskStatus.PROCESSING)
    return HealthResponse(
        status="ok",
        model_loaded=pipe is not None,
        device=str(pipe.device) if pipe else "N/A",
        queue_size=task_queue.qsize() if task_queue else 0,
        queue_max=app.state.queue_max_size,
        waiting=waiting,
        processing=processing,
        uptime_seconds=round(time.time() - app.state.start_time, 1),
    )


@app.post("/v1/video/generate", response_model=GenerateResponse)
async def submit_generation(req: GenerateRequest):
    """提交生成任务，立即返回 task_id"""
    if pipe is None:
        raise HTTPException(503, "Model not loaded yet")

    # 参数校验
    errors = []
    if req.width % DIM_MULTIPLE != 0 or req.height % DIM_MULTIPLE != 0:
        errors.append(f"width 和 height 必须能被 {DIM_MULTIPLE} 整除")
    if (req.num_frames - 1) % FRAME_TEMPORAL != 0:
        errors.append(f"num_frames 必须满足 {FRAME_TEMPORAL}k+1，如 81, 121, 161")

    # 条件图解码（仅在内存中，不写入 params）。Wan I2V 首帧必填。
    cond_first = cond_last = None
    try:
        cond_first = decode_base64_image(req.image_first)
    except ValueError as e:
        errors.append(f"首帧图无效: {e}")
    if req.image_last:
        try:
            cond_last = decode_base64_image(req.image_last)
        except ValueError as e:
            errors.append(f"尾帧图无效: {e}")

    if errors:
        raise HTTPException(400, "; ".join(errors))

    task_id = uuid.uuid4().hex[:12]
    params = req.model_dump()
    # base64 图片数据体积大，不能留在 params（会被 /tasks、/status 每次轮询回传）。
    # 仅保留轻量标记，真正的图存在 Task 属性上。
    params.pop("image_first", None)
    params.pop("image_last", None)
    params["has_first_frame"] = cond_first is not None
    params["has_last_frame"] = cond_last is not None

    task = Task(task_id, params)
    task.cond_first = cond_first
    task.cond_last = cond_last

    with tasks_lock:
        tasks[task_id] = task

    try:
        task_queue.put_nowait(task)
    except asyncio.QueueFull:
        with tasks_lock:
            tasks.pop(task_id, None)
        raise HTTPException(
            503,
            f"Queue full ({app.state.queue_max_size} max). Try again later or increase --queue-size."
        )

    return GenerateResponse(
        task_id=task_id,
        status="queued",
        queue_position=_queue_position(task_id),
    )


@app.get("/v1/video/tasks", response_model=list[TaskStatusResponse])
async def list_tasks():
    """列出当前仍保留在内存中的任务"""
    with tasks_lock:
        snapshot = list(tasks.values())
    snapshot.sort(key=lambda t: t.created_at, reverse=True)
    return [task_response(t) for t in snapshot]


@app.get("/v1/video/status/{task_id}", response_model=TaskStatusResponse)
async def task_status(task_id: str):
    """查询任务状态"""
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
    return task_response(task)


@app.post("/v1/video/cancel/{task_id}")
async def cancel_task(task_id: str):
    """取消排队中的任务（已开始的无法取消）"""
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
        if task.status == TaskStatus.CANCELLED:
            return {"task_id": task_id, "status": "cancelled", "message": "Already cancelled"}
        if task.status not in (TaskStatus.QUEUED, TaskStatus.WAITING):
            raise HTTPException(409, f"Cannot cancel task in '{task.status.value}' status")

        task.status = TaskStatus.CANCELLED
        task.finished_at = time.time()
    logger.info("[%s] Cancelled while queued", task_id)
    return {"task_id": task_id, "status": "cancelled"}


@app.get("/v1/video/result/{task_id}")
async def get_result(task_id: str):
    """下载生成的视频"""
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
        status = task.status
        error = task.error
        result_path = task.result_path

    if status in (TaskStatus.QUEUED, TaskStatus.WAITING, TaskStatus.PROCESSING):
        raise HTTPException(425, f"Task not ready yet (status: {status.value})")
    if status == TaskStatus.FAILED:
        raise HTTPException(500, error or "Generation failed")
    if status == TaskStatus.CANCELLED:
        raise HTTPException(410, "Task was cancelled")

    if result_path is None or not result_path.exists():
        raise HTTPException(404, "Result file not found")

    return FileResponse(
        result_path,
        media_type="video/mp4",
        filename=f"{task_id}.mp4",
    )


# ---- 辅助函数 ----

def task_response(task: Task) -> TaskStatusResponse:
    with tasks_lock:
        return TaskStatusResponse(
            id=task.id,
            status=task.status.value,
            queue_position=_queue_position(task.id),
            current_step=task.current_step,
            total_steps=task.total_steps,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
            params=task.params,
            error=task.error,
        )


def _queue_position(task_id: str) -> int | None:
    """计算排队位置（粗糙：按创建时间排序 queued 任务）"""
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None or task.status != TaskStatus.QUEUED:
            return None
        queued = [t for t in tasks.values() if t.status == TaskStatus.QUEUED]
        queued.sort(key=lambda t: t.created_at)
        for i, t in enumerate(queued):
            if t.id == task_id:
                return i + 1
    return None


def frame_count(frames) -> int:
    if isinstance(frames, np.ndarray):
        return int(frames.shape[1] if frames.ndim == 5 else frames.shape[0])
    return len(frames or [])


def write_video_file(output_path: Path, frames, fps: int):
    """把生成结果写入 outputs/*.mp4。Wan 返回 numpy [F,H,W,C]（或 [B,F,H,W,C]），值域 0-1，无音频。"""
    import imageio

    if isinstance(frames, np.ndarray):
        video = (frames * 255).round().clip(0, 255).astype("uint8")
        video = video[0] if video.ndim == 5 else video
        writer = imageio.get_writer(str(output_path), fps=fps)
        try:
            for frame in video:
                writer.append_data(frame)
        finally:
            writer.close()
        return

    # 兜底：PIL frames 序列
    writer = imageio.get_writer(str(output_path), fps=fps)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


# ---- 入口 ----

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Wan2.2-I2V-A14B Video Generation Server")
    parser.add_argument("--model", default="./Wan2.2-I2V-A14B-Diffusers",
                        help="本地 Diffusers 模型目录（由 prepare_base.py 预下载），默认 ./Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--queue-size", type=int, default=8,
                        help="最大排队数，超出拒绝新请求（默认 8）")
    parser.add_argument("--vae-tiling", action="store_true",
                        help="开启 VAE 分块解码，降低高分辨率/长视频的解码显存峰值（略慢）")
    parser.add_argument("--compile", action="store_true",
                        help="对两个 transformer 做 torch.compile（实验性，可能失败并自动回退）")
    args = parser.parse_args()

    if args.queue_size < 1:
        parser.error("--queue-size must be >= 1")

    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"Model directory not found: {args.model}. Run prepare_base.py first.")
    if not (model_path / "model_index.json").exists():
        parser.error(f"model_index.json not found under: {args.model}")

    app.state.model_id = args.model
    app.state.queue_max_size = args.queue_size
    app.state.vae_tiling = args.vae_tiling
    app.state.compile = args.compile

    logger.info("Starting server on %s:%d | model=%s | queue=%d | vae_tiling=%s | compile=%s",
                args.host, args.port, args.model, args.queue_size,
                args.vae_tiling, args.compile)
    uvicorn.run(app, host=args.host, port=args.port)
