"""
预下载 Wan2.2-I2V-A14B 完整 Diffusers 模型到本地目录。

与 LTX/GGUF 方案不同：Wan2.2-I2V-A14B 是 MoE 双专家模型，权重就是官方
diffusers 分片（safetensors，bf16 无量化），没有单文件 GGUF 替代一说，
因此这里下载**完整仓库**的全部组件：
    transformer/        高噪声专家
    transformer_2/      低噪声专家
    text_encoder/       UMT5-XXL
    vae/                AutoencoderKLWan
    scheduler/ tokenizer/ + model_index.json

下载后服务启动零下载、纯离线加载，目录可整体拷到离线机器。

用法:
    # 联网下载到默认目录 ./Wan2.2-I2V-A14B-Diffusers
    python prepare_base.py

    # 指定输出目录
    python prepare_base.py --output ./Wan2.2-I2V-A14B-Diffusers

下载量约 80 GB+（双专家 bf16 各 ~28GB + UMT5-XXL ~22GB + VAE），只需执行一次。
"""

import argparse
import sys
from pathlib import Path

REPO_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


def download_base(output_dir: str):
    from huggingface_hub import snapshot_download

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading full model from {REPO_ID}")
    print(f"Output directory: {out.resolve()}")
    print("Components: transformer/ (high-noise) + transformer_2/ (low-noise) + "
          "text_encoder/ + vae/ + scheduler/ + tokenizer/")
    print()
    print("Download size: ~80 GB+ (dual 14B experts bf16 + UMT5-XXL text encoder)")
    print("This may take a while depending on your connection.")
    print()

    snapshot_download(
        REPO_ID,
        local_dir=str(out),
    )

    print()
    print("Done! Start the server pointing at this directory:")
    print(f"  python server.py --model {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download full Wan2.2-I2V-A14B Diffusers model for offline serving"
    )
    parser.add_argument("--output", default="./Wan2.2-I2V-A14B-Diffusers",
                        help="输出目录，默认 ./Wan2.2-I2V-A14B-Diffusers")
    args = parser.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("请先安装 huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    download_base(args.output)
