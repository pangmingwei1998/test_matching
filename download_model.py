#!/usr/bin/env python3
"""
手动下载 BGE-M3 模型到本地
用于解决 Hugging Face 镜像站限流问题
"""

import os
import sys
from pathlib import Path

# 模型保存路径
MODEL_DIR = Path("/home/pmw/h20/Text_matching/models/bge-m3")
MODEL_NAME = "BAAI/bge-m3"


def download_with_git():
    """使用 git-lfs 下载模型"""
    print("=" * 60)
    print("方法 1: 使用 Git LFS 下载模型")
    print("=" * 60)

    # 检查 git-lfs
    import subprocess
    try:
        subprocess.run(["git", "lfs", "version"], capture_output=True, check=True)
        print("✓ git-lfs 已安装")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ git-lfs 未安装")
        print("\n请先安装 git-lfs:")
        print("  Ubuntu/Debian: sudo apt-get install git-lfs")
        print("  CentOS/RHEL:   sudo yum install git-lfs")
        print("  macOS:         brew install git-lfs")
        return False

    # 创建目录
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    # 克隆模型
    if MODEL_DIR.exists():
        print(f"\n目录已存在: {MODEL_DIR}")
        choice = input("是否删除并重新下载? (y/N): ")
        if choice.lower() == 'y':
            import shutil
            shutil.rmtree(MODEL_DIR)
        else:
            print("取消下载")
            return False

    print(f"\n正在下载模型到: {MODEL_DIR}")
    print("模型大小约 2.3 GB，请耐心等待...")

    try:
        # 尝试从镜像站下载
        mirror_url = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        clone_url = f"{mirror_url.replace('https://', 'https://')}/{MODEL_NAME}"

        subprocess.run([
            "git", "clone",
            f"https://huggingface.co/{MODEL_NAME}",
            str(MODEL_DIR)
        ], check=True)

        print(f"\n✓ 模型下载完成: {MODEL_DIR}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ 下载失败: {e}")
        return False


def download_with_huggingface():
    """使用 huggingface_hub 下载"""
    print("\n" + "=" * 60)
    print("方法 2: 使用 huggingface_hub 下载")
    print("=" * 60)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装 huggingface_hub:")
        print("  pip install huggingface_hub")
        return False

    # 创建目录
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n正在下载模型到: {MODEL_DIR}")
    print("模型大小约 2.3 GB，请耐心等待...")

    try:
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"\n✓ 模型下载完成: {MODEL_DIR}")
        return True
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False


def verify_model():
    """验证模型文件"""
    print("\n" + "=" * 60)
    print("验证模型文件")
    print("=" * 60)

    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    ]

    missing = []
    for file in required_files:
        path = MODEL_DIR / file
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ {file} ({size:.1f} MB)")
        else:
            print(f"✗ {file} 缺失")
            missing.append(file)

    if not missing:
        print("\n✓ 模型文件完整")
        return True
    else:
        print(f"\n✗ 模型文件不完整，缺失: {missing}")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║           BGE-M3 模型下载工具                            ║
╚══════════════════════════════════════════════════════════╝

如果遇到 Hugging Face 镜像站限流 (429 错误)，
可以使用本工具将模型下载到本地。

目标目录: %s
    """ % MODEL_DIR)

    if MODEL_DIR.exists() and verify_model():
        print("\n模型已存在且完整，无需重新下载")
        return

    print("\n请选择下载方式:")
    print("  1. 使用 Git LFS (推荐)")
    print("  2. 使用 huggingface_hub")
    print("  0. 退出")

    choice = input("\n请输入选项 (0-2): ").strip()

    if choice == "1":
        if download_with_git():
            verify_model()
    elif choice == "2":
        if download_with_huggingface():
            verify_model()
    else:
        print("退出")


if __name__ == "__main__":
    main()
