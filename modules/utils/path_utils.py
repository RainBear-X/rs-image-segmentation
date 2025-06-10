from pathlib import Path


def get_project_root() -> Path:
    """返回项目根目录的 Path 对象"""
    return Path(__file__).resolve().parents[2]
