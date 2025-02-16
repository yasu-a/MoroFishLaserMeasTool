import shutil
from pathlib import Path

from model.video import Video
from repo import ROOT_DIR_PATH

_BASE_DIR_PATH = ROOT_DIR_PATH / "video"


def get_temp_video_path() -> Path:
    _BASE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    return _BASE_DIR_PATH / "__out__.mp4"


def list_names() -> list[str]:
    _BASE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    lst = [
        (path, path.stat().st_mtime)
        for path in _BASE_DIR_PATH.iterdir()
        if path != get_temp_video_path()
    ]
    lst.sort(key=lambda x: x[1], reverse=True)
    return [path.stem for path, _ in lst]


def put(video: Video) -> None:
    path = _BASE_DIR_PATH / video.name
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def get(name: str) -> Video:
    path = _BASE_DIR_PATH / name
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {name}")
    return Video(name=name)


def get_item_path(name: str, item_name: str) -> Path:
    return _BASE_DIR_PATH / name / item_name


def exists(name: str) -> bool:
    if not name:
        return False
    return (_BASE_DIR_PATH / name).exists()
