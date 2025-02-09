import cv2

from model.image import Image
from repo.common import ROOT_DIR_PATH

_BASE_DIR_PATH = ROOT_DIR_PATH / "image"


def list_names() -> list[str]:
    lst = [(path, path.stat().st_mtime) for path in _BASE_DIR_PATH.iterdir()]
    lst.sort(key=lambda x: x[1], reverse=True)
    return [path.stem for path, _ in lst]


def put(image: Image) -> None:
    path = _BASE_DIR_PATH / f"{image.name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image.data)


def get(name: str) -> Image:
    path = _BASE_DIR_PATH / f"{name}.png"
    data = cv2.imread(str(path))
    return Image(name, data)


def exists(name: str) -> bool:
    return (_BASE_DIR_PATH / f"{name}.png").exists()
