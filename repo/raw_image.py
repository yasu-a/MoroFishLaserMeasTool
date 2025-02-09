import cv2

from model.raw_image import RawImage
from repo.common import ROOT_DIR_PATH

_BASE_DIR_PATH = ROOT_DIR_PATH / "raw_image"


def list_names() -> list[str]:
    _BASE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    lst = [(path, path.stat().st_mtime) for path in _BASE_DIR_PATH.iterdir()]
    lst.sort(key=lambda x: x[1], reverse=True)
    return [path.stem for path, _ in lst]


def put(image: RawImage) -> None:
    path = _BASE_DIR_PATH / f"{image.name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image.data)


def get(name: str) -> RawImage:
    path = _BASE_DIR_PATH / f"{name}.png"
    data = cv2.imread(str(path))
    return RawImage(name, data)


def exists(name: str) -> bool:
    return (_BASE_DIR_PATH / f"{name}.png").exists()
