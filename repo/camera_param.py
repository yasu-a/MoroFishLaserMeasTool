import json

from model.camera_param import CameraParamProfile
from repo.common import ROOT_DIR_PATH

_BASE_DIR_PATH = ROOT_DIR_PATH / "camera_param"


def list_names() -> list[str]:
    _BASE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    lst = [(path, path.stat().st_mtime) for path in _BASE_DIR_PATH.iterdir()]
    lst.sort(key=lambda x: x[1], reverse=True)
    return [path.stem for path, _ in lst]


def put(param: CameraParamProfile) -> None:
    path = _BASE_DIR_PATH / f"{param.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(param.to_json(), f)


def get(name: str) -> CameraParamProfile:
    path = _BASE_DIR_PATH / f"{name}.json"
    if path.exists():
        with path.open("r") as f:
            param = CameraParamProfile.from_json(json.load(f))
        assert isinstance(param, CameraParamProfile)
        return param
    else:
        raise FileNotFoundError(f"Camera param '{name}' not found")


def exists(name: str) -> bool:
    return (_BASE_DIR_PATH / f"{name}.json").exists()
