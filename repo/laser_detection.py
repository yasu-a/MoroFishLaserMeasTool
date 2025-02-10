import json

from model.laser_detection import LaserDetectionProfile
from repo.common import ROOT_DIR_PATH

_BASE_DIR_PATH = ROOT_DIR_PATH / "laser_detection"


def list_names() -> list[str]:
    _BASE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    lst = [(path, path.stat().st_mtime) for path in _BASE_DIR_PATH.iterdir()]
    lst.sort(key=lambda x: x[1], reverse=True)
    return [path.stem for path, _ in lst]


def put(profile: LaserDetectionProfile) -> None:
    path = _BASE_DIR_PATH / f"{profile.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(profile.to_json(), f)


def get(name: str) -> LaserDetectionProfile:
    path = _BASE_DIR_PATH / f"{name}.json"
    if path.exists():
        with path.open("r") as f:
            profile = LaserDetectionProfile.from_json(json.load(f))
        assert isinstance(profile, LaserDetectionProfile)
        return profile
    else:
        raise FileNotFoundError(f"Laser detection profile '{name}' not found")


def exists(name: str) -> bool:
    return (_BASE_DIR_PATH / f"{name}.json").exists()
