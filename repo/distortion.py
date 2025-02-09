import pickle

from model.distortion import DistortionProfile
from repo.common import ROOT_DIR_PATH

_BASE_DIR_PATH = ROOT_DIR_PATH / "distortion"


def list_names() -> list[str]:
    _BASE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    lst = [(path, path.stat().st_mtime) for path in _BASE_DIR_PATH.iterdir()]
    lst.sort(key=lambda x: x[1], reverse=True)
    return [path.stem for path, _ in lst]


def put(profile: DistortionProfile) -> None:
    path = _BASE_DIR_PATH / f"{profile.name}.pickle"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(profile, f)


def get(name: str) -> DistortionProfile:
    path = _BASE_DIR_PATH / f"{name}.pickle"
    if path.exists():
        with path.open("rb") as f:
            profile = pickle.load(f)
        assert isinstance(profile, DistortionProfile)
        return profile
    else:
        raise FileNotFoundError(f"Profile '{name}' not found")


def exists(name: str) -> bool:
    return (_BASE_DIR_PATH / f"{name}.pickle").exists()
