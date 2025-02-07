import json
from datetime import datetime

from model.global_config import GlobalConfig
from repo.common import ROOT_DIR_PATH

_BASE_FILE_PATH = ROOT_DIR_PATH / "global_config.json"

_cache: GlobalConfig | None = None
_mtime: datetime | None = None


def get() -> GlobalConfig:
    global _cache, _mtime
    if _cache is None:
        _BASE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _BASE_FILE_PATH.exists():
            with open(_BASE_FILE_PATH, "r") as file:
                _cache = GlobalConfig.from_json(json.load(file))
            _mtime = datetime.fromtimestamp(_BASE_FILE_PATH.stat().st_mtime)
        else:
            _cache = GlobalConfig.create_default()
            _mtime = datetime.now()
    return _cache


def put(config: GlobalConfig) -> None:
    global _cache, _mtime
    _cache = config
    with open(_BASE_FILE_PATH, "w") as file:
        json.dump(_cache.to_json(), file)
    _mtime = datetime.fromtimestamp(_BASE_FILE_PATH.stat().st_mtime)


def get_mtime() -> datetime:
    if _mtime is None:
        get()
    assert _mtime is not None, _mtime
    return _mtime
