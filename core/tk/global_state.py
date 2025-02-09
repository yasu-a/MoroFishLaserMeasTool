from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.tk.app import Application
    from core.tk.scene import Scene

_app: "Application"


def register_app(app: "Application") -> None:
    if "_app" in globals():
        raise ValueError("Only one application can be created")

    global _app
    _app = app


def get_app() -> "Application":
    return _app


def get_scene() -> "Scene":
    return _app.get_active_scene()
