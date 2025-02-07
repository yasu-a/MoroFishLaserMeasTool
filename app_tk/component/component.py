from abc import ABC, abstractmethod

from app_tk.event import KeyEvent
from app_tk.rendering import RenderingContext, RenderingResult

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app_tk.app import Application
    from app_tk.scene import Scene


class Component(ABC):
    def __init__(self, app: "Application", scene: "Scene", name: str = None):
        self._app = app
        self._scene = scene
        self._name = name

    def get_name(self) -> str | None:
        return self._name

    @abstractmethod
    def render(self, ctx: RenderingContext) -> RenderingResult:
        raise NotImplementedError()

    def key_event(self, event: KeyEvent) -> bool:
        return False

    @abstractmethod
    def focus_count(self) -> int:
        raise NotImplementedError()
