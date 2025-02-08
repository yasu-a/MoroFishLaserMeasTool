from abc import ABC, abstractmethod

from core.tk.event import KeyEvent
from core.tk.rendering import UIRenderingContext, RenderingResult, Canvas
from core.tk.scene import Scene


class Component(ABC):
    def __init__(self, scene: Scene, *, name: str = None):
        self.__scene = scene
        self.__name = name

    def get_scene(self) -> Scene:
        return self.__scene

    def get_name(self) -> str | None:
        return self.__name

    @abstractmethod
    def render(self, canvas: Canvas, ctx: UIRenderingContext) -> RenderingResult:
        raise NotImplementedError()

    def key_event(self, event: KeyEvent) -> bool:
        return False

    @abstractmethod
    def focus_count(self) -> int:
        raise NotImplementedError()
