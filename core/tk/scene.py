import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Iterable
from typing import TypeVar

import cv2
import numpy as np

from core.tk.app import ApplicationWindowSize
from core.tk.event import KeyEvent, MouseEvent
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext

if TYPE_CHECKING:
    from core.tk.component.component import Component


class SceneEventHandlers(ABC):
    def _on_button_triggered(self, sender: "Component") -> None:
        pass

    def _on_value_changed(self, sender: "Component") -> None:
        pass


class PictureInPicturePlaceholder:
    def __init__(self):
        self._im: np.ndarray | None = None
        self._x_ratio: float | None = None
        self._y_ratio: float | None = None

        self._screen_size: tuple[int, int] | None = None

        self._margin = 20

    def set_image(self, im: np.ndarray | None, *, width: int = None, height: int = None):
        if im is None:
            assert width is None and height is None
            self._im = None
        else:
            im_h, im_w = im.shape[:2]
            if width is None and height is None:
                raise ValueError("Either width or height must be specified")
            elif width is None:
                width = int(im_w / im_h * height)
            elif height is None:
                height = int(im_h / im_w * width)
            self._x_ratio = width / im_w
            self._y_ratio = height / im_h
            # noinspection PyTypeChecker
            self._im = cv2.resize(im, (width, height), cv2.INTER_AREA)

    def set_screen_size(self, width, height):
        self._screen_size = width, height

    def is_ready(self) -> bool:
        return self._screen_size is not None and self._im is not None

    def get_offset(self) -> tuple[int, int] | None:
        if not self.is_ready():
            return None
        x_size, y_size = self._screen_size
        return x_size - self._margin - self._im.shape[1], y_size - self._margin - self._im.shape[0]

    def get_end(self) -> tuple[int, int] | None:
        if not self.is_ready():
            return None
        x_size, y_size = self._screen_size
        return x_size - self._margin, y_size - self._margin

    def translate_screen_to_local(self, x: int, y: int) -> tuple[int, int] | None:
        if not self.is_ready():
            return None
        x_ofs, y_ofs = self.get_offset()
        x_local = (x - x_ofs) / self._x_ratio
        y_local = (y - y_ofs) / self._y_ratio
        x_end, y_end = self.get_end()
        if x_local < 0 or x_local >= x_end or y_local < 0 or y_local >= y_end:
            return None  # out of screen range
        return x_local, y_local

    def get_size(self) -> tuple[int, int] | None:
        if not self.is_ready():
            return None
        return self._im.shape[1], self._im.shape[0]

    def get_image(self) -> np.ndarray | None:
        return self._im


C = TypeVar("C", bound="Component")


class ComponentCollection:
    def __init__(self):
        self._components: "list[Component]" = []
        self._name_to_index: dict[str, int] = {}

    def add(self, c: "Component") -> None:
        index = len(self._components)
        if c.get_name() is not None:
            if c.get_name() in self._name_to_index:
                raise ValueError(f"Component with name '{c.get_name()}' already exists")
            self._name_to_index[c.get_name()] = index
        self._components.append(c)

    def find_by_name(self, name: str) -> "Component":
        if name not in self._name_to_index:
            raise ValueError(f"Component with name '{name}' not found")
        return self._components[self._name_to_index[name]]

    def __iter__(self) -> Iterable["Component"]:
        yield from self._components


class Scene(SceneEventHandlers, ABC):
    def __init__(self, *, is_stationed=False):
        self._components = ComponentCollection()
        self._is_stationed = is_stationed  # disallow go-back
        self._global_focus_index = 0

        self._pip = PictureInPicturePlaceholder()  # picture in picture

        self._listener_enabled = True

    def is_stationed(self) -> bool:
        return self._is_stationed

    @contextlib.contextmanager
    def disable_listener_context(self):
        prev_state = self._listener_enabled
        self._listener_enabled = False
        yield
        self._listener_enabled = prev_state

    def notify_listener(self, name: Literal["value-changed", "triggered"], sender: "Component"):
        if self._listener_enabled:
            if name == "value-changed":
                self._on_value_changed(sender)
            elif name == "triggered":
                self._on_button_triggered(sender)
            else:
                raise ValueError(f"Unknown event name: {name}")

    def add_component(self, component: "Component") -> None:
        self._components.add(component)

    def find_component(self, component_type: type[C], name: str) -> C:
        # for c in self._components:
        #     if not isinstance(c, component_type):
        #         continue
        #     if c.get_name() == name:
        #         return c
        # raise ValueError(f"No component {component_type} {name} found",
        #                  [(type(c), c.get_name()) for c in self._components])
        c = self._components.find_by_name(name)
        assert isinstance(c, component_type), (name, component_type)
        return c

    def get_total_focus_count(self):
        return sum(c.focus_count() for c in self._components)

    def map_global_focus_index(self, global_focus_index: int) \
            -> "tuple[Component | None, int]":  # component and local focus index
        for i, c in enumerate(self._components.__iter__()):
            if global_focus_index < c.focus_count():
                local_focus_index = global_focus_index
                return c, local_focus_index
            global_focus_index -= c.focus_count()
        return None, 0

    @abstractmethod
    def create_background(self, window_size: ApplicationWindowSize) -> np.ndarray | None:
        # シーンが背景を生成する
        raise NotImplementedError()

    def get_focus_component(self) -> "Component | None":
        global_focus_index = self._global_focus_index
        return self.map_global_focus_index(global_focus_index)[0]

    def set_picture_in_picture(
            self,
            image: np.ndarray | None,
            height: int = None,
            width: int = None,
    ) -> None:
        if image is not None and height is None and width is None:
            height = 300
        self._pip.set_image(image, width=width, height=height)

    def render_ui(self, ctx: UIRenderingContext) -> UIRenderingContext:
        self._pip.set_screen_size(width=ctx.canvas.width, height=ctx.canvas.height)

        # picture in picture at bottom left
        if self._pip.is_ready():
            ctx.canvas.paste(
                im=self._pip.get_image(),
                pos=self._pip.get_offset(),
            )
            ctx.canvas.rectangle(
                pos=self._pip.get_offset(),
                size=self._pip.get_size(),
                color=ctx.style.border_normal,
            )

        # render components
        for component in self._components:
            rendering_result = component.render(ctx)
            ctx.top += rendering_result.height + 2

        return ctx

    def translate_onto_picture_in_picture(self, x: int, y: int) -> tuple[int, int] | None:
        return self._pip.translate_screen_to_local(x, y)

    def move_focus(self, delta: int) -> None:
        total_focus_count = self.get_total_focus_count()
        if total_focus_count:
            self._global_focus_index += delta
            self._global_focus_index %= total_focus_count

    def key_event(self, event: KeyEvent) -> bool:
        handled = False
        for component in self._components:
            if component.key_event(event):
                handled = True
                break
        if not handled:
            if event.enter:
                if event.key == Key.UP:
                    self.move_focus(-1)
                    handled = True
                elif event.key == Key.DOWN:
                    self.move_focus(1)
                    handled = True
        return handled

    def mouse_event(self, event: MouseEvent) -> bool:
        return False

    def load_event(self):
        pass

    def start_event(self):
        pass

    def stop_event(self):
        pass

    def unload_event(self):
        pass

    def update(self):
        pass
