import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal
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


C = TypeVar("C", bound="Component")


class Scene(SceneEventHandlers, ABC):
    def __init__(self, *, is_stationed=False):
        self._components: list[Component] = []
        self._is_stationed = is_stationed  # disallow go-back
        self._global_focus_index = 0

        self._picture_in_picture: np.ndarray | None = None

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
        self._components.append(component)

    def find_component(self, component_type: type[C], name: str) -> C:
        for c in self._components:
            if not isinstance(c, component_type):
                continue
            if c.get_name() == name:
                return c
        raise ValueError(f"No component {component_type} {name} found",
                         [(type(c), c.get_name()) for c in self._components])

    def get_total_focus_count(self):
        return sum(c.focus_count() for c in self._components)

    def map_global_focus_index(self, global_focus_index: int) \
            -> "tuple[Component | None, int]":  # component and local focus index
        for i, c in enumerate(self._components):
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

    def set_picture_in_picture(self, image: np.ndarray | None, height: int = 200) -> None:
        if image is None:
            self._picture_in_picture = None
        else:
            sub_h, sub_w = image.shape[:2]
            sub_h_small = height
            sub_w_small = int(sub_w / sub_h * sub_h_small)
            im = cv2.resize(image, (sub_w_small, sub_h_small))
            self._picture_in_picture = im

    def render_ui(self, ctx: UIRenderingContext) -> UIRenderingContext:
        # picture in picture at bottom left
        if self._picture_in_picture is not None:
            margin = 20

            canvas = ctx.canvas
            h_canvas, w_canvas = canvas.height, canvas.width
            im = self._picture_in_picture
            h_im, w_im = im.shape[:2]
            canvas.paste(
                im=im,
                pos=(w_canvas - w_im - margin, h_canvas - h_im - margin)
            )
            canvas.rectangle(
                pos=(w_canvas - w_im - margin, h_canvas - h_im - margin),
                size=(w_im, h_im),
                color=ctx.style.border_normal,
            )

        # render components
        for component in self._components:
            rendering_result = component.render(ctx)
            ctx.top += rendering_result.height + 2

        return ctx

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
            if event.down:
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
