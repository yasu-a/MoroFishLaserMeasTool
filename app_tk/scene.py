from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TypeVar

import cv2
import numpy as np

from app_tk.component.component import Component
from app_tk.event import KeyEvent, MouseEvent
from app_tk.key import Key
from app_tk.rendering import RenderingContext

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app_tk.app import Application


class SceneEventHandlers(ABC):
    def on_button_triggered(self, sender: Component) -> None:
        pass

    def on_value_changed(self, sender: Component) -> None:
        pass


C = TypeVar("C", bound=Component)


class Scene(SceneEventHandlers, ABC):
    def __init__(self, app: "Application", is_stationed=False):
        self._app = app
        self._components: list[Component] = []
        self._is_stationed = is_stationed  # disallow go-back
        self._global_focus_index = 0

        self._picture_in_picture: np.ndarray | None = None

    def get_app(self) -> "Application":
        return self._app

    def is_stationed(self) -> bool:
        return self._is_stationed

    def add_component(self, *args, **kwargs) -> None:
        # add_component(Component(app, scene, ...))
        #  or
        # add_component(Component, ...)
        if len(args) == 0 and isinstance(args[0], Component):
            self._components.append(args[0])
        else:
            component_cls, *args = args
            component = component_cls(self.get_app(), self, *args, **kwargs)
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
            -> tuple[Component | None, int]:  # component and local focus index
        for i, c in enumerate(self._components):
            if global_focus_index < c.focus_count():
                local_focus_index = global_focus_index
                return c, local_focus_index
            global_focus_index -= c.focus_count()
        return None, 0

    @abstractmethod
    def render_canvas(self) -> np.ndarray | None:
        raise NotImplementedError()

    def get_focus_component(self) -> Component | None:
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

    def render_ui(self, rendering_ctx: RenderingContext) -> RenderingContext:
        if self._picture_in_picture is not None:
            # picture in picture at bottom left
            h, w = rendering_ctx.canvas.shape[:2]
            im = self._picture_in_picture
            sub_h_small, sub_w_small = im.shape[:2]
            rendering_ctx.canvas[h - sub_h_small:, w - sub_w_small:] = im
            cv2.rectangle(
                rendering_ctx.canvas,
                (w - sub_w_small, h - sub_h_small),
                (w, h),
                (50, 50, 255),
                3,
                cv2.LINE_AA,
            )

        for component in self._components:
            rendering_result = component.render(rendering_ctx)
            rendering_ctx = replace(
                rendering_ctx,
                top=rendering_ctx.top + rendering_result.height + 5,
            )

        return rendering_ctx

    def move_focus(self, delta: int) -> None:
        total_focus_count = self.get_total_focus_count()
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

    def show_event(self):
        pass

    def hide_event(self):
        pass

    def unload_event(self):
        pass

    def update(self):
        pass
