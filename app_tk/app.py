from abc import ABC, abstractmethod
from collections import deque
from typing import Literal

import cv2
import numpy as np

from app_tk.component.toast import Toast
from app_tk.event import KeyEvent, MouseEvent
from app_tk.key import Key
from app_tk.cv2_handler import CV2KeyHandler, CV2MouseHandler
from app_tk.rendering import RenderingContext
from app_tk.scene import Scene


class Application(ABC):
    def __init__(
            self,
            cv2_wait_key_delay=10,
            ui_color: tuple[int, int, int] = (0, 150, 0),
    ):
        self._cv2_wait_key_delay = cv2_wait_key_delay
        self._ui_color = ui_color

        self._key_handler = CV2KeyHandler()
        self._mouse_handler = CV2MouseHandler()
        self._focus_index: dict[Scene, int] = {}
        self._scene_stack: deque[Scene] = deque()
        self._signals: set[str] = set()

        self._toast: Toast | None = None

    def send_signal(self, name: str) -> None:
        self._signals.add(name)

    def _check_signal(self, name: str) -> bool:
        flag = name in self._signals
        self._signals.discard(name)
        return flag

    def set_ui_color(self, ui_color: tuple[int, int, int]) -> None:
        self._ui_color = ui_color

    def get_ui_color(self) -> tuple[int, int, int]:
        return self._ui_color

    def move_to(self, scene: Scene):
        if self._scene_stack:
            self._scene_stack[-1].hide_event()
        scene.load_event()
        scene.show_event()
        self._scene_stack.append(scene)

    def go_back(self):
        if self._scene_stack[-1].is_stationed():
            return
        if self._scene_stack:
            scene = self._scene_stack.pop()
            scene.hide_event()
            scene.unload_event()
        if self._scene_stack:
            self._scene_stack[-1].show_event()

    def make_toast(
            self,
            toast_type: Literal["info", "error"],
            message: str,
    ) -> None:
        if toast_type == "info":
            creator = Toast.create_info
        elif toast_type == "error":
            creator = Toast.create_error
        else:
            assert False, toast_type

        self._toast = creator(
            app=self,
            scene=self.get_active_scene(),
            message=message,
        )

    def get_active_scene(self) -> Scene | None:
        return self._scene_stack[-1] if self._scene_stack else None

    def key_event(self, event: KeyEvent) -> bool:
        scene = self.get_active_scene()
        if scene is None:
            return False
        if scene.key_event(event):
            return True
        if event.down:
            if event.key == Key.ESCAPE:
                self.go_back()
                return True
        return False

    def mouse_event(self, event: MouseEvent) -> bool:
        scene = self.get_active_scene()
        if scene is None:
            return False
        if scene.mouse_event(event):
            return True
        return False

    @abstractmethod
    def canvas_size_hint(self) -> tuple[int, int]:
        raise NotImplementedError()

    def create_rendering_context(
            self,
            canvas: np.ndarray,
            min_width: int,
            min_height: int,
            max_width: int,
            max_height: int,
    ) -> RenderingContext:
        canvas_height, canvas_width = canvas.shape[:2]

        canvas_height_pad = max(0, min_height - canvas_height)
        canvas_width_pad = max(0, min_width - canvas_width)
        if canvas_height_pad > 0 or canvas_width_pad > 0:
            canvas = np.pad(
                canvas,
                (
                    (0, canvas_height_pad),
                    (0, canvas_width_pad),
                    (0, 0),
                )
            )

        canvas_height, canvas_width = canvas.shape[:2]
        max_width = min(max_width, canvas_width)
        max_height = min(max_height, canvas_height)

        rendering_ctx = RenderingContext(
            canvas=canvas,
            color=self._ui_color,
            max_width=max_width,
            font=cv2.FONT_HERSHEY_DUPLEX,
            scale=0.5,
            top=0,
            left=0,
        )
        _ = max_height

        return rendering_ctx

    def render(self) -> np.ndarray:
        scene = self.get_active_scene()
        if scene is not None:
            im_out = scene.render_canvas()
        else:
            im_out = None

        if im_out is None:
            w, h = self.canvas_size_hint()
            im_out = np.zeros((h, w, 3), dtype=np.uint8)

        canvas = im_out[...]
        rendering_ctx = self.create_rendering_context(
            canvas,
            min_width=1000,
            min_height=700,
            max_width=canvas.shape[1],
            max_height=canvas.shape[0],
        )

        if scene is not None:
            scene.render_ui(rendering_ctx)

        if self._toast is not None:
            self._toast.render(rendering_ctx)

        return rendering_ctx.canvas

    def do_event(self):
        scene = self.get_active_scene()

        # key events
        for evt in self._key_handler.cv2_wait_key_and_iter_key_events(self._cv2_wait_key_delay):
            self.key_event(evt)

        # mouse events
        for evt in self._mouse_handler.cv2_iter_mouse_events():
            self.mouse_event(evt)

        # remove expired toast
        if self._toast is not None:
            if self._toast.is_expired():
                self._toast = None

    def update(self):
        scene = self.get_active_scene()
        if scene is not None:
            scene.update()

    @abstractmethod
    def loop(self):
        raise NotImplementedError()
