from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from core.tk.component.global_state import register_app
from core.tk.cv2_handler import CV2KeyHandler, CV2MouseHandler
from core.tk.event import KeyEvent, MouseEvent
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext, Canvas
from core.tk.style import ApplicationUIStyle, _DEFAULT_STYLE

if TYPE_CHECKING:
    from core.tk.scene import Scene
    from core.tk.component.toast import Toast


@dataclass(frozen=True)
class ApplicationWindowSize:
    min_width: int
    min_height: int
    max_width: int
    max_height: int

    def coerce(self, im: np.ndarray) -> np.ndarray:
        src_h, src_w = im.shape[:2]
        h_pad, w_pad = 0, 0
        if src_h < self.min_height:
            h_pad = self.min_height - src_h
        if src_w < self.min_width:
            w_pad = self.min_width - src_w
        if h_pad != 0 or w_pad != 0:
            im = np.pad(
                im,
                ((0, h_pad), (0, w_pad), (0, 0)),
            )

        if src_h > self.max_height:
            im = im[:self.max_height, :, :]
        if src_w > self.max_width:
            im = im[:, :self.max_width, :]

        return im


@dataclass(slots=True)
class ApplicationParams:
    rendering_fps: float
    cv2_wait_key_delay: int
    window_size: ApplicationWindowSize
    style: ApplicationUIStyle


_DEFAULT_PARAMS = ApplicationParams(
    rendering_fps=30,
    cv2_wait_key_delay=10,
    window_size=ApplicationWindowSize(
        min_width=1000,
        min_height=700,
        max_width=1920,
        max_height=1080,
    ),
    style=_DEFAULT_STYLE,
)


class Application(ABC):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        register_app(instance)
        return instance

    def __init__(
            self,
    ):
        self._params = _DEFAULT_PARAMS

        self._key_handler = CV2KeyHandler(self._params.cv2_wait_key_delay)
        self._mouse_handler = CV2MouseHandler()
        self._focus_index: "dict[Scene, int]" = {}
        self._scene_stack: "deque[Scene]" = deque()
        self._signals: set[str] = set()

        self._toast: Toast | None = None

    def send_signal(self, name: str) -> None:
        self._signals.add(name)

    def _check_signal(self, name: str) -> bool:
        flag = name in self._signals
        self._signals.discard(name)
        return flag

    def move_to(self, scene: "Scene"):
        if self._scene_stack:
            self._scene_stack[-1].hide_event()
        scene.load_event()
        scene.show_event()
        self._scene_stack.append(scene)

    def move_back(self):
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
            toast: "Toast",
    ) -> None:
        self._toast = toast

    def get_active_scene(self) -> "Scene | None":
        return self._scene_stack[-1] if self._scene_stack else None

    def key_event(self, event: KeyEvent) -> bool:
        scene = self.get_active_scene()
        if scene is None:
            return False
        if scene.key_event(event):
            return True
        if event.down:
            if event.key == Key.ESCAPE:
                self.move_back()
                return True
        return False

    def mouse_event(self, event: MouseEvent) -> bool:
        scene = self.get_active_scene()
        if scene is None:
            return False
        if scene.mouse_event(event):
            return True
        return False

    def create_ui_rendering_context(self) -> UIRenderingContext:
        rendering_ctx = UIRenderingContext(
            style=self._params.style,
            font=cv2.FONT_HERSHEY_DUPLEX,
            scale=0.5,
            top=0,
            left=0,
        )

        return rendering_ctx

    def create_background_fallback(self) -> np.ndarray:
        # シーンがないとき，もしくはシーンが背景を生成できないときに背景を生成する
        return np.zeros(
            (self._params.window_size.min_height, self._params.window_size.min_height, 3),
            np.uint8,
        )

    def create_background(self) -> np.ndarray:
        scene = self.get_active_scene()
        im = None
        if scene is not None:
            im = scene.create_background(self._params.window_size)
        if im is None:
            im = self.create_background_fallback()
        return im

    def render(self) -> np.ndarray:
        im: np.ndarray = self.create_background()

        rendering_ctx = self.create_ui_rendering_context()

        scene = self.get_active_scene()
        if scene is not None:
            scene.render_ui(Canvas(im), rendering_ctx)

        if self._toast is not None:
            self._toast.render(Canvas(im), rendering_ctx)

        return im

    def do_event(self):
        # key events
        for evt in self._key_handler.cv2_wait_key_and_iter_key_events():
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
