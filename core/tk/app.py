from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from core.tk.cv2_handler import CV2KeyHandler, CV2MouseHandler
from core.tk.event import KeyEvent, MouseEvent
from core.tk.font_renderer import CharPrinter, ConsolaCharFactory
from core.tk.global_state import register_app
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext, Canvas
from core.tk.style import ApplicationUIStyle, _DEFAULT_STYLE

if TYPE_CHECKING:
    from core.tk.scene import Scene
    from core.tk.component.toast import Toast
    from core.tk.dialog import Dialog


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
    char_printer: CharPrinter


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
    char_printer=CharPrinter(ConsolaCharFactory()),
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

        self._toast: "Toast| None" = None

        self._dialog: "Dialog| None" = None

    def send_signal(self, name: str) -> None:
        self._signals.add(name)

    def _check_signal(self, name: str) -> bool:
        flag = name in self._signals
        self._signals.discard(name)
        return flag

    def move_to(self, scene: "Scene"):
        if self._scene_stack:
            self._scene_stack[-1].stop_event()
        scene.load_event()
        scene.start_event()
        self._scene_stack.append(scene)

    def move_back(self):
        if self._scene_stack[-1].is_stationed():
            return
        if self._scene_stack:
            scene = self._scene_stack.pop()
            scene.stop_event()
            scene.unload_event()
        if self._scene_stack:
            self._scene_stack[-1].start_event()

    def make_toast(self, toast: "Toast") -> None:
        self._toast = toast

    def withdraw_toast(self) -> None:
        if self._toast is not None:
            self._toast = None

    def show_dialog(self, dialog: "Dialog") -> None:
        if self.get_active_scene() is not None:
            self.get_active_scene().stop_event()
        self._dialog = dialog
        self._dialog.load_event()
        self._dialog.start_event()

    def close_dialog(self) -> None:
        if self._dialog is not None:
            self._dialog.stop_event()
            self._dialog.unload_event()
            self._dialog = None
        if self.get_active_scene() is not None:
            self.get_active_scene().start_event()

    def get_active_scene(self) -> "Scene | None":
        return self._scene_stack[-1] if self._scene_stack else None

    def get_active_dialog(self) -> "Dialog | None":
        return self._dialog

    def key_event(self, event: KeyEvent) -> bool:
        if self._dialog is not None:
            if self._dialog.key_event(event):
                return True
            return False
        else:
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
        if self._dialog is not None:
            if self._dialog.mouse_event(event):
                return True
            return False
        else:
            scene = self.get_active_scene()
            if scene is None:
                return False
            if scene.mouse_event(event):
                return True
            return False

    def create_ui_rendering_context(
            self,
            ctx_name: Literal["normal", "dialog", "toast"],
    ) -> UIRenderingContext:
        style = self._params.style
        if ctx_name == "normal":
            fg_colors = {"normal": style.fg_color}
            bg_colors = {}
            edge_colors = {"normal": style.edge_color}
        elif ctx_name == "dialog":
            fg_colors = {
                "info": style.message_info_fg_color,
                "error": style.message_error_fg_color,
                "query": style.query_fg_color,
            }
            bg_colors = {
                "info": style.message_info_bg_color,
                "error": style.message_error_bg_color,
                "query": style.query_bg_color,
            }
            edge_colors = {}
        elif ctx_name == "toast":
            fg_colors = {
                "info": style.toast_info_fg_color,
                "error": style.toast_error_fg_color,
            }
            bg_colors = {
                "info": style.toast_info_bg_color,
                "error": style.toast_error_bg_color,
            }
            edge_colors = {}
        else:
            assert False, ctx_name

        rendering_ctx = UIRenderingContext(
            style=self._params.style,
            fg_colors=fg_colors,
            bg_colors=bg_colors,
            edge_colors=edge_colors,
            char_printer=self._params.char_printer,
            font_size=15,
            top=0,
            left=0,
            max_width=400,
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

        scene = self.get_active_scene()
        if scene is not None:
            ctx = self.create_ui_rendering_context(ctx_name="normal")
            with ctx.offer_canvas_image(im):
                with ctx.enter_sub_context("normal"):
                    scene.render_ui(ctx)

        if self._toast is not None:
            ctx = self.create_ui_rendering_context(ctx_name="toast")
            with ctx.offer_canvas_image(im):
                self._toast.render(ctx)

        if self._dialog is not None:
            ctx = self.create_ui_rendering_context(ctx_name="dialog")
            buf = self._dialog.create_background(self._params.window_size)
            with ctx.offer_canvas_image(buf):
                self._dialog.render_ui(ctx)
            bg_color = tuple(buf[-1, -1, :])
            mask = np.count_nonzero(buf - bg_color, axis=2) > 0
            x_idx = np.where(np.count_nonzero(mask, axis=0) != 0)[0]
            y_idx = np.where(np.count_nonzero(mask, axis=1) != 0)[0]
            x1, x2 = x_idx.min(), x_idx.max()
            y1, y2 = y_idx.min(), y_idx.max()
            buf = buf[y1:y2, x1:x2, :]
            padding = 10
            buf_pad = np.zeros(
                (buf.shape[0] + padding * 2, buf.shape[1] + padding * 2, 3),
                np.uint8,
            )
            buf_pad[...] = bg_color
            buf_pad[padding:-padding, padding:-padding, :] = buf
            im_canvas = Canvas(im, ctx)
            im_canvas.paste(
                im=buf_pad,
                pos=(
                    im.shape[1] // 2 - (x2 - x1) // 2,
                    im.shape[0] // 2 - (y2 - y1) // 2,
                ),
            )

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
                self.withdraw_toast()

    def update(self):
        scene = self.get_active_scene()
        if scene is not None:
            scene.update()

        dialog = self.get_active_dialog()
        if dialog is not None:
            dialog.update()

    @abstractmethod
    def loop(self):
        raise NotImplementedError()
