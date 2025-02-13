from typing import cast

import numpy as np

from camera_server import CaptureResult
from core.tk.app import ApplicationWindowSize
from core.tk.event import KeyEvent
from core.tk.global_state import get_app
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext
from core.tk.scene import Scene
from my_app import MyApplication


class MyScene(Scene):
    def __init__(self, *, is_stationed=False):
        super().__init__(is_stationed=is_stationed)
        self._is_visible = True

    def create_background(self, window_size: "ApplicationWindowSize") -> np.ndarray | None:
        app: MyApplication = cast(MyApplication, get_app())
        im = None

        last_capture_undistort: CaptureResult | None = app.last_capture_undistort
        if last_capture_undistort is not None:
            im = last_capture_undistort.frame

        if im is None:
            last_capture: CaptureResult | None = app.last_capture
            if last_capture is not None:
                im = last_capture.frame

        if im is None:
            return None
        else:
            return window_size.coerce(im.copy())

    def render_ui(self, ctx: UIRenderingContext) -> UIRenderingContext:
        if not self._is_visible:
            return ctx
        return super().render_ui(ctx)

    def key_event(self, event: KeyEvent) -> bool:
        if super().key_event(event):
            return True
        if event.down:
            if event.key == Key.TAB:
                self._is_visible = not self._is_visible
                return True
        return False
