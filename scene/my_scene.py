from typing import cast

import numpy as np

from camera_server import CaptureResult
from core.tk.app import ApplicationWindowSize
from core.tk.component.global_state import get_app
from core.tk.scene import Scene
from my_app import MyApplication


class MyScene(Scene):
    def create_background(self, window_size: "ApplicationWindowSize") -> np.ndarray | None:
        last_capture: CaptureResult | None = cast(MyApplication, get_app()).last_capture
        if last_capture is None:
            return None
        else:
            return window_size.coerce(last_capture.frame.copy())
