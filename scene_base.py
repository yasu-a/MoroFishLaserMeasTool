import numpy as np

from app_tk.scene import Scene
from camera_server import CaptureResult
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_app import MyApplication


class MyScene(Scene):
    def get_app(self) -> "MyApplication":
        # noinspection PyTypeChecker
        return super().get_app()

    def render_canvas(self) -> np.ndarray | None:
        last_capture: CaptureResult | None = self.get_app().last_capture
        if last_capture is None:
            return None
        else:
            return last_capture.frame.copy()
