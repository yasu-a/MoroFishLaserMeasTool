import time
from typing import Literal

import numpy as np

from core.tk.component.component import Component
from core.tk.rendering import UIRenderingContext, RenderingResult, Canvas
from core.tk.scene import Scene


class Toast(Component):
    def __init__(
            self,
            scene: Scene,
            message_type: Literal["info", "error"],
            message: str,
    ):
        super().__init__(scene)
        self._message_type = message_type
        self._message = message
        self._time_of_birth = time.monotonic()
        expired_in_seconds = max(1.5, min(5.0, len(self._message) / 20))
        self._time_of_death = self._time_of_birth + expired_in_seconds

    def is_expired(self):
        return self._time_of_death < time.monotonic()

    def _get_animation_factor(self) -> float:
        now = time.monotonic()
        a = min(1.0, (now - self._time_of_birth) / 0.2)
        b = min(1.0, (self._time_of_death - now) / 0.1)
        return min(a, b)

    FONT_SCALE = 1.5

    def render(self, ctx: UIRenderingContext) -> RenderingResult:
        with ctx.enter_sub_context(self._message_type):
            canvas: Canvas = ctx.canvas

            margin = 5
            x1, x2 = margin, canvas.width - margin

            buf = np.zeros((100, x2 - x1, 3), np.uint8)
            buf[:] = ctx.bg_color

            buf_canvas = Canvas(buf, ctx)
            padding = 5
            height = buf_canvas.text(
                text=self._message,
                pos=(padding, padding),
                max_width=buf.shape[1] - padding * 2,
                max_height=buf.shape[0] - padding * 2,
                fg_color=ctx.fg_color,
                scale=2,
            )
            buf = buf[:height + padding * 2]

            y_start = canvas.height
            y_last = canvas.height - (height + padding * 2 + margin * 2)
            y1 = y_start + int((y_last - y_start) * self._get_animation_factor())
            canvas.paste(
                im=buf,
                pos=(x1, y1),
            )

        return RenderingResult(height=0)

    def focus_count(self) -> int:
        return 0
