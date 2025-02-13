import time
from functools import lru_cache
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
        expired_in_seconds = max(5.0, min(8.0, len(self._message) / 10))
        self._time_of_death = self._time_of_birth + expired_in_seconds

        self._margin = 30  # content margin
        self._padding = 5  # content text padding

    def is_expired(self):
        return self._time_of_death < time.monotonic()

    def _get_animation_factor(self) -> float:
        now = time.monotonic()
        a = min(1.0, (now - self._time_of_birth) / 0.2)
        b = min(1.0, (self._time_of_death - now) / 0.1)
        return min(a, b)

    FONT_SCALE = 1.5

    @lru_cache(maxsize=3)
    def _render_message_box(
            self,
            ctx: UIRenderingContext,
            canvas_width: int,
    ):
        buf = np.zeros((100, canvas_width - self._margin * 2, 3), np.uint8)
        buf[:] = ctx.bg_color

        buf_canvas = Canvas(buf, ctx)
        height = buf_canvas.text(
            text=self._message,
            pos=(self._padding, self._padding),
            max_width=buf.shape[1] - self._padding * 2,
            max_height=buf.shape[0] - self._padding * 2,
            fg_color=ctx.fg_color,
            scale=2,
        )
        buf = buf[:height + self._padding * 2]

        return buf

    def render(self, ctx: UIRenderingContext) -> RenderingResult:
        with ctx.enter_sub_context(self._message_type):
            buf = self._render_message_box(ctx, ctx.canvas.width)
            y_start = ctx.canvas.height
            y_last = ctx.canvas.height - (buf.shape[0] + self._margin * 2)
            y = y_start + int((y_last - y_start) * self._get_animation_factor())
            x = self._margin
            ctx.canvas.paste(
                im=buf,
                pos=(x, y),
            )

        return RenderingResult(height=0)

    def focus_count(self) -> int:
        return 0
