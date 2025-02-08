import time
from typing import Literal

import cv2

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
        self._birthtime = time.monotonic()
        self._expired_in_seconds = max(1.5, min(5.0, len(self._message) / 20))

    def is_expired(self):
        return self._birthtime + self._expired_in_seconds < time.monotonic()

    FONT_SCALE = 1.5

    def render(self, canvas: Canvas, ctx: UIRenderingContext) -> RenderingResult:
        cv2.rectangle(
            canvas.im,
            (0, int(canvas.im.shape[0] - ctx.font_height * self.FONT_SCALE)),
            (canvas.im.shape[1], canvas.im.shape[0]),
            (
                ctx.style.toast_info_bg_color if self._message_type == "info"
                else ctx.style.toast_error_bg_color
            ),
            -1,
        )
        cv2.putText(
            canvas.im,
            self._message,
            (
                ctx.left,
                int(canvas.im.shape[0]
                    - ctx.font_height * self.FONT_SCALE
                    + ctx.font_offset_y * self.FONT_SCALE)
            ),
            ctx.font,
            ctx.scale * self.FONT_SCALE,
            (
                ctx.style.toast_info_fg_color if self._message_type == "info"
                else ctx.style.toast_error_fg_color
            ),
            1,
            cv2.LINE_AA,
        )
        return RenderingResult(
            height=ctx.font_height,
        )

    def focus_count(self) -> int:
        return 0
