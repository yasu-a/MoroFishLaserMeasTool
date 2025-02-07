import time
from typing import TYPE_CHECKING

import cv2

from app_tk.component.component import Component
from app_tk.rendering import RenderingContext, RenderingResult

if TYPE_CHECKING:
    from app_tk.app import Application
    from app_tk.scene import Scene


class Toast(Component):
    def __init__(
            self,
            app: "Application",
            scene: "Scene",
            message: str,
            bg_color: tuple[int, int, int],
            fg_color: tuple[int, int, int],
    ):
        super().__init__(app, scene)
        self._message = message
        self._bg_color = bg_color
        self._fg_color = fg_color
        self._birthtime = time.monotonic()
        self._expired_in_seconds = max(1.5, min(5.0, len(self._message) / 20))

    def is_expired(self):
        return self._birthtime + self._expired_in_seconds < time.monotonic()

    FONT_SCALE = 1.5

    def render(self, ctx: RenderingContext) -> RenderingResult:
        cv2.rectangle(
            ctx.canvas,
            (0, int(ctx.canvas.shape[0] - ctx.font_height * self.FONT_SCALE)),
            (ctx.canvas.shape[1], ctx.canvas.shape[0]),
            self._bg_color,
            -1,
        )
        cv2.putText(
            ctx.canvas,
            self._message,
            (
                ctx.left,
                int(ctx.canvas.shape[0]
                    - ctx.font_height * self.FONT_SCALE
                    + ctx.font_offset_y * self.FONT_SCALE)
            ),
            ctx.font,
            ctx.scale * self.FONT_SCALE,
            self._fg_color,
            1,
            cv2.LINE_AA,
        )
        return RenderingResult(
            height=ctx.font_height,
        )

    def focus_count(self) -> int:
        return 0

    @classmethod
    def create_info(cls, app: "Application", scene: "Scene", message: str):
        return cls(
            app=app,
            scene=scene,
            message=message,
            bg_color=(150, 0, 0),
            fg_color=(255, 255, 255),
        )

    @classmethod
    def create_error(cls, app: "Application", scene: "Scene", message: str):
        return cls(
            app=app,
            scene=scene,
            message=message,
            bg_color=(0, 0, 150),
            fg_color=(255, 255, 255),
        )
