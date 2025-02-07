import cv2

from app_tk.component.component import Component
from app_tk.event import KeyEvent
from app_tk.rendering import RenderingContext, RenderingResult

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app_tk.app import Application
    from app_tk.scene import Scene


class LabelComponent(Component):
    def __init__(
            self,
            app: "Application",
            scene: "Scene",
            text: str,
            bold=False,
            name: str = None,
    ):
        super().__init__(app, scene, name=name)
        self._text = text
        self._bold = bold

    def get_text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text

    def render(self, ctx: RenderingContext) -> RenderingResult:
        lines = self._text.strip().split("\n")
        for i, line in enumerate(lines):
            x = ctx.left
            y = ctx.top + ctx.font_height * i + ctx.font_offset_y
            cv2.putText(
                ctx.canvas,
                line,
                (x, y),
                ctx.font,
                ctx.scale,
                ctx.color,
                thickness=2 if self._bold else 1,
                lineType=cv2.LINE_AA,
            )
        return RenderingResult(
            height=ctx.font_height * len(lines),
        )

    def key_event(self, event: KeyEvent) -> bool:
        return super().key_event(event)

    def focus_count(self) -> int:
        return 0
