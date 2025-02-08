import cv2

from core.tk.component.component import Component
from core.tk.event import KeyEvent
from core.tk.rendering import UIRenderingContext, RenderingResult, Canvas
from core.tk.scene import Scene


class LabelComponent(Component):
    def __init__(
            self,
            scene: Scene,
            text: str,
            bold=False,
            name: str = None,
    ):
        super().__init__(scene, name=name)
        self._text = text
        self._bold = bold

    def get_text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text

    def render(self, canvas: Canvas, ctx: UIRenderingContext) -> RenderingResult:
        lines = self._text.strip().split("\n")
        for i, line in enumerate(lines):
            x = ctx.left
            y = ctx.top + ctx.font_height * i + ctx.font_offset_y
            cv2.putText(
                canvas.im,
                line,
                (x, y),
                ctx.font,
                ctx.scale,
                ctx.style.fg_color,
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
