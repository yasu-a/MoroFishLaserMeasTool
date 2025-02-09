from core.tk.component.component import Component
from core.tk.event import KeyEvent
from core.tk.rendering import UIRenderingContext, RenderingResult
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

    def render(self, ctx: UIRenderingContext) -> RenderingResult:
        height = ctx.canvas.text(
            text=self._text,
            pos=(ctx.left, ctx.top),
            max_width=ctx.max_width,
            fg_color=ctx.fg_color,
            edge_color=ctx.edge_color,
            bold=self._bold,
        )
        return RenderingResult(
            height=height,
        )

    def key_event(self, event: KeyEvent) -> bool:
        return super().key_event(event)

    def focus_count(self) -> int:
        return 0
