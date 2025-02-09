from core.tk.component.component import Component
from core.tk.event import KeyEvent
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext, RenderingResult, Canvas
from core.tk.scene import Scene


class ButtonComponent(Component):
    def __init__(self, scene: Scene, text: str, *, name: str = None):
        super().__init__(scene, name=name)
        self._text = text.replace("\n", " ")

    def get_text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text.replace("\n", " ")

    def render(self, canvas: Canvas, ctx: UIRenderingContext) -> RenderingResult:
        text = " >>> " + self._text
        bg_color = ctx.style.edge_color if self.get_scene().get_focus_component() is self else None
        height = canvas.text(
            text=text,
            pos=(ctx.left, ctx.top),
            max_width=ctx.max_width,
            fg_color=ctx.style.fg_color,
            edge_color=ctx.style.edge_color,
            bg_color=bg_color,
        )
        return RenderingResult(
            height=height,
        )

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if self.get_scene().get_focus_component() is self:
                if event.key == Key.ENTER or event.key == Key.RIGHT:
                    self.get_scene().notify_listener("triggered", self)
                    return True
        return super().key_event(event)

    def focus_count(self) -> int:
        return 1
