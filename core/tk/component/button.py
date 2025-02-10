from core.tk.component.component import Component
from core.tk.event import KeyEvent
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext, RenderingResult
from core.tk.scene import Scene


class ButtonComponent(Component):
    def __init__(self, scene: Scene, text: str, *, name: str = None):
        super().__init__(scene, name=name)
        self._text = text.replace("\n", " ")

    def get_text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text.replace("\n", " ")

    def render(self, ctx: UIRenderingContext) -> RenderingResult:
        text = " >>> " + self._text
        bg_color = ctx.style.edge_color if self.get_scene().get_focus_component() is self else None
        height = ctx.canvas.text(
            text=text,
            pos=(ctx.left, ctx.top),
            max_width=ctx.max_width,
            fg_color=ctx.fg_color,
            edge_color=ctx.edge_color,
            bg_color=bg_color,
        )
        return RenderingResult(
            height=height,
        )

    def click(self):
        self.get_scene().notify_listener("triggered", self)

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if event.key == Key.ENTER:
                if self.get_scene().get_focus_component() is self:
                    self.click()
                    return True
        return super().key_event(event)

    def focus_count(self) -> int:
        return 1
