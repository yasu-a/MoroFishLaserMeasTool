from core.tk.component.component import Component
from core.tk.event import KeyEvent
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext, RenderingResult
from core.tk.scene import Scene


class CheckBoxComponent(Component):
    def __init__(
            self,
            scene: Scene,
            text: str,
            value: bool = False,
            *,
            name: str = None,
    ):
        super().__init__(scene, name=name)
        self._text = text.replace("\n", " ")
        self._value = value

    def get_text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text.replace("\n", " ")

    def get_value(self) -> bool:
        return self._value

    def set_value(self, value: bool) -> None:
        self._value = value
        self.get_scene().notify_listener("value-changed", self)

    def render(self, ctx: UIRenderingContext) -> RenderingResult:
        text = f" [{'X' if self._value else ' '}] " + self._text
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

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if self.get_scene().get_focus_component() is self:
                if event.key == Key.SPACE or event.key == Key.ENTER:
                    self._value = not self._value
                    self.get_scene().notify_listener("value-changed", self)
                    return True
        return super().key_event(event)

    def focus_count(self) -> int:
        return 1
