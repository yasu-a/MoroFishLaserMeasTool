from core.tk.component.component import Component
from core.tk.event import KeyEvent
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext, RenderingResult
from core.tk.scene import Scene


class LineEditComponent(Component):
    def __init__(
            self,
            scene: "Scene",
            value: str = "",
            max_length: int = 25,
            name: str = None,
    ):
        super().__init__(scene, name=name)
        self._max_length = max_length
        self._value = value

    def get_value(self):
        return self._value

    def set_value(self, value: str) -> None:
        self._value = value[:self._max_length]
        self.get_scene().notify_listener("value-changed", self)

    def append_value(self, new_value: str) -> None:
        self.set_value(self._value + new_value)

    def pop_value(self) -> None:
        if self._value:
            self.set_value(self._value[:-1])

    def render(self, ctx: UIRenderingContext) -> RenderingResult:
        text = self._value
        bg_color = ctx.style.edge_color if self.get_scene().get_focus_component() is self else None
        height = ctx.canvas.text(
            text=text if text else ' ',
            pos=(ctx.left, ctx.top),
            max_width=ctx.max_width,
            fg_color=ctx.fg_color,
            edge_color=ctx.edge_color,
            bg_color=bg_color,
        )
        ctx.canvas.rectangle(
            pos=(ctx.left, ctx.top),
            size=(ctx.max_width, height),
            color=ctx.style.fg_color,
        )
        return RenderingResult(
            height=height,
        )

    def key_event(self, event: KeyEvent) -> bool:
        if self.get_scene().get_focus_component() is self:
            if event.enter:
                mapping = Key.printable_char_map()
                char = mapping.get((event.key, event.modifiers))
                if char is not None:
                    self.append_value(char)
                    return True
                elif event.key == Key.BACKSPACE:
                    self.pop_value()
                    return True
        return super().key_event(event)

    def focus_count(self) -> int:
        return 1
