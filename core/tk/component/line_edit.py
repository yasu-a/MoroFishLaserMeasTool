import cv2

from core.tk.component.component import Component
from core.tk.event import KeyEvent
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext, RenderingResult, Canvas
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

    def render(self, canvas: Canvas, ctx: UIRenderingContext) -> RenderingResult:
        cv2.rectangle(
            canvas.im,
            (ctx.left + 20, ctx.top),
            (ctx.left + 300, ctx.top + ctx.font_height),
            ctx.style.fg_color,
            1,
        )
        cv2.putText(
            canvas.im,
            self._value,
            (ctx.left + 30, ctx.top + ctx.font_offset_y),
            ctx.font,
            ctx.scale,
            ctx.style.fg_color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        if self.get_scene().get_focus_component() is self:
            cv2.putText(
                canvas.im,
                ">",
                (ctx.left, ctx.top + ctx.font_offset_y),
                ctx.font,
                ctx.scale,
                ctx.style.fg_color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        return RenderingResult(
            height=ctx.font_height,
        )

    def key_event(self, event: KeyEvent) -> bool:
        if self.get_scene().get_focus_component() is self:
            if event.down:
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
