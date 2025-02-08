import cv2

from core.tk.component.component import Component
from core.tk.event import KeyEvent
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext, RenderingResult, Canvas
from core.tk.scene import Scene


class SpinBoxComponent(Component):
    def __init__(
            self,
            scene: Scene,
            value: int = 0,
            *,
            min_value: int,
            max_value: int,
            name: str = None,
    ):
        super().__init__(scene, name=name)
        self._min_value = min_value
        self._max_value = max_value
        self._value = max(self._min_value, min(self._max_value, value))

    def get_value(self) -> int:
        return self._value

    def set_value(self, value: int) -> None:
        old_value = self._value
        self._value = max(self._min_value, min(self._max_value, value))
        if self._value != old_value:
            self.get_scene().notify_listener("value-changed", self)

    def increment_value(self, *, delta: int = 1) -> None:
        self.set_value(self._value + delta)

    def decrement_value(self, *, delta: int = 1) -> None:
        self.increment_value(delta=-delta)

    def render(self, canvas: Canvas, ctx: UIRenderingContext) -> RenderingResult:
        cv2.rectangle(
            canvas.im,
            (ctx.left + 20, ctx.top),
            (ctx.left + 100, ctx.top + ctx.font_height),
            ctx.style.fg_color,
            1,
        )
        cv2.putText(
            canvas.im,
            str(self._value),
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
                if event.key == Key.RIGHT:
                    self.increment_value()
                    return True
                if event.key == Key.LEFT:
                    self.decrement_value()
                    return True
        return super().key_event(event)

    def focus_count(self) -> int:
        return 1
