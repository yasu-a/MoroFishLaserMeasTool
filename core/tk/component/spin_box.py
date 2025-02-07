import cv2

from core.tk.app import Application
from core.tk.component.component import Component
from core.tk.event import KeyEvent
from core.tk.key import Key
from core.tk.rendering import RenderingContext, RenderingResult
from core.tk.scene import Scene


class SpinBoxComponent(Component):
    def __init__(
            self,
            app: "Application",
            scene: "Scene",
            min_value: int,
            max_value: int,
            value: int,
            name: str = None,
    ):
        super().__init__(app, scene, name=name)
        self._min_value = min_value
        self._max_value = max_value
        self._value = value

    def get_value(self):
        return self._value

    def set_value(self, value: int):
        self._value = max(self._min_value, min(self._max_value, value))

    def render(self, ctx: RenderingContext) -> RenderingResult:
        cv2.rectangle(
            ctx.canvas,
            (ctx.left + 20, ctx.top),
            (ctx.left + 100, ctx.top + ctx.font_height),
            ctx.color,
            1,
        )
        cv2.putText(
            ctx.canvas,
            str(self._value),
            (ctx.left + 30, ctx.top + ctx.font_offset_y),
            ctx.font,
            ctx.scale,
            ctx.color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        if self._scene.get_focus_component() is self:
            cv2.putText(
                ctx.canvas,
                ">",
                (ctx.left, ctx.top + ctx.font_offset_y),
                ctx.font,
                ctx.scale,
                ctx.color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        return RenderingResult(
            height=ctx.font_height,
        )

    def key_event(self, event: KeyEvent) -> bool:
        if self._scene.get_focus_component() is self:
            if event.down:
                if event.key == Key.RIGHT:
                    prev_value = self._value
                    self._value += 1
                    if self._value > self._max_value:
                        self._value = self._max_value
                    if prev_value != self._value:
                        self._scene.on_value_changed(self)
                    return True
                if event.key == Key.LEFT:
                    prev_value = self._value
                    self._value -= 1
                    if self._value < self._min_value:
                        self._value = self._min_value
                    if prev_value != self._value:
                        self._scene.on_value_changed(self)
                    return True
        return super().key_event(event)

    def focus_count(self) -> int:
        return 1
