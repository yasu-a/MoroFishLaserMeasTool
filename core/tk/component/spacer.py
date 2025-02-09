from core.tk.component.component import Component
from core.tk.event import KeyEvent
from core.tk.rendering import UIRenderingContext, RenderingResult


class SpacerComponent(Component):
    def render(self, ctx: UIRenderingContext) -> RenderingResult:
        return RenderingResult(
            height=8,
        )

    def key_event(self, event: KeyEvent) -> bool:
        return super().key_event(event)

    def focus_count(self) -> int:
        return 0
